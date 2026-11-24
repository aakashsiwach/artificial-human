"""Replaceable text-embedding backends for semantic memory.

The memory domain model never depends on a particular backend. A backend
declares its identity (backend, model, dimensions, version) and its *locality*:

* ``in_process`` — runs inside this Python process (fastembed / ONNX Runtime).
* ``loopback`` — an HTTP service on this machine (for example Ollama).
* ``remote`` — any other host. Remote backends are refused unless the operator
  sets ``ARTIFICIAL_HUMAN_EMBEDDING_ALLOW_REMOTE=true`` and uses HTTPS, and the
  indexer additionally requires the owner's ``external_embedding`` consent.

No backend is enabled by default: without configuration memory retrieval is
lexical-only. Embeddings are dense distributed representations produced by a
pretrained language model; they are not a model of cortical representations.
"""

from __future__ import annotations

import math
import os
import sys
import threading
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

import httpx

Locality = Literal["in_process", "loopback", "remote"]
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
EXTERNAL_EMBEDDING_CONSENT = "external_embedding"


class EmbeddingError(RuntimeError):
    """An embedding backend could not produce vectors. Messages never include input text."""


class EmbeddingUnavailableError(EmbeddingError):
    """The configured backend cannot be used (missing package, model, or policy)."""


@dataclass(frozen=True, slots=True)
class EmbeddingModelInfo:
    backend: str
    model_id: str
    dimensions: int
    version: str
    locality: Locality

    @property
    def space(self) -> str:
        """Vectors are comparable only within the same space."""
        return f"{self.backend}:{self.model_id}:{self.dimensions}"

    def public(self) -> dict[str, object]:
        return {
            "backend": self.backend, "model_id": self.model_id,
            "dimensions": self.dimensions, "version": self.version,
            "locality": self.locality, "space": self.space,
        }


class EmbeddingProvider(Protocol):
    def describe(self) -> EmbeddingModelInfo:
        """Return model identity; may load the model on first call."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed stored memory text."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a retrieval query."""


def normalise_vector(values: Sequence[float], dimensions: int) -> list[float]:
    """Validate a vector and scale it to unit length so cosine equals dot product."""
    if len(values) != dimensions:
        raise EmbeddingError(
            f"Embedding backend returned {len(values)} dimensions; expected {dimensions}"
        )
    floats = [float(value) for value in values]
    if not all(math.isfinite(value) for value in floats):
        raise EmbeddingError("Embedding backend returned a non-finite value")
    norm = math.sqrt(math.sumprod(floats, floats))
    if norm == 0.0:
        raise EmbeddingError("Embedding backend returned a zero vector")
    return [value / norm for value in floats]


def encode_vector(values: Sequence[float]) -> bytes:
    """Store as little-endian float32 for a compact, portable BLOB."""
    packed = array("f", values)
    if sys.byteorder != "little":
        packed.byteswap()
    return packed.tobytes()


def decode_vector(blob: bytes) -> array[float]:
    unpacked = array("f")
    unpacked.frombytes(blob)
    if sys.byteorder != "little":
        unpacked.byteswap()
    return unpacked


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Dot product of unit vectors."""
    return math.sumprod(left, right)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


class FastEmbedProvider:
    """Local, in-process ONNX embeddings through the optional ``fastembed`` package.

    The model weights are downloaded once into ``cache_dir`` (only weights are
    fetched; no memory content leaves the process). With ``offline=True`` only
    an already-cached model is used.
    """

    def __init__(
        self,
        model: str = DEFAULT_FASTEMBED_MODEL,
        *,
        cache_dir: str | None = None,
        offline: bool = False,
        threads: int | None = None,
        query_prefix: str = "",
    ) -> None:
        self.model = model
        self.cache_dir = cache_dir
        self.offline = offline
        self.threads = threads
        self.query_prefix = query_prefix
        self._engine: Any = None
        self._info: EmbeddingModelInfo | None = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._engine is not None:
                return self._engine
            try:
                import fastembed
                from fastembed import TextEmbedding
            except ImportError as error:
                raise EmbeddingUnavailableError(
                    "The fastembed backend requires the optional dependency: "
                    "pip install 'artificial-human[semantic]'"
                ) from error
            supported = {
                item["model"]: item for item in TextEmbedding.list_supported_models()
            }
            if self.model not in supported:
                raise EmbeddingUnavailableError(f"fastembed does not support model {self.model!r}")
            kwargs: dict[str, Any] = {"model_name": self.model, "cache_dir": self.cache_dir}
            if self.threads:
                kwargs["threads"] = self.threads
            if self.offline:
                kwargs["local_files_only"] = True
            try:
                engine = TextEmbedding(**kwargs)
            except Exception as error:  # third-party loader raises broadly
                raise EmbeddingUnavailableError(
                    f"Could not load embedding model {self.model!r} ({type(error).__name__})"
                ) from error
            self._info = EmbeddingModelInfo(
                backend="fastembed", model_id=self.model,
                dimensions=int(supported[self.model]["dim"]),
                version=f"fastembed-{fastembed.__version__}", locality="in_process",
            )
            self._engine = engine
            return engine

    def describe(self) -> EmbeddingModelInfo:
        self._load()
        assert self._info is not None
        return self._info

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        engine = self._load()
        info = self.describe()
        try:
            raw = list(engine.embed(list(texts)))
        except Exception as error:  # ONNX runtime errors vary by platform
            raise EmbeddingError(f"Local embedding failed ({type(error).__name__})") from error
        return [normalise_vector(vector.tolist(), info.dimensions) for vector in raw]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts) if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.query_prefix + text])[0]


class OpenAICompatibleEmbeddingProvider:
    """``POST {base_url}/embeddings`` adapter for loopback services such as Ollama.

    Loopback HTTP is allowed. Any other host is *remote*: it must use HTTPS and
    must be explicitly allowed with ``allow_remote=True``.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dimensions: int,
        api_key: str = "",
        allow_remote: bool = False,
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("Embedding base_url must be an absolute HTTP or HTTPS URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Embedding base_url cannot contain credentials, query, or fragment")
        if not model.strip():
            raise ValueError("An embedding model identifier is required")
        if dimensions < 1:
            raise ValueError("Embedding dimensions must be positive")
        locality: Locality = "loopback" if parsed.hostname in LOCAL_HOSTS else "remote"
        if locality == "remote":
            if not allow_remote:
                raise EmbeddingUnavailableError(
                    "Remote embedding endpoints are disabled. Set "
                    "ARTIFICIAL_HUMAN_EMBEDDING_ALLOW_REMOTE=true to opt in explicitly."
                )
            if parsed.scheme != "https":
                raise ValueError("Remote embedding endpoints must use HTTPS")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._info = EmbeddingModelInfo(
            backend="openai_compatible", model_id=model, dimensions=dimensions,
            version="api", locality=locality,
        )
        self._client = httpx.Client(timeout=timeout_seconds, transport=transport)

    def describe(self) -> EmbeddingModelInfo:
        return self._info

    def _request(self, texts: Sequence[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = self._client.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json={"model": self.model, "input": list(texts)},
            )
        except httpx.RequestError as error:
            raise EmbeddingError("Could not reach the embedding endpoint") from error
        if response.is_error:
            raise EmbeddingError(f"Embedding endpoint returned HTTP {response.status_code}")
        try:
            data = sorted(response.json()["data"], key=lambda item: item["index"])
            vectors = [item["embedding"] for item in data]
        except (ValueError, KeyError, TypeError) as error:
            raise EmbeddingError("Embedding endpoint returned an invalid response") from error
        if len(vectors) != len(texts):
            raise EmbeddingError("Embedding endpoint returned the wrong number of vectors")
        return [normalise_vector(vector, self._info.dimensions) for vector in vectors]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._request(texts) if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self._request([text])[0]

    def close(self) -> None:
        self._client.close()


def embedding_provider_from_env() -> EmbeddingProvider | None:
    """Build the configured backend; ``None`` means lexical-only retrieval.

    ``ARTIFICIAL_HUMAN_EMBEDDING_BACKEND``: ``none`` (default), ``fastembed``,
    or ``openai_compatible``.
    """
    backend = os.getenv("ARTIFICIAL_HUMAN_EMBEDDING_BACKEND", "none").strip().lower()
    if backend in ("", "none", "off", "disabled"):
        return None
    if backend == "fastembed":
        threads = os.getenv("ARTIFICIAL_HUMAN_EMBEDDING_THREADS", "").strip()
        return FastEmbedProvider(
            os.getenv("ARTIFICIAL_HUMAN_EMBEDDING_MODEL", "").strip() or DEFAULT_FASTEMBED_MODEL,
            cache_dir=os.getenv("ARTIFICIAL_HUMAN_EMBEDDING_CACHE_DIR", "").strip()
            or ".artificial-human/models",
            offline=_env_flag("ARTIFICIAL_HUMAN_EMBEDDING_OFFLINE"),
            threads=int(threads) if threads else None,
            query_prefix=os.getenv("ARTIFICIAL_HUMAN_EMBEDDING_QUERY_PREFIX", ""),
        )
    if backend == "openai_compatible":
        dimensions = os.getenv("ARTIFICIAL_HUMAN_EMBEDDING_DIMENSIONS", "").strip()
        if not dimensions:
            raise ValueError(
                "ARTIFICIAL_HUMAN_EMBEDDING_DIMENSIONS is required for openai_compatible"
            )
        return OpenAICompatibleEmbeddingProvider(
            base_url=os.getenv("ARTIFICIAL_HUMAN_EMBEDDING_BASE_URL", "").strip()
            or "http://127.0.0.1:11434/v1",
            model=os.getenv("ARTIFICIAL_HUMAN_EMBEDDING_MODEL", "").strip(),
            dimensions=int(dimensions),
            api_key=os.getenv("ARTIFICIAL_HUMAN_EMBEDDING_API_KEY", ""),
            allow_remote=_env_flag("ARTIFICIAL_HUMAN_EMBEDDING_ALLOW_REMOTE"),
        )
    raise ValueError(f"Unknown ARTIFICIAL_HUMAN_EMBEDDING_BACKEND: {backend!r}")
