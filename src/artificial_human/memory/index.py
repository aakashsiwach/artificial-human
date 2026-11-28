"""Consent-aware semantic indexing of long-term memory records."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from artificial_human.memory.embeddings import (
    EXTERNAL_EMBEDDING_CONSENT,
    EmbeddingError,
    EmbeddingModelInfo,
    EmbeddingProvider,
    decode_vector,
    encode_vector,
)
from artificial_human.memory.repository import (
    MemoryRecord,
    content_fingerprint,
)
from artificial_human.memory.contracts import MemoryStore

# Log identifiers and error classes only: never memory content, vectors, or keys.
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IndexOutcome:
    memory_id: str
    status: str  # indexed | failed | skipped | disabled
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ReindexReport:
    space: str | None
    indexed: int
    failed: int
    skipped: int
    removed_other_spaces: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "space": self.space, "indexed": self.indexed, "failed": self.failed,
            "skipped": self.skipped, "removed_other_spaces": self.removed_other_spaces,
        }


class SemanticIndex:
    """Builds and queries vectors for eligible memories.

    Eligibility = the record is visible under active ``personal_memory``
    consent; for a remote backend the owner must also hold an active
    ``external_embedding`` consent. A failure is recorded per memory and never
    modifies or removes the memory itself, so lexical retrieval keeps working.
    """

    def __init__(self, repository: MemoryStore, provider: EmbeddingProvider | None) -> None:
        self.repository = repository
        self.provider = provider
        self._last_error: str | None = None

    @property
    def configured(self) -> bool:
        return self.provider is not None

    def model_info(self) -> EmbeddingModelInfo | None:
        """Model identity, or ``None`` if disabled or the model cannot load."""
        if self.provider is None:
            return None
        try:
            info = self.provider.describe()
        except EmbeddingError as error:
            self._last_error = type(error).__name__
            logger.warning("embedding backend unavailable: %s", type(error).__name__)
            return None
        self._last_error = None
        return info

    def _permitted(self, owner_id: str, info: EmbeddingModelInfo) -> bool:
        if info.locality != "remote":
            return True
        return self.repository.has_active_consent(owner_id, EXTERNAL_EMBEDDING_CONSENT)

    def _embed_and_store(
        self, owner_id: str, info: EmbeddingModelInfo, records: Sequence[MemoryRecord]
    ) -> list[IndexOutcome]:
        assert self.provider is not None
        try:
            vectors = self.provider.embed_documents([item.content for item in records])
        except EmbeddingError as error:
            outcomes = []
            for item in records:
                self.repository.record_index_failure(
                    owner_id=owner_id, memory_id=item.id, space=info.space,
                    error_type=type(error).__name__,
                )
                logger.warning(
                    "embedding failed memory_id=%s error=%s", item.id, type(error).__name__
                )
                outcomes.append(IndexOutcome(item.id, "failed", type(error).__name__))
            return outcomes
        outcomes = []
        for item, vector in zip(records, vectors, strict=True):
            stored = self.repository.store_embedding(
                owner_id=owner_id, memory_id=item.id, space=info.space,
                model_id=info.model_id, model_version=info.version,
                dimensions=info.dimensions, content_hash=content_fingerprint(item.content),
                vector=encode_vector(vector),
            )
            outcomes.append(
                IndexOutcome(item.id, "indexed")
                if stored
                else IndexOutcome(item.id, "skipped", "memory changed, deleted, or not eligible")
            )
        return outcomes

    def index_memory(self, owner_id: str, memory_id: str) -> IndexOutcome:
        """Incrementally index one record after it is created or corrected."""
        info = self.model_info()
        if info is None:
            reason = self._last_error or "no embedding backend configured"
            return IndexOutcome(memory_id, "disabled", reason)
        if not self._permitted(owner_id, info):
            return IndexOutcome(memory_id, "skipped", "external_embedding consent required")
        try:
            record = self.repository.get(owner_id, memory_id)
        except KeyError:
            return IndexOutcome(memory_id, "skipped", "memory is not visible under consent")
        return self._embed_and_store(owner_id, info, [record])[0]

    def reindex(self, owner_id: str, *, full: bool = False, batch_size: int = 32) -> ReindexReport:
        """Index missing or stale vectors; ``full`` re-embeds everything.

        Vectors from any other embedding space (a previous model) are removed
        so only one comparable space is used for retrieval.
        """
        info = self.model_info()
        if info is None:
            return ReindexReport(None, 0, 0, 0, 0)
        if not self._permitted(owner_id, info):
            return ReindexReport(info.space, 0, 0, len(self.repository.iter_visible(owner_id)), 0)
        removed = self.repository.delete_embeddings_outside_space(owner_id, info.space)
        records = self.repository.iter_visible(owner_id)
        existing = {} if full else self.repository.embeddings(owner_id, info.space)
        pending = [item for item in records if item.id not in existing]
        indexed = failed = skipped = 0
        for start in range(0, len(pending), batch_size):
            for outcome in self._embed_and_store(owner_id, info, pending[start:start + batch_size]):
                if outcome.status == "indexed":
                    indexed += 1
                elif outcome.status == "failed":
                    failed += 1
                else:
                    skipped += 1
        return ReindexReport(info.space, indexed, failed, skipped, removed)

    def status(self, owner_id: str) -> dict[str, Any]:
        info = self.model_info()
        counts = self.repository.index_counts(owner_id, info.space if info else None)
        if self.provider is None:
            mode = "lexical_only"
        elif info is None:
            mode = "lexical_only_backend_unavailable"
        elif not self._permitted(owner_id, info):
            mode = "lexical_only_external_consent_missing"
        else:
            mode = "hybrid"
        return {
            "retrieval_mode": mode,
            "embedding_model": info.public() if info else None,
            "backend_error": self._last_error,
            "eligible_memories": counts.eligible,
            "indexed_memories": counts.indexed,
            "missing_or_stale": max(counts.eligible - counts.indexed, 0),
            "failed": counts.failed,
            "vectors_from_other_models": counts.other_spaces,
            "failures": list(counts.failures),
        }

    def query_vector(self, owner_id: str, query: str) -> tuple[EmbeddingModelInfo, list[float]] | None:
        info = self.model_info()
        if info is None or self.provider is None or not self._permitted(owner_id, info):
            return None
        try:
            return info, self.provider.embed_query(query)
        except EmbeddingError as error:
            logger.warning("query embedding failed error=%s", type(error).__name__)
            self._last_error = type(error).__name__
            return None

    def similarities(
        self, owner_id: str, info: EmbeddingModelInfo, vector: Sequence[float],
        memory_ids: Sequence[str],
    ) -> dict[str, float]:
        """Exact (brute-force) cosine similarity against stored unit vectors."""
        stored = self.repository.embeddings(owner_id, info.space, memory_ids)
        return {
            memory_id: math.sumprod(vector, decode_vector(item.vector))
            for memory_id, item in stored.items()
            if item.dimensions == info.dimensions
        }
