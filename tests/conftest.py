"""Shared test doubles.

``ConceptEmbedding`` is a TEST-ONLY fake. It maps hand-picked synonyms onto
shared dimensions so semantic-retrieval *plumbing* can be tested without
downloading a model. It is not a semantic embedding and is never used outside
the test suite; real semantic behaviour is measured by the benchmark.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import pytest

from artificial_human.memory.embeddings import (
    EmbeddingError,
    EmbeddingModelInfo,
    normalise_vector,
)

_CONCEPTS: tuple[tuple[str, ...], ...] = (
    ("boil", "boils", "boiling", "vapor", "vapour", "steam"),
    ("water", "h2o"),
    ("temperature", "heat", "celsius", "degrees", "hot"),
    ("work", "works", "job", "employer", "company", "engineer"),
    ("live", "lives", "city", "relocated", "home"),
    ("allergic", "allergy", "allergies", "peanuts"),
    ("drink", "tea", "coffee"),
    ("trip", "travel", "flew", "lisbon", "hotel"),
    ("editor", "vim", "code"),
)


class ConceptEmbedding:
    """Deterministic test-only fake embedding (see module docstring)."""

    def __init__(
        self, *, model_id: str = "test-concepts", locality: str = "in_process",
        fail: bool = False,
    ) -> None:
        self.model_id = model_id
        self.locality = locality
        self.fail = fail
        self.calls: list[list[str]] = []

    def describe(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            backend="test", model_id=self.model_id, dimensions=len(_CONCEPTS) + 1,
            version="1", locality=self.locality,  # type: ignore[arg-type]
        )

    def _vector(self, text: str) -> list[float]:
        words = set(re.findall(r"[a-z0-9]+", text.lower()))
        values = [float(len(words & set(group))) for group in _CONCEPTS]
        values.append(0.05)  # bias dimension keeps every vector non-zero
        return normalise_vector(values, len(values))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.fail:
            raise EmbeddingError("simulated backend outage")
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if self.fail:
            raise EmbeddingError("simulated backend outage")
        return self._vector(text)


@pytest.fixture
def concept_embedding() -> ConceptEmbedding:
    return ConceptEmbedding()
