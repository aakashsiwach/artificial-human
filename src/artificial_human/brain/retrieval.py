from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from artificial_human.memory.repository import MemoryRecord


_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]{2,}")


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def rank_texts(
    query: str,
    documents: Sequence[tuple[str, str]],
    *,
    limit: int = 5,
) -> list[tuple[str, float]]:
    """Rank text with a small BM25-style lexical scorer; this is not vector search."""
    if limit < 1:
        raise ValueError("limit must be positive")
    query_terms = set(tokenize(query))
    if not query_terms or not documents:
        return []

    tokenized = [(document_id, tokenize(text)) for document_id, text in documents]
    document_frequency = Counter(
        term for term in query_terms
        for _, terms in tokenized
        if term in set(terms)
    )
    total = len(tokenized)
    average_length = sum(len(terms) for _, terms in tokenized) / max(total, 1)
    k1 = 1.5
    b = 0.75
    ranked: list[tuple[str, float]] = []

    for document_id, terms in tokenized:
        frequencies = Counter(terms)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if frequency == 0:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log(1 + (total - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (
                1 - b + b * len(terms) / max(average_length, 1.0)
            )
            score += inverse_frequency * (frequency * (k1 + 1) / denominator)
        if score > 0:
            ranked.append((document_id, score))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:limit]


@dataclass(frozen=True, slots=True)
class MemoryMatch:
    memory: MemoryRecord
    score: float
    channels: tuple[str, ...] = ("lexical",)
    explanation: dict[str, Any] = field(default_factory=dict)


class MemoryRetriever:
    """Retrieves relevant, consent-visible memories without modifying them."""

    def retrieve(
        self,
        query: str,
        memories: Sequence[MemoryRecord],
        *,
        limit: int = 5,
    ) -> list[MemoryMatch]:
        ranked = rank_texts(
            query,
            [(item.id, f"{item.kind.value} {item.content}") for item in memories],
            limit=limit,
        )
        by_id = {item.id: item for item in memories}
        return [
            MemoryMatch(memory=by_id[memory_id], score=score)
            for memory_id, score in ranked
        ]
