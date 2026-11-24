"""Standard ranked-retrieval metrics (binary and graded relevance)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def recall_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    if not relevant:
        raise ValueError("recall is undefined without relevant items")
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Standard P@K: relevant hits in the top K divided by K (not by results returned)."""
    if k < 1:
        raise ValueError("k must be positive")
    return len(set(ranked[:k]) & set(relevant)) / k


def reciprocal_rank(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    for position, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """nDCG with exponential gain ``2**grade - 1`` and log2 position discount."""
    if not relevant:
        raise ValueError("nDCG is undefined without relevant items")

    def gain(grade: int, position: int) -> float:
        return (2**grade - 1) / math.log2(position + 1)

    dcg = sum(
        gain(relevant.get(item, 0), position)
        for position, item in enumerate(ranked[:k], start=1)
    )
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(gain(grade, position) for position, grade in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
