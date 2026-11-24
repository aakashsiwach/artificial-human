from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AttentionCandidate:
    """A candidate competing for limited cognitive attention."""

    reference_id: str
    kind: str
    relevance: float
    salience: float = 0.0


@dataclass(frozen=True, slots=True)
class AttendedItem:
    reference_id: str
    kind: str
    weight: float


class AttentionSystem:
    """Selects a bounded set of relevant, salient items.

    Relevance is supplied by retrieval; salience must be explicit metadata.
    The weighted score is an engineering heuristic, not a neural activation
    model or a claim to reproduce biological attention.
    """

    def select(
        self,
        candidates: list[AttentionCandidate],
        *,
        limit: int = 5,
    ) -> list[AttendedItem]:
        if limit < 1:
            raise ValueError("Attention limit must be positive")
        scored: list[AttendedItem] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.reference_id in seen:
                continue
            if not 0.0 <= candidate.relevance <= 1.0:
                raise ValueError("Candidate relevance must be between 0 and 1")
            if not 0.0 <= candidate.salience <= 1.0:
                raise ValueError("Candidate salience must be between 0 and 1")
            seen.add(candidate.reference_id)
            weight = 0.8 * candidate.relevance + 0.2 * candidate.salience
            scored.append(
                AttendedItem(
                    reference_id=candidate.reference_id,
                    kind=candidate.kind,
                    weight=round(weight, 4),
                )
            )
        scored.sort(key=lambda item: (-item.weight, item.reference_id))
        return scored[:limit]
