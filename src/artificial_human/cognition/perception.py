from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from artificial_human.brain.retrieval import tokenize


Modality = Literal["language"]


@dataclass(frozen=True, slots=True)
class Percept:
    """A normalized input representation at the sensory-to-cognition boundary."""

    id: str
    content: str
    modality: Modality
    source: str
    observed_at: str
    tokens: tuple[str, ...]


class PerceptionSystem:
    """Converts explicitly supplied input into a structured percept.

    This first implementation accepts language only. It does not access ambient
    microphones, cameras, files, or other sensors.
    """

    def perceive(
        self,
        content: str,
        *,
        source: str = "user",
        modality: Modality = "language",
    ) -> Percept:
        normalized = content.strip()
        if not normalized:
            raise ValueError("Perception requires non-empty input")
        if not source.strip():
            raise ValueError("Perception source is required")
        return Percept(
            id=str(uuid4()),
            content=normalized,
            modality=modality,
            source=source,
            observed_at=datetime.now(UTC).isoformat(),
            tokens=tuple(tokenize(normalized)),
        )
