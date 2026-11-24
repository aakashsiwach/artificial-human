from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from artificial_human.memory.repository import MemoryRecord


@dataclass(frozen=True, slots=True)
class ConsolidationProposal:
    source_memory_ids: tuple[str, ...]
    summary: str
    status: str = "pending_review"
    id: str = field(default_factory=lambda: str(uuid4()))


class SleepSystem:
    """Prepares review queues; it never silently rewrites or deletes memories."""

    def prepare(self, memories: list[MemoryRecord]) -> ConsolidationProposal | None:
        if len(memories) < 2:
            return None
        kinds = sorted({item.kind.value for item in memories})
        return ConsolidationProposal(
            source_memory_ids=tuple(item.id for item in memories),
            summary=(
                f"Review {len(memories)} memories across {', '.join(kinds)}. "
                "A future consolidation model may suggest a synthesis; owner approval is required."
            ),
        )
