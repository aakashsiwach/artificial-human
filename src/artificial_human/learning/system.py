from __future__ import annotations

from typing import Any

from artificial_human.memory.repository import (
    LearningExample,
    MemoryRepository,
)


class LearningSystem:
    """Collects owner demonstrations and exports only explicitly approved examples.

    This prepares supervised data. It does not fine-tune a model or infer identity
    traits automatically.
    """

    def __init__(self, repository: MemoryRepository) -> None:
        self.repository = repository

    def capture(
        self, *, owner_id: str, task_category: str, input_text: str,
        target_text: str, source: str, consent_id: str,
    ) -> LearningExample:
        return self.repository.add_training_example(
            owner_id=owner_id, task_category=task_category, input_text=input_text,
            target_text=target_text, source=source, consent_id=consent_id,
        )

    def review(
        self, *, owner_id: str, example_id: str, decision: str,
        reviewer: str, note: str = "",
    ) -> None:
        changed = self.repository.review_training_example(
            owner_id, example_id, decision=decision, reviewer=reviewer, note=note
        )
        if not changed:
            raise KeyError(example_id)

    def examples(
        self, owner_id: str, *, status: str | None = None, limit: int = 100
    ) -> list[LearningExample]:
        return self.repository.list_training_examples(owner_id, status=status, limit=limit)

    def export_approved_corpus(self, owner_id: str) -> list[dict[str, Any]]:
        return self.repository.export_training_corpus(owner_id)
