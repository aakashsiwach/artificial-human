"""Offline controls for replay-driven, owner-reviewed memory consolidation."""

from __future__ import annotations

from statistics import fmean
from typing import Any

from artificial_human.cognition.benchmark.common import (
    DEFAULT_SEED,
    SuiteResult,
    check,
)
from artificial_human.cognition.episodic_replay import EpisodicEvent, EpisodicReplayMemory
from artificial_human.core.models import MemoryKind
from artificial_human.memory.repository import MemoryRecord, MemoryRepository
from artificial_human.sleep.consolidation import SleepSystem


PROBE_COUNT = 9


def run_sleep_consolidation_suite(seed: int = DEFAULT_SEED) -> SuiteResult:
    """Detect recurring directed event pairs, reject one-offs/conflicts, require review."""
    recurrence_hits: list[float] = []
    singleton_false_positives: list[float] = []
    contradiction_false_positives: list[float] = []
    approval_provenance: list[float] = []
    silent_write_violations: list[float] = []
    by_probe: dict[str, dict[str, dict[str, Any]]] = {}

    for index in range(PROBE_COUNT):
        owner_id = f"consolidation-owner-{index:02d}"
        repository = MemoryRepository(":memory:")
        consent = repository.grant_consent(owner_id)
        replay = EpisodicReplayMemory()
        sleep = SleepSystem()

        def add_event(
            name: str,
            minute: int,
            *,
            repo: MemoryRepository = repository,
            owner: str = owner_id,
            sequence_index: int = index,
            grant_id: str = consent.id,
        ) -> MemoryRecord:
            return repo.add(
                owner_id=owner,
                kind=MemoryKind.EPISODIC,
                content=f"Project event {name} in controlled sequence {sequence_index}",
                source="synthetic owner event fixture",
                consent_id=grant_id,
                occurred_at=f"2026-11-{sequence_index + 1:02d}T10:{minute:02d}:00+00:00",
            )

        recurring_a = add_event("recurring-a", 1)
        recurring_b = add_event("recurring-b", 2)
        singleton_a = add_event("singleton-a", 3)
        singleton_b = add_event("singleton-b", 4)
        conflict_a = add_event("conflict-a", 5)
        conflict_b = add_event("conflict-b", 6)
        repository.link_contradiction(
            owner_id, conflict_a.id, conflict_b.id, note="Synthetic conflict control"
        )

        def record_pair(
            first: MemoryRecord,
            second: MemoryRecord,
            sessions: tuple[str, ...],
            *,
            replay_buffer: EpisodicReplayMemory = replay,
            sequence_index: int = index,
        ) -> None:
            for session in sessions:
                replay_buffer.record_episode(
                    [
                        EpisodicEvent(first.id, first.occurred_at or first.created_at),
                        EpisodicEvent(second.id, second.occurred_at or second.created_at),
                    ],
                    context_id=f"owner-{sequence_index:04d}.session-{session}",
                )

        record_pair(recurring_a, recurring_b, ("a", "b"))
        record_pair(singleton_a, singleton_b, ("a",))
        record_pair(conflict_a, conflict_b, ("a", "b"))
        before_count = len(repository.list(owner_id))
        proposals = sleep.prepare_from_episodes(
            owner_id=owner_id,
            context_prefix=f"owner-{index:04d}.session-",
            episodic_memory=replay,
            repository=repository,
        )
        after_preview_count = len(repository.list(owner_id))

        matching = [
            item for item in proposals
            if item.source_memory_ids == (recurring_a.id, recurring_b.id)
        ]
        singleton_match = any(
            item.source_memory_ids == (singleton_a.id, singleton_b.id)
            for item in proposals
        )
        conflict_match = any(
            item.source_memory_ids == (conflict_a.id, conflict_b.id)
            for item in proposals
        )
        recurrence_hits.append(float(len(matching) == 1))
        singleton_false_positives.append(float(singleton_match))
        contradiction_false_positives.append(float(conflict_match))
        silent_write_violations.append(float(after_preview_count != before_count))

        approved_memory_id: str | None = None
        provenance_ok = False
        if matching:
            derived = repository.add_consolidated_memory(
                owner_id=owner_id,
                source_memory_ids=matching[0].source_memory_ids,
                content=matching[0].summary,
                source=f"owner-approved sleep consolidation:{matching[0].id}",
                consent_id=consent.id,
                metadata={"consolidation_proposal_id": matching[0].id},
            )
            approved_memory_id = derived.id
            reviewed = sleep.review(
                owner_id,
                matching[0].id,
                decision="approved",
                reviewer=owner_id,
                note="Synthetic owner-review benchmark approval",
                consolidated_memory_id=derived.id,
            )
            provenance_ok = (
                reviewed.status == "approved"
                and reviewed.consolidated_memory_id == derived.id
                and derived.kind is MemoryKind.SEMANTIC
                and derived.epistemic_status == "inferred"
                and derived.metadata.get("consolidation_source_memory_ids")
                == [recurring_a.id, recurring_b.id]
            )
        approval_provenance.append(float(provenance_ok))
        by_probe[f"probe_{index + 1:02d}"] = {
            "recurrence": {
                "expected_pattern_proposed": recurrence_hits[-1],
                "recurrence_sessions": 2 if matching else 0,
                "proposal_count": len(proposals),
            },
            "controls": {
                "singleton_false_positive": singleton_false_positives[-1],
                "contradiction_false_positive": contradiction_false_positives[-1],
            },
            "review": {
                "silent_write": silent_write_violations[-1],
                "approved_inferred_memory_has_provenance": approval_provenance[-1],
                "approved_memory_created": float(approved_memory_id is not None),
            },
        }

    recurrence_rate = fmean(recurrence_hits)
    singleton_rate = fmean(singleton_false_positives)
    conflict_rate = fmean(contradiction_false_positives)
    provenance_rate = fmean(approval_provenance)
    silent_write_rate = fmean(silent_write_violations)
    checks = [
        check(
            "recurring adjacent transitions across distinct sessions are proposed",
            recurrence_rate == 1.0,
            f"Recurring-pattern proposal rate={recurrence_rate:.3f}",
        ),
        check(
            "single-session patterns are not proposed",
            singleton_rate == 0.0,
            f"Singleton false-positive rate={singleton_rate:.3f}",
        ),
        check(
            "explicit contradictions are excluded from consolidation",
            conflict_rate == 0.0,
            f"Contradiction false-positive rate={conflict_rate:.3f}",
        ),
        check(
            "preview never writes a synthesized memory",
            silent_write_rate == 0.0,
            f"Silent-write violation rate={silent_write_rate:.3f}",
        ),
        check(
            "approved consolidation is stored as inferred with source provenance",
            provenance_rate == 1.0,
            f"Approved-memory provenance rate={provenance_rate:.3f}",
        ),
    ]
    return SuiteResult(
        suite="sleep-consolidation",
        title="Replay-driven pattern detection and owner-reviewed consolidation",
        evidence_category="synthetic software-mechanism benchmark",
        splits={"tuning": 0, "eval": PROBE_COUNT},
        systems={
            "replay_pattern_detection": {
                "recurring_pattern_proposal_rate": recurrence_rate,
                "singleton_false_positive_rate": singleton_rate,
                "contradiction_false_positive_rate": conflict_rate,
            },
            "owner_review": {
                "silent_write_violation_rate": silent_write_rate,
                "approved_inferred_provenance_rate": provenance_rate,
            },
        },
        by_condition=by_probe,
        checks=checks,
        findings=[
            (
                "The consolidation detector proposes a directed adjacent-event pair only "
                "after it appears in two distinct owner/session contexts."
            ),
            (
                "Preview is non-mutating. A separate owner-approved repository operation "
                "creates an inferred semantic memory with source-memory provenance."
            ),
            (
                "These fixed synthetic controls do not establish a real-world schema, "
                "human sleep replay, or biological systems consolidation."
            ),
        ],
        extra={
            "probe_count": PROBE_COUNT,
            "seed": seed,
            "minimum_distinct_sessions": 2,
            "owner_approval_required": True,
            "persistent_memory_created_during_preview": False,
        },
    )
