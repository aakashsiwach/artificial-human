"""Offline controls for owner-reviewed event-schema induction."""

from __future__ import annotations

from functools import partial
from hashlib import sha256
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
from artificial_human.cognition.schema_extraction import SchemaExtractionSystem, SchemaProposal


PROBE_COUNT = 9
EVENT_TYPES = ("order.created", "payment.authorized", "order.completed")


def _add_event(
    repository: MemoryRepository,
    *,
    owner_id: str,
    consent_id: str,
    event_type: str,
    attributes: dict[str, str | int | bool],
    occurred_at: str,
    content: str,
) -> MemoryRecord:
    return repository.add(
        owner_id=owner_id,
        kind=MemoryKind.EPISODIC,
        content=content,
        source="synthetic schema-induction fixture",
        consent_id=consent_id,
        occurred_at=occurred_at,
        metadata={"event_type": event_type, "attributes": attributes},
    )


def _record_sequence(
    replay: EpisodicReplayMemory,
    *,
    owner_id: str,
    session_id: str,
    records: tuple[MemoryRecord, ...],
) -> None:
    owner_digest = sha256(owner_id.encode("utf-8")).hexdigest()[:16]
    replay.record_episode(
        [
            EpisodicEvent(record.id, record.occurred_at or record.created_at)
            for record in records
        ],
        context_id=f"owner-{owner_digest}.session-{session_id}",
    )


def _persist_schema(
    repository: MemoryRepository,
    owner_id: str,
    proposal: SchemaProposal,
    consent_id: str,
) -> str:
    record = repository.add_schema_memory(
        owner_id=owner_id,
        source_sequences=proposal.source_sequences,
        event_types=proposal.event_types,
        schema=proposal.schema,
        content=proposal.summary,
        source=f"owner-approved schema extraction:{proposal.id}",
        consent_id=consent_id,
        metadata={"schema_proposal_id": proposal.id},
    )
    return record.id


def run_schema_extraction_suite(seed: int = DEFAULT_SEED) -> SuiteResult:
    """Measure recurrent schema recall, slot induction, controls, and approval safety."""
    detection_hits: list[float] = []
    slot_hits: list[float] = []
    constant_hits: list[float] = []
    sensitive_exclusion_hits: list[float] = []
    singleton_false_positives: list[float] = []
    contradiction_false_positives: list[float] = []
    silent_write_violations: list[float] = []
    approved_provenance_hits: list[float] = []
    by_probe: dict[str, dict[str, dict[str, Any]]] = {}

    for index in range(PROBE_COUNT):
        owner_id = f"schema-eval-owner-{index:02d}"
        repository = MemoryRepository(":memory:")
        consent = repository.grant_consent(owner_id)
        replay = EpisodicReplayMemory()
        extractor = SchemaExtractionSystem()

        example_providers = ("stripe", "razorpay", "adyen")
        example_amounts = (100 + index, 175 + index, 250 + index)
        for example_index, (provider, amount) in enumerate(
            zip(example_providers, example_amounts, strict=True)
        ):
            minute = 1 + example_index * 5
            session = f"recurring-{example_index}"
            records = (
                _add_event(
                    repository,
                    owner_id=owner_id,
                    consent_id=consent.id,
                    event_type=EVENT_TYPES[0],
                    attributes={"currency": "INR", "region": "IN", "api_key": "excluded"},
                    occurred_at=f"2026-11-{index + 1:02d}T10:{minute:02d}:00+00:00",
                    content=f"Order created probe {index} example {example_index}",
                ),
                _add_event(
                    repository,
                    owner_id=owner_id,
                    consent_id=consent.id,
                    event_type=EVENT_TYPES[1],
                    attributes={"provider": provider, "amount": amount, "password": "excluded"},
                    occurred_at=f"2026-11-{index + 1:02d}T10:{minute + 1:02d}:00+00:00",
                    content=f"Payment authorized probe {index} example {example_index}",
                ),
                _add_event(
                    repository,
                    owner_id=owner_id,
                    consent_id=consent.id,
                    event_type=EVENT_TYPES[2],
                    attributes={"currency": "INR"},
                    occurred_at=f"2026-11-{index + 1:02d}T10:{minute + 2:02d}:00+00:00",
                    content=f"Order completed probe {index} example {example_index}",
                ),
            )
            _record_sequence(
                replay, owner_id=owner_id, session_id=session, records=records
            )

        # A two-event pattern seen only once is a negative control.
        singleton = (
            _add_event(
                repository, owner_id=owner_id, consent_id=consent.id,
                event_type="support.ticket_opened", attributes={},
                occurred_at=f"2026-11-{index + 1:02d}T11:01:00+00:00",
                content=f"Singleton ticket opened {index}",
            ),
            _add_event(
                repository, owner_id=owner_id, consent_id=consent.id,
                event_type="support.ticket_closed", attributes={},
                occurred_at=f"2026-11-{index + 1:02d}T11:02:00+00:00",
                content=f"Singleton ticket closed {index}",
            ),
        )
        _record_sequence(
            replay, owner_id=owner_id, session_id="singleton", records=singleton
        )

        # A repeated pattern with explicit source conflicts is a second negative control.
        for example_index in range(3):
            minute = 20 + example_index * 3
            pair = (
                _add_event(
                    repository, owner_id=owner_id, consent_id=consent.id,
                    event_type="deployment.started", attributes={"environment": "staging"},
                    occurred_at=f"2026-11-{index + 1:02d}T12:{minute:02d}:00+00:00",
                    content=f"Conflicted deployment start {index}-{example_index}",
                ),
                _add_event(
                    repository, owner_id=owner_id, consent_id=consent.id,
                    event_type="deployment.completed", attributes={"environment": "staging"},
                    occurred_at=f"2026-11-{index + 1:02d}T12:{minute + 1:02d}:00+00:00",
                    content=f"Conflicted deployment completion {index}-{example_index}",
                ),
            )
            repository.link_contradiction(
                owner_id, pair[0].id, pair[1].id, note="Synthetic contradiction control"
            )
            _record_sequence(
                replay,
                owner_id=owner_id,
                session_id=f"conflict-{example_index}",
                records=pair,
            )

        before_preview = len(repository.list(owner_id))
        proposals = extractor.prepare_from_episodes(
            owner_id=owner_id,
            context_prefix=f"owner-{sha256(owner_id.encode('utf-8')).hexdigest()[:16]}.session-",
            episodic_memory=replay,
            repository=repository,
        )
        after_preview = len(repository.list(owner_id))
        recurring = [item for item in proposals if item.event_types == EVENT_TYPES]
        singleton = any(
            item.event_types == ("support.ticket_opened", "support.ticket_closed")
            for item in proposals
        )
        contradiction = any(
            item.event_types == ("deployment.started", "deployment.completed")
            for item in proposals
        )

        detection_hits.append(float(len(recurring) == 1))
        singleton_false_positives.append(float(singleton))
        contradiction_false_positives.append(float(contradiction))
        silent_write_violations.append(float(before_preview != after_preview))

        slot_ok = False
        constant_ok = False
        sensitive_ok = False
        provenance_ok = False
        memory_id: str | None = None
        if recurring:
            schema = recurring[0].schema
            steps = schema["steps"]
            slot_ok = (
                steps[1]["attributes"].get("provider", {}).get("kind") == "slot"
                and steps[1]["attributes"].get("amount", {}).get("kind") == "slot"
                and "stripe" not in str(schema)
                and "razorpay" not in str(schema)
                and "adyen" not in str(schema)
            )
            constant_ok = (
                steps[0]["attributes"].get("currency")
                == {"kind": "constant", "value": "INR"}
            )
            sensitive_ok = "api_key" not in str(schema) and "password" not in str(schema)

            reviewed, memory_id = extractor.approve_with(
                owner_id,
                recurring[0].id,
                reviewer=owner_id,
                note="Synthetic benchmark owner approval",
                persist=partial(_persist_schema, repository, owner_id),
            )
            derived = repository.get(owner_id, memory_id)
            provenance_ok = (
                reviewed.status == "approved"
                and reviewed.schema_memory_id == memory_id
                and derived.kind is MemoryKind.PROCEDURAL
                and derived.epistemic_status == "inferred"
                and derived.metadata.get("schema_source_sequences")
                == [list(sequence) for sequence in recurring[0].source_sequences]
            )
        slot_hits.append(float(slot_ok))
        constant_hits.append(float(constant_ok))
        sensitive_exclusion_hits.append(float(sensitive_ok))
        approved_provenance_hits.append(float(provenance_ok))
        by_probe[f"probe_{index + 1:02d}"] = {
            "generalisation": {
                "recurrent_schema_detected": detection_hits[-1],
                "variable_attributes_become_slots": slot_hits[-1],
                "stable_attributes_remain_constants": constant_hits[-1],
                "sensitive_attributes_excluded": sensitive_exclusion_hits[-1],
                "proposal_count": len(proposals),
            },
            "negative_controls": {
                "singleton_false_positive": singleton_false_positives[-1],
                "contradiction_false_positive": contradiction_false_positives[-1],
            },
            "review": {
                "preview_silent_write_violation": silent_write_violations[-1],
                "approved_schema_has_provenance": approved_provenance_hits[-1],
                "approved_schema_memory_created": float(memory_id is not None),
            },
        }
        repository.close()

    metrics = {
        "recurrent_schema_detection_rate": fmean(detection_hits),
        "variable_attribute_slot_rate": fmean(slot_hits),
        "stable_attribute_constant_rate": fmean(constant_hits),
        "sensitive_attribute_exclusion_rate": fmean(sensitive_exclusion_hits),
        "singleton_false_positive_rate": fmean(singleton_false_positives),
        "contradiction_false_positive_rate": fmean(contradiction_false_positives),
        "preview_silent_write_violation_rate": fmean(silent_write_violations),
        "approved_schema_provenance_rate": fmean(approved_provenance_hits),
    }
    checks = [
        check(
            "recurrent owner-annotated event sequences produce schemas",
            metrics["recurrent_schema_detection_rate"] == 1.0,
            f"Detection rate={metrics['recurrent_schema_detection_rate']:.3f}",
        ),
        check(
            "varying attributes become typed slots without copying their values",
            metrics["variable_attribute_slot_rate"] == 1.0,
            f"Variable-slot rate={metrics['variable_attribute_slot_rate']:.3f}",
        ),
        check(
            "stable attributes remain explicit constants",
            metrics["stable_attribute_constant_rate"] == 1.0,
            f"Constant-retention rate={metrics['stable_attribute_constant_rate']:.3f}",
        ),
        check(
            "sensitive attribute keys are excluded",
            metrics["sensitive_attribute_exclusion_rate"] == 1.0,
            f"Sensitive-key exclusion rate={metrics['sensitive_attribute_exclusion_rate']:.3f}",
        ),
        check(
            "single-session patterns do not generalize",
            metrics["singleton_false_positive_rate"] == 0.0,
            f"Singleton false-positive rate={metrics['singleton_false_positive_rate']:.3f}",
        ),
        check(
            "explicitly contradicted source events are excluded",
            metrics["contradiction_false_positive_rate"] == 0.0,
            f"Contradiction false-positive rate={metrics['contradiction_false_positive_rate']:.3f}",
        ),
        check(
            "schema preview never writes persistent memory",
            metrics["preview_silent_write_violation_rate"] == 0.0,
            f"Silent-write violation rate={metrics['preview_silent_write_violation_rate']:.3f}",
        ),
        check(
            "approved procedural schemas retain example provenance",
            metrics["approved_schema_provenance_rate"] == 1.0,
            f"Approved provenance rate={metrics['approved_schema_provenance_rate']:.3f}",
        ),
    ]
    return SuiteResult(
        suite="schema-extraction",
        title="Owner-annotated event-schema induction and generalisation",
        evidence_category="synthetic structured-event benchmark",
        splits={"tuning": 0, "eval": PROBE_COUNT},
        systems={"schema_induction": metrics},
        by_condition=by_probe,
        checks=checks,
        findings=[
            (
                "Schema extraction uses explicitly supplied event types and primitive "
                "attributes; it does not infer event labels from raw text."
            ),
            (
                "Attribute values that vary across examples become typed slots, while "
                "values shared by every example remain constants."
            ),
            (
                "The suite uses project-authored synthetic examples and does not establish "
                "generalisation quality on real owner workflows."
            ),
        ],
        extra={
            "probe_count": PROBE_COUNT,
            "seed": seed,
            "minimum_distinct_sessions": 2,
            "maximum_sequence_length": 4,
            "owner_approval_required": True,
            "preview_persists_memory": False,
        },
    )
