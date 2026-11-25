from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pytest

from artificial_human.cognition.episodic_replay import EpisodicEvent, EpisodicReplayMemory
from artificial_human.cognition.schema_extraction import SchemaExtractionSystem
from artificial_human.core.models import MemoryKind
from artificial_human.memory.repository import (
    MemoryNotFoundError,
    MemoryRecord,
    MemoryRepository,
)


OWNER = "schema-owner"
PREFIX = "owner-0123456789abcdef.session-"
EVENT_TYPES = ("order.created", "payment.authorized", "order.completed")


def _add_event(
    repository: MemoryRepository,
    *,
    consent_id: str,
    event_type: str,
    attributes: dict[str, object],
    minute: int,
    content: str,
) -> MemoryRecord:
    return repository.add(
        owner_id=OWNER,
        kind=MemoryKind.EPISODIC,
        content=content,
        source="owner-authored structured event",
        consent_id=consent_id,
        occurred_at=f"2026-11-25T10:{minute:02d}:00+00:00",
        metadata={"event_type": event_type, "attributes": attributes},
    )


def _record_example(
    repository: MemoryRepository,
    replay: EpisodicReplayMemory,
    consent_id: str,
    session: str,
    *,
    provider: str,
    amount: int,
    first_minute: int,
) -> tuple[MemoryRecord, MemoryRecord, MemoryRecord]:
    records = (
        _add_event(
            repository,
            consent_id=consent_id,
            event_type=EVENT_TYPES[0],
            attributes={"currency": "INR", "region": "IN", "password": "never-copy"},
            minute=first_minute,
            content=f"Order created example for {provider}",
        ),
        _add_event(
            repository,
            consent_id=consent_id,
            event_type=EVENT_TYPES[1],
            attributes={"provider": provider, "amount": amount, "api_key": "never-copy"},
            minute=first_minute + 1,
            content=f"Payment authorized with {provider}",
        ),
        _add_event(
            repository,
            consent_id=consent_id,
            event_type=EVENT_TYPES[2],
            attributes={"currency": "INR", "provider": provider},
            minute=first_minute + 2,
            content=f"Order completed with {provider}",
        ),
    )
    replay.record_episode(
        [
            EpisodicEvent(record.id, record.occurred_at or record.created_at)
            for record in records
        ],
        context_id=f"{PREFIX}{session}",
    )
    return records


def test_extracts_maximal_recurrent_schema_and_generalizes_varying_attributes() -> None:
    repository = MemoryRepository(":memory:")
    consent = repository.grant_consent(OWNER)
    replay = EpisodicReplayMemory()
    extractor = SchemaExtractionSystem()
    examples: tuple[
        tuple[MemoryRecord, MemoryRecord, MemoryRecord],
        tuple[MemoryRecord, MemoryRecord, MemoryRecord],
        tuple[MemoryRecord, MemoryRecord, MemoryRecord],
    ] = (
        _record_example(
            repository, replay, consent.id, "session-a",
            provider="stripe", amount=100, first_minute=1,
        ),
        _record_example(
            repository, replay, consent.id, "session-b",
            provider="razorpay", amount=175, first_minute=10,
        ),
        _record_example(
            repository, replay, consent.id, "session-c",
            provider="adyen", amount=250, first_minute=20,
        ),
    )

    proposals = extractor.prepare_from_episodes(
        owner_id=OWNER,
        context_prefix=PREFIX,
        episodic_memory=replay,
        repository=repository,
    )

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.event_types == EVENT_TYPES
    assert proposal.support_count == 3
    assert set(proposal.source_sequences) == {
        tuple(record.id for record in example) for example in examples
    }
    steps = proposal.schema["steps"]
    assert [step["event_type"] for step in steps] == list(EVENT_TYPES)
    assert steps[0]["attributes"]["currency"] == {"kind": "constant", "value": "INR"}
    assert steps[0]["attributes"]["region"] == {"kind": "constant", "value": "IN"}
    assert steps[1]["attributes"]["provider"] == {
        "kind": "slot", "name": "step_1_provider", "value_type": "string"
    }
    assert steps[1]["attributes"]["amount"] == {
        "kind": "slot", "name": "step_1_amount", "value_type": "integer"
    }
    assert "stripe" not in str(proposal.schema)
    assert "razorpay" not in str(proposal.schema)
    assert "adyen" not in str(proposal.schema)
    assert "password" not in str(proposal.schema)
    assert "api_key" not in str(proposal.schema)

    # Repeated preview of the same examples does not create duplicate proposals.
    assert extractor.prepare_from_episodes(
        owner_id=OWNER,
        context_prefix=PREFIX,
        episodic_memory=replay,
        repository=repository,
    ) == ()


def test_single_session_unannotated_and_replayed_identical_examples_do_not_generalize() -> None:
    repository = MemoryRepository(":memory:")
    consent = repository.grant_consent(OWNER)
    replay = EpisodicReplayMemory()
    extractor = SchemaExtractionSystem()
    records = _record_example(
        repository, replay, consent.id, "only-session",
        provider="stripe", amount=100, first_minute=1,
    )
    replay.record_episode(
        [EpisodicEvent(record.id, record.occurred_at or record.created_at) for record in records],
        context_id=f"{PREFIX}second-session",
    )
    _add_event(
        repository,
        consent_id=consent.id,
        event_type="unannotated.event",
        attributes={},
        minute=30,
        content="A free-text event without replay metadata",
    )

    assert extractor.prepare_from_episodes(
        owner_id=OWNER,
        context_prefix=PREFIX,
        episodic_memory=replay,
        repository=repository,
    ) == ()


def test_mixed_consent_and_contradictory_examples_are_excluded() -> None:
    repository = MemoryRepository(":memory:")
    first_consent = repository.grant_consent(OWNER)
    second_consent = repository.grant_consent(OWNER)
    replay = EpisodicReplayMemory()
    extractor = SchemaExtractionSystem(min_distinct_sessions=2)

    example_a = _record_example(
        repository, replay, first_consent.id, "mixed-a",
        provider="stripe", amount=100, first_minute=1,
    )
    example_b = _record_example(
        repository, replay, second_consent.id, "mixed-b",
        provider="razorpay", amount=175, first_minute=10,
    )
    _record_example(
        repository, replay, first_consent.id, "mixed-c",
        provider="adyen", amount=250, first_minute=20,
    )
    repository.link_contradiction(
        OWNER, example_a[0].id, example_a[1].id, note="Synthetic owner-marked conflict"
    )

    proposals = extractor.prepare_from_episodes(
        owner_id=OWNER,
        context_prefix=PREFIX,
        episodic_memory=replay,
        repository=repository,
    )
    # The conflicted first event and the other-consent example must not participate.
    assert all(proposal.event_types != EVENT_TYPES for proposal in proposals)
    mixed_consent_ids = {record.id for record in example_b}
    for proposal in proposals:
        assert not mixed_consent_ids.intersection(proposal.source_memory_ids)
        assert all(
            example_a[0].id not in sequence
            for sequence in proposal.source_sequences
        )


def test_schema_queue_is_owner_scoped_bounded_and_invalidates_changed_sources() -> None:
    repository = MemoryRepository(":memory:")
    consent = repository.grant_consent(OWNER)
    replay = EpisodicReplayMemory()
    extractor = SchemaExtractionSystem(max_proposals=1)
    first = _record_example(
        repository, replay, consent.id, "session-a",
        provider="stripe", amount=100, first_minute=1,
    )
    second = _record_example(
        repository, replay, consent.id, "session-b",
        provider="razorpay", amount=175, first_minute=10,
    )
    proposal = extractor.prepare_from_episodes(
        owner_id=OWNER,
        context_prefix=PREFIX,
        episodic_memory=replay,
        repository=repository,
    )[0]

    assert extractor.list_proposals("other-owner") == ()
    assert extractor.forget_memory(OWNER, first[0].id) == 1
    with pytest.raises(KeyError):
        extractor.proposal_context(OWNER, proposal.id)

    # A new source set can be proposed after the stale candidate is invalidated.
    third = _record_example(
        repository, replay, consent.id, "session-c",
        provider="adyen", amount=250, first_minute=20,
    )
    refreshed = extractor.prepare_from_episodes(
        owner_id=OWNER,
        context_prefix=PREFIX,
        episodic_memory=replay,
        repository=repository,
    )
    assert len(refreshed) == 1
    assert third[0].id in refreshed[0].source_memory_ids
    assert second[0].id in refreshed[0].source_memory_ids


def test_repository_schema_write_revalidates_examples_and_persists_provenance() -> None:
    repository = MemoryRepository(":memory:")
    consent = repository.grant_consent(OWNER)
    replay = EpisodicReplayMemory()
    extractor = SchemaExtractionSystem()
    examples = (
        _record_example(
            repository, replay, consent.id, "session-a",
            provider="stripe", amount=100, first_minute=1,
        ),
        _record_example(
            repository, replay, consent.id, "session-b",
            provider="razorpay", amount=175, first_minute=10,
        ),
    )
    proposal = extractor.prepare_from_episodes(
        owner_id=OWNER,
        context_prefix=PREFIX,
        episodic_memory=replay,
        repository=repository,
    )[0]

    record = repository.add_schema_memory(
        owner_id=OWNER,
        source_sequences=proposal.source_sequences,
        event_types=proposal.event_types,
        schema=proposal.schema,
        content=proposal.summary,
        source=f"owner-approved schema extraction:{proposal.id}",
        consent_id=consent.id,
        metadata={"schema_proposal_id": proposal.id},
    )
    assert record.kind is MemoryKind.PROCEDURAL
    assert record.epistemic_status == "inferred"
    assert record.metadata["knowledge_type"] == "ordered_event_schema"
    assert record.metadata["schema"] == proposal.schema
    assert record.metadata["schema_source_sequences"] == [
        list(sequence) for sequence in proposal.source_sequences
    ]
    assert record.metadata["schema_support_count"] == 2

    # A source event-type change invalidates the proposed generalisation.
    repository.correct(
        OWNER, examples[0][0].id, metadata={"event_type": "order.cancelled", "attributes": {}}
    )
    with pytest.raises(ValueError, match="event type changed"):
        repository.add_schema_memory(
            owner_id=OWNER,
            source_sequences=proposal.source_sequences,
            event_types=proposal.event_types,
            schema=proposal.schema,
            content=proposal.summary,
            source="stale schema proposal",
            consent_id=consent.id,
        )


def test_repository_rejects_contradictions_and_revoked_schema_sources() -> None:
    repository = MemoryRepository(":memory:")
    consent = repository.grant_consent(OWNER)
    first = _add_event(
        repository, consent_id=consent.id, event_type="task.started",
        attributes={}, minute=1, content="Task started",
    )
    second = _add_event(
        repository, consent_id=consent.id, event_type="task.completed",
        attributes={}, minute=2, content="Task completed",
    )
    third = _add_event(
        repository, consent_id=consent.id, event_type="task.started",
        attributes={}, minute=3, content="Another task started",
    )
    fourth = _add_event(
        repository, consent_id=consent.id, event_type="task.completed",
        attributes={}, minute=4, content="Another task completed",
    )
    sequences = ((first.id, second.id), (third.id, fourth.id))
    schema = {"version": 1, "steps": [{"event_type": "task.started"}, {"event_type": "task.completed"}]}
    repository.link_contradiction(OWNER, first.id, second.id, note="Conflict")

    with pytest.raises(ValueError, match="Contradictory"):
        repository.add_schema_memory(
            owner_id=OWNER,
            source_sequences=sequences,
            event_types=("task.started", "task.completed"),
            schema=schema,
            content="Task workflow schema",
            source="test",
            consent_id=consent.id,
        )

    assert repository.revoke_consent(OWNER, consent.id)
    with pytest.raises(MemoryNotFoundError):
        repository.add_schema_memory(
            owner_id=OWNER,
            source_sequences=sequences,
            event_types=("task.started", "task.completed"),
            schema=schema,
            content="Task workflow schema",
            source="test",
            consent_id=consent.id,
        )


def test_schema_approval_is_serialized_against_duplicate_requests() -> None:
    repository = MemoryRepository(":memory:")
    consent = repository.grant_consent(OWNER)
    replay = EpisodicReplayMemory()
    extractor = SchemaExtractionSystem()
    _record_example(
        repository, replay, consent.id, "session-a",
        provider="stripe", amount=100, first_minute=1,
    )
    _record_example(
        repository, replay, consent.id, "session-b",
        provider="razorpay", amount=175, first_minute=10,
    )
    proposal = extractor.prepare_from_episodes(
        owner_id=OWNER,
        context_prefix=PREFIX,
        episodic_memory=replay,
        repository=repository,
    )[0]
    calls = 0
    lock = Lock()

    def persist(_proposal, _consent_id: str) -> str:
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.02)
        return "schema-memory-id"

    def approve() -> str:
        _reviewed, memory_id = extractor.approve_with(
            OWNER, proposal.id, reviewer=OWNER, persist=persist
        )
        return memory_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(approve) for _ in range(2)]
        results: list[str] = []
        failures: list[str] = []
        for future in futures:
            try:
                results.append(future.result())
            except ValueError as error:
                failures.append(str(error))

    assert results == ["schema-memory-id"]
    assert len(failures) == 1
    assert calls == 1



def test_schema_rejection_does_not_write_a_procedural_memory() -> None:
    repository = MemoryRepository(":memory:")
    consent = repository.grant_consent(OWNER)
    replay = EpisodicReplayMemory()
    extractor = SchemaExtractionSystem()
    _record_example(
        repository, replay, consent.id, "reject-a",
        provider="stripe", amount=100, first_minute=1,
    )
    _record_example(
        repository, replay, consent.id, "reject-b",
        provider="razorpay", amount=175, first_minute=10,
    )
    proposal = extractor.prepare_from_episodes(
        owner_id=OWNER,
        context_prefix=PREFIX,
        episodic_memory=replay,
        repository=repository,
    )[0]
    before = len(repository.list(OWNER))
    rejected = extractor.review(
        OWNER, proposal.id, decision="rejected",
        reviewer=OWNER, note="This workflow is too narrow to generalize.",
    )
    assert rejected.status == "rejected"
    assert len(repository.list(OWNER)) == before


def test_new_contradiction_invalidates_an_existing_schema_proposal() -> None:
    repository = MemoryRepository(":memory:")
    consent = repository.grant_consent(OWNER)
    replay = EpisodicReplayMemory()
    extractor = SchemaExtractionSystem()
    first = _record_example(
        repository, replay, consent.id, "conflict-a",
        provider="stripe", amount=100, first_minute=1,
    )
    _record_example(
        repository, replay, consent.id, "conflict-b",
        provider="razorpay", amount=175, first_minute=10,
    )
    proposal = extractor.prepare_from_episodes(
        owner_id=OWNER,
        context_prefix=PREFIX,
        episodic_memory=replay,
        repository=repository,
    )[0]

    repository.link_contradiction(
        OWNER, first[0].id, first[1].id, note="Newly identified conflict"
    )
    assert extractor.forget_memory(OWNER, first[0].id) == 1
    with pytest.raises(KeyError):
        extractor.proposal_context(OWNER, proposal.id)


def test_schema_source_sequences_must_be_distinct_and_match_event_types() -> None:
    repository = MemoryRepository(":memory:")
    consent = repository.grant_consent(OWNER)
    first = _add_event(
        repository, consent_id=consent.id, event_type="task.started",
        attributes={}, minute=1, content="Task started",
    )
    second = _add_event(
        repository, consent_id=consent.id, event_type="task.completed",
        attributes={}, minute=2, content="Task completed",
    )
    schema = {
        "version": 1,
        "steps": [
            {"event_type": "task.started"},
            {"event_type": "task.completed"},
        ],
    }
    with pytest.raises(ValueError, match="distinct source sequences"):
        repository.add_schema_memory(
            owner_id=OWNER,
            source_sequences=((first.id, second.id), (first.id, second.id)),
            event_types=("task.started", "task.completed"),
            schema=schema,
            content="A repeated task schema",
            source="owner approval",
            consent_id=consent.id,
        )
