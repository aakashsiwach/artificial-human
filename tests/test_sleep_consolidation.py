from __future__ import annotations

from artificial_human.cognition.episodic_replay import EpisodicEvent, EpisodicReplayMemory
from artificial_human.core.models import MemoryKind
from artificial_human.memory.repository import MemoryRepository
from artificial_human.sleep.consolidation import SleepSystem


OWNER = "owner"
PREFIX = "owner-0123456789abcdef.session-"


def _memory(repository: MemoryRepository, consent_id: str, name: str, minute: int):
    return repository.add(
        owner_id=OWNER,
        kind=MemoryKind.EPISODIC,
        content=f"{name} happened during the project workflow",
        source="owner event record",
        consent_id=consent_id,
        occurred_at=f"2026-11-25T10:{minute:02d}:00+00:00",
    )


def _record_pair(
    replay: EpisodicReplayMemory,
    left,
    right,
    *session_suffixes: str,
) -> None:
    for suffix in session_suffixes:
        replay.record_episode(
            [
                EpisodicEvent(left.id, left.occurred_at or left.created_at),
                EpisodicEvent(right.id, right.occurred_at or right.created_at),
            ],
            context_id=f"{PREFIX}{suffix}",
        )


def test_consolidation_detects_only_patterns_repeated_across_distinct_sessions() -> None:
    repository = MemoryRepository(":memory:")
    consent = repository.grant_consent(OWNER)
    replay = EpisodicReplayMemory()
    sleep = SleepSystem()
    first = _memory(repository, consent.id, "Project kickoff", 1)
    second = _memory(repository, consent.id, "Architecture review", 2)
    singleton_first = _memory(repository, consent.id, "One-off task", 3)
    singleton_second = _memory(repository, consent.id, "One-off result", 4)

    _record_pair(replay, first, second, "session-a", "session-b")
    _record_pair(replay, singleton_first, singleton_second, "session-a")

    proposals = sleep.prepare_from_episodes(
        owner_id=OWNER,
        context_prefix=PREFIX,
        episodic_memory=replay,
        repository=repository,
    )

    assert len(proposals) == 1
    assert proposals[0].source_memory_ids == (first.id, second.id)
    assert proposals[0].recurrence_count == 2
    assert proposals[0].status == "pending_review"
    assert "not an established fact" in proposals[0].summary
    assert sleep.list_proposals(OWNER) == proposals

    # Preview is idempotent; an already queued candidate is not duplicated.
    assert sleep.prepare_from_episodes(
        owner_id=OWNER,
        context_prefix=PREFIX,
        episodic_memory=replay,
        repository=repository,
    ) == ()


def test_consolidation_ignores_contradictions_mixed_consent_and_wrong_owner() -> None:
    repository = MemoryRepository(":memory:")
    consent = repository.grant_consent(OWNER)
    second_consent = repository.grant_consent(OWNER)
    replay = EpisodicReplayMemory()
    sleep = SleepSystem()
    conflicting_a = _memory(repository, consent.id, "Old deployment plan", 1)
    conflicting_b = _memory(repository, consent.id, "Different deployment plan", 2)
    mixed_a = _memory(repository, consent.id, "Mixed consent event A", 3)
    mixed_b = _memory(repository, second_consent.id, "Mixed consent event B", 4)

    repository.link_contradiction(
        OWNER, conflicting_a.id, conflicting_b.id, note="Owner marked as conflicting"
    )
    _record_pair(replay, conflicting_a, conflicting_b, "session-a", "session-b")
    _record_pair(replay, mixed_a, mixed_b, "session-a", "session-b")

    other_prefix = "owner-fedcba9876543210.session-"
    replay.record_episode(
        [
            EpisodicEvent(conflicting_a.id, conflicting_a.occurred_at or conflicting_a.created_at),
            EpisodicEvent(conflicting_b.id, conflicting_b.occurred_at or conflicting_b.created_at),
        ],
        context_id=f"{other_prefix}session-a",
    )

    proposals = sleep.prepare_from_episodes(
        owner_id=OWNER,
        context_prefix=PREFIX,
        episodic_memory=replay,
        repository=repository,
    )
    assert proposals == ()


def test_review_is_owner_scoped_and_only_pending_proposals_can_change() -> None:
    repository = MemoryRepository(":memory:")
    consent = repository.grant_consent(OWNER)
    replay = EpisodicReplayMemory()
    sleep = SleepSystem()
    first = _memory(repository, consent.id, "First event", 1)
    second = _memory(repository, consent.id, "Second event", 2)
    _record_pair(replay, first, second, "session-a", "session-b")
    proposal = sleep.prepare_from_episodes(
        owner_id=OWNER,
        context_prefix=PREFIX,
        episodic_memory=replay,
        repository=repository,
    )[0]

    try:
        sleep.proposal_context("another-owner", proposal.id)
    except KeyError:
        pass
    else:
        raise AssertionError("A proposal must not be visible to another owner")

    reviewed = sleep.review(
        OWNER,
        proposal.id,
        decision="rejected",
        reviewer=OWNER,
        note="Pattern needs more evidence",
    )
    assert reviewed.status == "rejected"
    assert reviewed.reviewed_at is not None
    assert sleep.snapshot(OWNER)["rejected"] == 1

    try:
        sleep.review(
            OWNER, proposal.id, decision="approved", reviewer=OWNER,
            consolidated_memory_id="derived-memory",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("A reviewed proposal cannot be reviewed a second time")


def test_proposals_are_cleared_when_source_memory_or_consent_is_removed() -> None:
    repository = MemoryRepository(":memory:")
    consent = repository.grant_consent(OWNER)
    replay = EpisodicReplayMemory()
    sleep = SleepSystem()
    first = _memory(repository, consent.id, "First event", 1)
    second = _memory(repository, consent.id, "Second event", 2)
    _record_pair(replay, first, second, "session-a", "session-b")
    proposal = sleep.prepare_from_episodes(
        owner_id=OWNER,
        context_prefix=PREFIX,
        episodic_memory=replay,
        repository=repository,
    )[0]

    assert sleep.forget_memory(OWNER, first.id) == 1
    try:
        sleep.get_proposal(OWNER, proposal.id)
    except KeyError:
        pass
    else:
        raise AssertionError("A proposal using a changed source must be invalidated")
    _record_pair(replay, first, second, "session-c", "session-d")
    refreshed = sleep.prepare_from_episodes(
        owner_id=OWNER,
        context_prefix=PREFIX,
        episodic_memory=replay,
        repository=repository,
    )
    assert len(refreshed) == 1

    assert sleep.clear_consent(OWNER, consent.id) == 1
    assert sleep.snapshot(OWNER)["pending_review"] == 0


def test_consolidation_queue_is_bounded_without_discarding_pending_reviews() -> None:
    repository = MemoryRepository(":memory:")
    consent = repository.grant_consent(OWNER)
    replay = EpisodicReplayMemory()
    sleep = SleepSystem(max_proposals=1)
    records = [
        _memory(repository, consent.id, f"Event {index}", index + 1)
        for index in range(4)
    ]
    _record_pair(replay, records[0], records[1], "session-a", "session-b")
    _record_pair(replay, records[2], records[3], "session-c", "session-d")

    proposals = sleep.prepare_from_episodes(
        owner_id=OWNER,
        context_prefix=PREFIX,
        episodic_memory=replay,
        repository=repository,
    )
    assert len(proposals) == 1
    assert sleep.snapshot(OWNER)["pending_review"] == 1
    assert len(sleep.list_proposals(OWNER, limit=100)) == 1
