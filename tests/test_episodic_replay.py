from __future__ import annotations

from artificial_human.cognition.episodic_replay import (
    EpisodicEvent,
    EpisodicReplayMemory,
)


def _event(memory_id: str, minute: int) -> EpisodicEvent:
    return EpisodicEvent(
        memory_id=memory_id,
        occurred_at=f"2026-11-25T10:{minute:02d}:00+00:00",
    )


def test_episode_orders_events_and_replays_adjacent_neighbours() -> None:
    replay = EpisodicReplayMemory()
    # Input order deliberately differs from event order.
    assert replay.record_episode(
        [_event("event-c", 3), _event("event-a", 1), _event("event-b", 2)],
        context_id="owner-a.session-a",
    )

    after_a = replay.replay(["event-a"], context_id="owner-a.session-a")
    assert [(item.memory_id, item.relation) for item in after_a] == [("event-b", "after")]

    around_b = replay.replay(["event-b"], context_id="owner-a.session-a")
    assert {(item.memory_id, item.relation) for item in around_b} == {
        ("event-a", "before"),
        ("event-c", "after"),
    }


def test_replay_isolated_by_owner_and_session_context() -> None:
    replay = EpisodicReplayMemory()
    replay.record_episode(
        [_event("event-a", 1), _event("event-b", 2)],
        context_id="owner-a.session-a",
    )

    assert replay.replay(["event-a"], context_id="owner-b.session-a") == ()
    assert replay.replay(["event-a"], context_id="owner-a.session-b") == ()
    assert replay.replay(["event-a"], context_id="owner-a.session-a")


def test_explicit_conflicts_block_temporal_replay() -> None:
    replay = EpisodicReplayMemory()
    replay.record_episode(
        [_event("event-a", 1), _event("event-b", 2)],
        context_id="owner-a.session-a",
    )

    assert replay.replay(
        ["event-a"],
        context_id="owner-a.session-a",
        blocked_pairs=[("event-a", "event-b")],
    ) == ()


def test_successful_replay_strengthens_only_selected_transitions() -> None:
    replay = EpisodicReplayMemory(retrieval_rate=0.2)
    replay.record_episode(
        [_event("event-a", 1), _event("event-b", 2), _event("event-c", 3)],
        context_id="owner-a.session-a",
    )

    result = replay.replay(["event-a"], context_id="owner-a.session-a")
    assert len(result) == 1
    assert replay.reinforce_in_context(
        result, context_id="owner-a.session-a"
    ) == 1
    assert replay.snapshot()["retrieval_strengthenings"] == 1


def test_episode_and_transition_budgets_are_enforced() -> None:
    replay = EpisodicReplayMemory(
        max_contexts=1,
        max_episodes_per_context=1,
        max_events_per_episode=3,
        max_transitions=2,
    )
    replay.record_episode(
        [_event("event-a", 1), _event("event-b", 2), _event("event-c", 3)],
        context_id="owner-a.session-a",
    )
    replay.record_episode(
        [_event("event-d", 4), _event("event-e", 5)],
        context_id="owner-a.session-a",
    )
    replay.record_episode(
        [_event("event-f", 6), _event("event-g", 7)],
        context_id="owner-b.session-b",
    )

    state = replay.snapshot()
    assert state["context_count"] == 1
    assert state["episode_count"] == 1
    assert state["transition_count"] <= 2
    assert state["evicted_transitions"] >= 2
    assert replay.replay(["event-a"], context_id="owner-a.session-a") == ()
    assert replay.replay(["event-f"], context_id="owner-b.session-b")


def test_context_cleanup_discards_all_episode_traces() -> None:
    replay = EpisodicReplayMemory()
    replay.record_episode(
        [_event("event-a", 1), _event("event-b", 2)],
        context_id="owner-a.session-a",
    )
    replay.record_episode(
        [_event("event-c", 3), _event("event-d", 4)],
        context_id="owner-b.session-b",
    )

    replay.forget_context("owner-a.session-a")
    assert replay.replay(["event-a"], context_id="owner-a.session-a") == ()
    assert replay.replay(["event-c"], context_id="owner-b.session-b")

    replay.forget_contexts("owner-b.")
    assert replay.snapshot()["context_count"] == 0
    assert replay.snapshot()["transition_count"] == 0


def test_replay_rejects_invalid_contexts_ids_and_timestamps() -> None:
    replay = EpisodicReplayMemory()
    for context in ("Raw Owner ID", "../bad", ""):
        try:
            replay.record_episode(
                [_event("event-a", 1), _event("event-b", 2)],
                context_id=context,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid context must be rejected")

    for bad_id in ("contains spaces", "../unsafe"):
        try:
            replay.record_episode(
                [_event(bad_id, 1), _event("event-b", 2)],
                context_id="owner-a.session-a",
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid memory ID must be rejected")

    try:
        replay.record_episode(
            [
                EpisodicEvent(memory_id="event-a", occurred_at="not-a-time"),
                _event("event-b", 2),
            ],
            context_id="owner-a.session-a",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid timestamps must be rejected")

