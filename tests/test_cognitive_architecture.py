from __future__ import annotations

import pytest

from artificial_human.cognition.attention import AttentionCandidate, AttentionSystem
from artificial_human.cognition.perception import PerceptionSystem
from artificial_human.cognition.prediction import PredictionSystem
from artificial_human.cognition.working_memory import (
    WorkingMemory,
    WorkingMemorySessions,
    new_working_memory_entry,
)


def _entry(item_id: str, content: str, salience: float = 0.5):
    return new_working_memory_entry(
        id=item_id,
        content=content,
        kind="test",
        source="unit-test",
        salience=salience,
    )


def test_perception_normalizes_explicit_language_input() -> None:
    percept = PerceptionSystem().perceive("  Remember this concept.  ")
    assert percept.content == "Remember this concept."
    assert percept.modality == "language"
    assert "remember" in percept.tokens
    assert percept.source == "user"


def test_perception_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        PerceptionSystem().perceive("   ")


def test_attention_selects_highest_weight_candidates_within_capacity() -> None:
    selected = AttentionSystem().select(
        [
            AttentionCandidate("low", "memory", relevance=0.1),
            AttentionCandidate("high", "memory", relevance=0.9),
            AttentionCandidate("mid", "example", relevance=0.5),
        ],
        limit=2,
    )
    assert [item.reference_id for item in selected] == ["high", "mid"]


def test_attention_rejects_out_of_range_signals() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        AttentionSystem().select(
            [AttentionCandidate("bad", "memory", relevance=1.5)]
        )


def test_working_memory_is_bounded_and_tracks_reactivation() -> None:
    memory = WorkingMemory(capacity=2)
    memory.remember(_entry("a", "first", 0.3))
    memory.remember(_entry("b", "second", 0.4))
    memory.remember(_entry("a", "first revisited", 0.8))
    assert len(memory.items()) == 2
    assert next(item for item in memory.items() if item.id == "a").access_count == 1
    memory.remember(_entry("c", "third", 0.9))
    assert len(memory.items()) == 2
    assert "c" in {item.id for item in memory.items()}


def test_working_memory_rejects_invalid_salience() -> None:
    memory = WorkingMemory()
    with pytest.raises(ValueError, match="between 0 and 1"):
        memory.remember(_entry("bad", "invalid", 1.2))


def test_working_memory_sessions_are_owner_scoped_and_clearable() -> None:
    sessions = WorkingMemorySessions(capacity=3, max_sessions=2)
    owner_workspace = sessions.get("owner-a", "session")
    owner_workspace.remember(_entry("private", "owner a context"))
    other_workspace = sessions.get("owner-b", "session")
    assert other_workspace.items() == ()
    assert sessions.clear("owner-a", "session") is True
    assert sessions.get("owner-a", "session").items() == ()
    assert sessions.clear("owner-a", "session") is True


def test_working_memory_sessions_validate_ids_and_bound_session_count() -> None:
    sessions = WorkingMemorySessions(max_sessions=1)
    sessions.get("owner", "first")
    sessions.get("owner", "second")
    with pytest.raises(ValueError, match="128 characters"):
        sessions.get("owner", "x" * 129)


def test_prediction_assesses_novelty_against_existing_context() -> None:
    perception = PerceptionSystem()
    predictor = PredictionSystem()
    expected = perception.perceive("The hippocampus supports episodic memory.")
    matching = predictor.assess(
        expected,
        ["The hippocampus supports episodic memory."],
    )
    novel = predictor.assess(expected, ["The heart circulates blood."])
    empty = predictor.assess(expected, [])
    assert matching.comparable is True
    assert matching.novelty == 0
    assert novel.novelty == 1
    assert empty.comparable is False
