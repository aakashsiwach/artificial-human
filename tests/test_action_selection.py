from __future__ import annotations

import pytest

from artificial_human.neural.action_selection import (
    ActionSelectionCircuit,
    ActionSpec,
    SafeActionSpace,
    run_action_selection_comparison,
)


@pytest.fixture
def action_space() -> SafeActionSpace:
    return SafeActionSpace(
        (
            ActionSpec("go", "Move in the simulated task"),
            ActionSpec("wait", "Wait in the simulated task"),
            ActionSpec("inspect", "Inspect the simulated task"),
        )
    )


def test_selection_is_deterministic_and_returns_inspectable_pathways(
    action_space: SafeActionSpace,
) -> None:
    first = ActionSelectionCircuit(action_space, seed=7)
    second = ActionSelectionCircuit(action_space, seed=7)

    assert first.select("state-a") == second.select("state-a")
    choice = first.select("state-a", explore=False)
    assert choice.action_id == "go"
    assert choice.uncertainty == 1.0
    assert choice.net_value == choice.direct_activation - choice.indirect_suppression


def test_positive_prediction_error_strengthens_direct_channel(
    action_space: SafeActionSpace,
) -> None:
    circuit = ActionSelectionCircuit(action_space, learning_rate=0.2)
    update = circuit.observe("state-a", "go", 1.0, terminal=True)

    assert update.prediction_error == 1.0
    assert update.direct_change == pytest.approx(0.2)
    assert update.indirect_change == 0.0
    assert update.value_after > update.value_before


def test_negative_prediction_error_strengthens_indirect_channel(
    action_space: SafeActionSpace,
) -> None:
    circuit = ActionSelectionCircuit(action_space, learning_rate=0.2)
    update = circuit.observe("state-a", "go", -1.0, terminal=True)

    assert update.prediction_error == -1.0
    assert update.direct_change == 0.0
    assert update.indirect_change == pytest.approx(0.2)
    assert update.value_after < update.value_before


def test_reward_and_pathway_weights_are_bounded(
    action_space: SafeActionSpace,
) -> None:
    circuit = ActionSelectionCircuit(action_space, learning_rate=1.0)
    for _ in range(100):
        circuit.observe("state-a", "go", 1.0, terminal=True)
        circuit.observe("state-a", "wait", -1.0, terminal=True)

    channels = circuit.snapshot()["channels"]
    assert isinstance(channels, list)
    assert all(
        0.0 <= item["direct_activation"] <= circuit.MAX_CHANNEL_WEIGHT
        and 0.0 <= item["indirect_suppression"] <= circuit.MAX_CHANNEL_WEIGHT
        for item in channels
    )


@pytest.mark.parametrize("reward", [1.01, -1.01, float("nan"), float("inf"), True])
def test_reward_hacking_inputs_are_rejected(
    action_space: SafeActionSpace,
    reward: float,
) -> None:
    circuit = ActionSelectionCircuit(action_space)
    with pytest.raises(ValueError, match="reward"):
        circuit.observe("state-a", "go", reward, terminal=True)


def test_learned_reward_cannot_expand_the_normative_action_allowlist(
    action_space: SafeActionSpace,
) -> None:
    circuit = ActionSelectionCircuit(action_space)
    with pytest.raises(ValueError, match="not in the approved simulation allowlist"):
        circuit.observe("state-a", "launch-device", 1.0, terminal=True)
    with pytest.raises(ValueError, match="not in the approved simulation allowlist"):
        circuit.select("state-a", ("go", "launch-device"))


def test_available_action_subset_is_respected(
    action_space: SafeActionSpace,
) -> None:
    circuit = ActionSelectionCircuit(action_space)
    circuit.observe("state-a", "wait", 1.0, terminal=True)
    choice = circuit.select("state-a", ("go", "inspect"), explore=False)
    assert choice.action_id in {"go", "inspect"}


def test_nonterminal_updates_require_a_next_state(
    action_space: SafeActionSpace,
) -> None:
    circuit = ActionSelectionCircuit(action_space)
    with pytest.raises(ValueError, match="next_state_id"):
        circuit.observe("state-a", "go", 0.2)


def test_state_action_capacity_is_enforced(
    action_space: SafeActionSpace,
) -> None:
    circuit = ActionSelectionCircuit(action_space, max_state_action_pairs=1)
    circuit.observe("state-a", "go", 0.2, terminal=True)
    with pytest.raises(RuntimeError, match="capacity"):
        circuit.observe("state-a", "wait", 0.2, terminal=True)


def test_reset_clears_all_learned_runtime_state(
    action_space: SafeActionSpace,
) -> None:
    circuit = ActionSelectionCircuit(action_space)
    circuit.observe("state-a", "go", 1.0, terminal=True)
    circuit.reset()

    assert circuit.snapshot()["state_action_pairs"] == 0
    assert circuit.snapshot()["updates"] == 0


def test_contextual_learning_beats_random_and_fixed_policy_on_heldout_episodes() -> None:
    first = run_action_selection_comparison()
    second = run_action_selection_comparison()

    assert first == second
    assert first.learned_accuracy >= 0.9
    assert first.learned_accuracy > first.random_accuracy
    assert first.learned_accuracy > first.fixed_policy_accuracy
    assert first.learned_mean_reward > 0.0


def test_reward_context_shift_can_retrain_the_opponent_channels(
    action_space: SafeActionSpace,
) -> None:
    circuit = ActionSelectionCircuit(
        action_space,
        learning_rate=0.25,
        exploration_rate=0.0,
        exploration_bonus=0.0,
    )
    for _ in range(30):
        circuit.observe("context", "go", 1.0, terminal=True)
        circuit.observe("context", "wait", -1.0, terminal=True)
    assert circuit.select("context", explore=False).action_id == "go"

    for _ in range(50):
        circuit.observe("context", "go", -1.0, terminal=True)
        circuit.observe("context", "wait", 1.0, terminal=True)
    assert circuit.select("context", explore=False).action_id == "wait"
