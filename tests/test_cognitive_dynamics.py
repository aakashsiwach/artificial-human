import pytest

from artificial_human.cognition.dynamics import RecurrentDynamicsEngine


def test_recurrent_dynamics_propagates_bounded_activity_without_retaining_text() -> None:
    engine = RecurrentDynamicsEngine(cycles_per_input=8)

    first = engine.process("Explain how memory and sequence context connect.")
    assert first["version"] == "recurrent-population-dynamics-v1"
    assert first["inputs_processed"] == 1
    assert first["integration_steps"] == 8
    assert first["population_count"] == 7
    assert first["projection_count"] == 10
    assert first["active_population_count"] > 0
    assert first["plasticity_enabled"] is False
    assert first["state_storage"] == "volatile_process_memory"
    assert all(
        0.0 <= population["activity"] <= 1.0
        for population in first["populations"]
    )
    assert all(
        0.0 <= population["external_drive"] <= 1.0
        for population in first["populations"]
    )
    assert "Explain how memory" not in str(first)


def test_recurrent_dynamics_retains_only_activity_between_inputs() -> None:
    engine = RecurrentDynamicsEngine()
    engine.process("Explain episodic memory.")
    first = engine.snapshot().as_dict()
    engine.process("Plan safe implementation steps.")
    second = engine.snapshot().as_dict()

    assert second["inputs_processed"] == 2
    assert second["integration_steps"] == 16
    assert second["mean_activity"] >= 0.0
    assert second["populations"] != first["populations"]


def test_recurrent_dynamics_reset_clears_volatile_state() -> None:
    engine = RecurrentDynamicsEngine()
    engine.process("Explain associative memory.")
    engine.reset()
    state = engine.snapshot().as_dict()

    assert state["inputs_processed"] == 0
    assert state["integration_steps"] == 0
    assert state["mean_activity"] == 0.0
    assert state["active_population_count"] == 0
    assert all(item["activity"] == 0.0 for item in state["populations"])


@pytest.mark.parametrize("cycles", [0, -1, 33, True])
def test_recurrent_dynamics_rejects_unbounded_cycle_counts(cycles) -> None:
    with pytest.raises(ValueError, match="cycles_per_input"):
        RecurrentDynamicsEngine(cycles_per_input=cycles)


def test_recurrent_dynamics_rejects_empty_inputs() -> None:
    engine = RecurrentDynamicsEngine()
    with pytest.raises(ValueError, match="non-empty text"):
        engine.process("   ")


def test_idle_tick_advances_resting_activity_without_counting_user_input() -> None:
    engine = RecurrentDynamicsEngine()
    before = engine.snapshot().as_dict()

    after = engine.tick_idle()

    assert after["idle_ticks"] == 1
    assert after["inputs_processed"] == 0
    assert after["integration_steps"] == 1
    assert after["intrinsic_activity_model"] == "deterministic_low_amplitude_approximation"
    assert after["plasticity_enabled"] is False
    assert after["populations"] != before["populations"]


def test_repeated_idle_ticks_remain_bounded_and_reset_cleanly() -> None:
    engine = RecurrentDynamicsEngine()
    for _ in range(100):
        state = engine.tick_idle()

    assert state["idle_ticks"] == 100
    assert state["inputs_processed"] == 0
    assert all(0.0 <= item["activity"] <= 1.0 for item in state["populations"])

    engine.reset()
    reset_state = engine.snapshot().as_dict()
    assert reset_state["idle_ticks"] == 0
    assert reset_state["integration_steps"] == 0
    assert reset_state["mean_activity"] == 0.0
