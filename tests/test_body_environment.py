from __future__ import annotations

import pytest

from artificial_human.neural.body_environment import (
    SimulatedBodyEnvironment,
    run_closed_loop_task,
)


def test_simulated_body_rejects_out_of_bounds_commands() -> None:
    environment = SimulatedBodyEnvironment()
    with pytest.raises(ValueError, match="motor command"):
        environment.step(1.01)


def test_simulated_body_is_deterministic_and_bounded() -> None:
    first = SimulatedBodyEnvironment()
    second = SimulatedBodyEnvironment()
    for command in (1.0, 0.5, -0.3, 0.0):
        assert first.step(command) == second.step(command)
    assert -1.0 <= first.state.position <= 1.0
    assert -1.0 <= first.state.velocity <= 1.0


def test_closed_loop_task_reaches_goal_through_sensing_and_action() -> None:
    result = run_closed_loop_task()

    assert result.success
    assert result.steps == 240
    assert result.final_distance <= 0.12
    assert abs(result.final_velocity) <= 0.12
    counts = dict(result.action_counts)
    assert counts["approach"] > 0
    assert counts["hold"] > 0
    assert result.neural_spike_count > 0
    first_trace = result.trace[0]
    sensor_midpoint = (
        first_trace.visual_position + first_trace.proprioceptive_position
    ) / 2.0
    assert abs(first_trace.fused_position - sensor_midpoint) < 0.06


def test_closed_loop_trace_is_reproducible_for_fixed_seed() -> None:
    first = run_closed_loop_task(seed=41)
    second = run_closed_loop_task(seed=41)

    assert first == second
    assert first.trace == second.trace


def test_disturbance_shift_causes_prediction_error_and_adaptation() -> None:
    result = run_closed_loop_task()

    assert result.mean_abs_prediction_error > 0.0
    assert result.mean_abs_prediction_error_after_shift > 0.0
    assert result.max_abs_prediction_error > 0.0
    assert result.learned_disturbance_bias > 0.01


def test_synthetic_sensory_data_is_ephemeral() -> None:
    result = run_closed_loop_task()

    assert result.sensory_retained_count == 0


def test_delayed_noisy_feedback_still_completes_the_simulated_task() -> None:
    result = run_closed_loop_task(
        feedback_delay_steps=5,
        sensor_noise_std=0.03,
        seed=17,
    )

    assert result.success
    assert result.mean_abs_prediction_error > 0.0


def test_closed_loop_task_rejects_unbounded_configuration() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        run_closed_loop_task(max_steps=20)
    with pytest.raises(ValueError, match="sensor_noise_std"):
        run_closed_loop_task(sensor_noise_std=0.1)
