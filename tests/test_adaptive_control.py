from __future__ import annotations

import pytest

from artificial_human.neural.adaptive_control import (
    AdaptiveForwardController,
    run_adaptive_control_comparison,
)


def test_forward_model_learns_from_sensory_prediction_error() -> None:
    controller = AdaptiveForwardController()
    velocity = 0.2
    command = 0.3
    actual_next_velocity = 0.82 * velocity + 0.28 * command + 0.12

    before = controller.model.predict(
        __import__("artificial_human.neural.world_model", fromlist=["WorldState"]).WorldState((velocity,), 0),
        (command,),
    ).state.values[0]
    initial_error = abs(actual_next_velocity - before)

    for tick in range(30):
        controller.observe(
            velocity,
            command,
            actual_next_velocity,
            tick=tick,
        )

    after = controller.model.predict(
        __import__("artificial_human.neural.world_model", fromlist=["WorldState"]).WorldState((velocity,), 31),
        (command,),
    ).state.values[0]
    assert abs(actual_next_velocity - after) < initial_error
    assert controller.model.observations == 30


def test_feedback_delay_holds_and_then_releases_updates() -> None:
    controller = AdaptiveForwardController(feedback_delay_steps=2)

    assert controller.observe(0.1, 0.2, 0.3, tick=0) is None
    assert controller.observe(0.1, 0.2, 0.3, tick=1) is None
    assert controller.pending_feedback == 2

    update = controller.observe(0.1, 0.2, 0.3, tick=2)
    assert update is not None
    assert controller.pending_feedback == 2
    assert controller.model.observations == 1


def test_motor_command_is_bounded_and_prediction_is_inspectable() -> None:
    controller = AdaptiveForwardController()
    command = controller.command(-1.0, -1.0, 1.0, tick=0)

    assert -1.0 <= command.command <= 1.0
    assert math_is_finite(command.predicted_next_velocity)
    assert math_is_finite(command.predicted_next_position)
    assert command.uncertainty > 0.0


def test_reset_discards_delayed_feedback_and_restores_model_prior() -> None:
    controller = AdaptiveForwardController(feedback_delay_steps=3)
    initial = controller.snapshot()
    controller.observe(0.2, 0.4, 0.7, tick=0)
    controller.observe(0.2, 0.4, 0.7, tick=1)
    controller.observe(0.2, 0.4, 0.7, tick=2)
    controller.observe(0.2, 0.4, 0.7, tick=3)

    assert controller.model.observations == 1
    controller.reset()

    assert controller.snapshot()["pending_feedback"] == 0
    assert controller.snapshot()["observations"] == 0
    assert controller.snapshot()["bias"] == initial["bias"]
    assert controller.snapshot()["damping"] == initial["damping"]
    assert controller.snapshot()["control_gain"] == initial["control_gain"]


def test_adaptive_forward_control_beats_frozen_model_and_recovers_after_shift() -> None:
    first = run_adaptive_control_comparison()
    second = run_adaptive_control_comparison()

    assert first == second
    assert first.adaptive_rmse < first.baseline_rmse * 0.5
    assert first.relative_improvement > 0.5
    assert first.adaptive_post_shift_rmse < first.baseline_post_shift_rmse
    assert first.adaptive_late_recovery_rmse < first.adaptive_early_recovery_rmse


def test_adaptive_control_handles_delayed_noisy_feedback() -> None:
    result = run_adaptive_control_comparison(
        feedback_delay_steps=5,
        sensor_noise_std=0.06,
    )

    assert result.adaptive_rmse < result.baseline_rmse
    assert result.relative_improvement > 0.3
    assert result.adaptive_post_shift_rmse < result.baseline_post_shift_rmse


def test_controller_rejects_invalid_delay_and_sensor_values() -> None:
    with pytest.raises(ValueError, match="feedback_delay_steps"):
        AdaptiveForwardController(feedback_delay_steps=33)
    with pytest.raises(ValueError, match="sensor_noise_std"):
        run_adaptive_control_comparison(sensor_noise_std=0.2)


def math_is_finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")
