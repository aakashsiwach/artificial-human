"""Bounded forward-model learning and simulation-only motor adaptation.

This module is a computational hypothesis inspired by cerebellar forward models
and sensory prediction-error-driven adaptation. It is not an anatomical model
and contains no real-world actuator interface.
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass

from artificial_human.neural.world_model import (
    PredictiveWorldModel,
    PredictionError,
    WorldState,
)


def _clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


def _finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class ForwardMotorCommand:
    """A bounded simulated command and its predicted sensory consequence."""

    tick: int
    command: float
    predicted_next_velocity: float
    predicted_next_position: float
    target_error: float
    uncertainty: float


@dataclass(frozen=True, slots=True)
class AdaptiveControlComparison:
    """Paired fixed-model and adaptive-model tracking metrics."""

    steps: int
    feedback_delay_steps: int
    sensor_noise_std: float
    baseline_rmse: float
    adaptive_rmse: float
    relative_improvement: float
    baseline_post_shift_rmse: float
    adaptive_post_shift_rmse: float
    adaptive_early_recovery_rmse: float
    adaptive_late_recovery_rmse: float
    learned_damping: float
    learned_control_gain: float
    learned_disturbance_bias: float
    seed: int


class AdaptiveForwardController:
    """One-dimensional forward model with bounded, delayed-error adaptation.

    The forward model predicts next velocity from current velocity and motor
    command. A simple inverse step chooses the command expected to produce a
    desired next velocity. The forward model is the existing bounded online
    linear predictor, initialized with an explicit engineering prior.
    """

    VERSION = "adaptive-forward-control-v1"
    MAX_FEEDBACK_DELAY = 32

    def __init__(
        self,
        *,
        learning_rate: float = 0.18,
        position_gain: float = 0.55,
        feedback_delay_steps: int = 0,
        learning_enabled: bool = True,
        model: PredictiveWorldModel | None = None,
    ) -> None:
        _finite(learning_rate, "learning_rate")
        _finite(position_gain, "position_gain")
        if not 0.001 <= learning_rate <= 1.0:
            raise ValueError("learning_rate must be in [0.001, 1]")
        if not 0.05 <= position_gain <= 1.0:
            raise ValueError("position_gain must be in [0.05, 1]")
        if (
            isinstance(feedback_delay_steps, bool)
            or not isinstance(feedback_delay_steps, int)
            or not 0 <= feedback_delay_steps <= self.MAX_FEEDBACK_DELAY
        ):
            raise ValueError("feedback_delay_steps must be in [0, 32]")
        if not isinstance(learning_enabled, bool):
            raise ValueError("learning_enabled must be a boolean")

        selected_model = model or PredictiveWorldModel(
            1, 1, learning_rate=learning_rate, residual_decay=0.95
        )
        if selected_model.state_dimensions != 1 or selected_model.action_dimensions != 1:
            raise ValueError("adaptive forward controller requires a 1D state and 1D action model")
        if model is None:
            # Features are [bias, current velocity, motor command].
            selected_model.weights[0] = [0.0, 0.65, 0.20]
        self.model = selected_model
        self.learning_rate = float(learning_rate)
        self.position_gain = float(position_gain)
        self.feedback_delay_steps = feedback_delay_steps
        self.learning_enabled = learning_enabled
        self._initial_weights = [row.copy() for row in self.model.weights]
        self._initial_variance = self.model.residual_variance.copy()
        self._pending: deque[tuple[WorldState, tuple[float, ...], WorldState]] = deque()

    @property
    def pending_feedback(self) -> int:
        return len(self._pending)

    def command(
        self,
        position: float,
        velocity: float,
        target_position: float,
        *,
        tick: int,
    ) -> ForwardMotorCommand:
        """Predict the next sensory state and issue a bounded simulated command."""
        for name, value in (
            ("position", position),
            ("velocity", velocity),
            ("target_position", target_position),
        ):
            _finite(value, name)
        if not -1.0 <= velocity <= 1.0:
            raise ValueError("velocity must be in [-1, 1]")
        if not -1.0 <= target_position <= 1.0:
            raise ValueError("target_position must be in [-1, 1]")
        if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
            raise ValueError("tick must be a non-negative integer")

        bias, damping, learned_gain = self.model.weights[0]
        control_gain = max(0.05, learned_gain)
        desired_velocity = _clamp(
            self.position_gain * (target_position - position)
        )
        raw_command = (
            desired_velocity - damping * velocity - bias
        ) / control_gain
        motor_command = _clamp(raw_command)

        state = WorldState((float(velocity),), tick)
        prediction = self.model.predict(state, (motor_command,))
        predicted_velocity = prediction.state.values[0]
        return ForwardMotorCommand(
            tick=tick,
            command=round(motor_command, 8),
            predicted_next_velocity=predicted_velocity,
            predicted_next_position=round(position + predicted_velocity, 8),
            target_error=round(target_position - position, 8),
            uncertainty=prediction.standard_deviation[0],
        )

    def observe(
        self,
        previous_velocity: float,
        command: float,
        observed_next_velocity: float,
        *,
        tick: int,
    ) -> PredictionError | None:
        """Queue sensory feedback and learn only after the configured delay."""
        if not self.learning_enabled:
            return None
        for name, value in (
            ("previous_velocity", previous_velocity),
            ("command", command),
            ("observed_next_velocity", observed_next_velocity),
        ):
            _finite(value, name)
        if not -1.0 <= previous_velocity <= 1.0:
            raise ValueError("previous_velocity must be in [-1, 1]")
        if not -1.0 <= command <= 1.0:
            raise ValueError("command must be in [-1, 1]")
        if not -1.0 <= observed_next_velocity <= 1.0:
            raise ValueError("observed_next_velocity must be in [-1, 1]")
        if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
            raise ValueError("tick must be a non-negative integer")

        state = WorldState((float(previous_velocity),), tick)
        action = (float(command),)
        observed = WorldState((float(observed_next_velocity),), tick + 1)
        self._pending.append((state, action, observed))
        if len(self._pending) <= self.feedback_delay_steps:
            return None

        prior_state, prior_action, prior_observation = self._pending.popleft()
        return self.model.observe(
            prior_state,
            prior_action,
            prior_observation,
            update=True,
        )

    def snapshot(self) -> dict[str, object]:
        """Return a bounded, inspectable snapshot of the adaptive controller."""
        return {
            "version": self.VERSION,
            "learning_enabled": self.learning_enabled,
            "feedback_delay_steps": self.feedback_delay_steps,
            "pending_feedback": self.pending_feedback,
            "observations": self.model.observations,
            "bias": round(self.model.weights[0][0], 8),
            "damping": round(self.model.weights[0][1], 8),
            "control_gain": round(self.model.weights[0][2], 8),
            "residual_standard_deviation": round(
                math.sqrt(self.model.residual_variance[0]), 8
            ),
        }

    def reset(self) -> None:
        """Discard pending sensory feedback and restore the source-defined prior."""
        self.model.weights = [row.copy() for row in self._initial_weights]
        self.model.residual_variance = self._initial_variance.copy()
        self.model.observations = 0
        self._pending.clear()


def _tracking_rmse(errors: list[float]) -> float:
    return math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else 0.0


def _simulate_tracking(
    controller: AdaptiveForwardController,
    *,
    steps: int,
    sensor_noise_std: float,
    seed: int,
) -> tuple[list[float], int]:
    rng = random.Random(seed)
    position = 0.0
    velocity = 0.0
    errors: list[float] = []
    shift_tick = steps // 2

    for tick in range(steps):
        target = 0.65 * math.sin(tick * 0.025)
        sensed_position = position + rng.gauss(0.0, sensor_noise_std)
        sensed_velocity = _clamp(
            velocity + rng.gauss(0.0, sensor_noise_std)
        )
        motor = controller.command(
            sensed_position,
            sensed_velocity,
            target,
            tick=tick,
        )

        # Deterministic plant with a mid-run disturbance shift.
        disturbance = 0.12 + (0.08 if tick >= shift_tick else 0.0)
        next_velocity = _clamp(
            0.82 * velocity + 0.28 * motor.command + disturbance
        )
        next_position = position + next_velocity
        observed_velocity = _clamp(
            next_velocity + rng.gauss(0.0, sensor_noise_std)
        )
        controller.observe(
            sensed_velocity,
            motor.command,
            observed_velocity,
            tick=tick,
        )

        position = next_position
        velocity = next_velocity
        errors.append(target - position)

    return errors, shift_tick


def run_adaptive_control_comparison(
    *,
    steps: int = 600,
    feedback_delay_steps: int = 2,
    sensor_noise_std: float = 0.02,
    seed: int = 23,
) -> AdaptiveControlComparison:
    """Compare a frozen feedback controller with an adapting forward model.

    Both controllers receive the same fixed-seed sensory-noise stream and act
    in the same deterministic one-dimensional plant. A disturbance changes at
    the midpoint. Metrics report tracking error after a 100-step warm-up and
    post-shift recovery; the task is a software experiment, not biological proof.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or not 300 <= steps <= 10_000:
        raise ValueError("steps must be in [300, 10000]")
    if (
        isinstance(feedback_delay_steps, bool)
        or not isinstance(feedback_delay_steps, int)
        or not 0 <= feedback_delay_steps <= AdaptiveForwardController.MAX_FEEDBACK_DELAY
    ):
        raise ValueError("feedback_delay_steps must be in [0, 32]")
    _finite(sensor_noise_std, "sensor_noise_std")
    if not 0.0 <= sensor_noise_std <= 0.1:
        raise ValueError("sensor_noise_std must be in [0, 0.1]")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    baseline = AdaptiveForwardController(
        feedback_delay_steps=feedback_delay_steps,
        learning_enabled=False,
    )
    adaptive = AdaptiveForwardController(
        feedback_delay_steps=feedback_delay_steps,
        learning_enabled=True,
    )
    baseline_errors, shift_tick = _simulate_tracking(
        baseline,
        steps=steps,
        sensor_noise_std=sensor_noise_std,
        seed=seed,
    )
    adaptive_errors, _ = _simulate_tracking(
        adaptive,
        steps=steps,
        sensor_noise_std=sensor_noise_std,
        seed=seed,
    )

    warmup = 100
    recovery_window = 60
    baseline_rmse = _tracking_rmse(baseline_errors[warmup:])
    adaptive_rmse = _tracking_rmse(adaptive_errors[warmup:])
    baseline_post = _tracking_rmse(baseline_errors[shift_tick + recovery_window:])
    adaptive_post = _tracking_rmse(adaptive_errors[shift_tick + recovery_window:])
    early_recovery = _tracking_rmse(
        adaptive_errors[shift_tick:shift_tick + recovery_window]
    )
    late_recovery = _tracking_rmse(adaptive_errors[-recovery_window:])
    bias, damping, gain = adaptive.model.weights[0]

    return AdaptiveControlComparison(
        steps=steps,
        feedback_delay_steps=feedback_delay_steps,
        sensor_noise_std=sensor_noise_std,
        baseline_rmse=round(baseline_rmse, 8),
        adaptive_rmse=round(adaptive_rmse, 8),
        relative_improvement=round(
            (baseline_rmse - adaptive_rmse) / baseline_rmse
            if baseline_rmse else 0.0,
            8,
        ),
        baseline_post_shift_rmse=round(baseline_post, 8),
        adaptive_post_shift_rmse=round(adaptive_post, 8),
        adaptive_early_recovery_rmse=round(early_recovery, 8),
        adaptive_late_recovery_rmse=round(late_recovery, 8),
        learned_damping=round(damping, 8),
        learned_control_gain=round(gain, 8),
        learned_disturbance_bias=round(bias, 8),
        seed=seed,
    )
