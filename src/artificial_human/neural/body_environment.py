"""Deterministic simulated body/environment with a closed sensory-action loop.

All inputs are synthetic, all actions remain inside a bounded simulator, and
runtime learning remains in process memory. This is an engineering integration
experiment, not a biological body or a claim of organism-level consciousness.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from artificial_human.neural.action_selection import (
    ActionSelectionCircuit,
    ActionSpec,
    SafeActionSpace,
)
from artificial_human.neural.adaptive_control import AdaptiveForwardController
from artificial_human.neural.circuits import build_recurrent_microcircuit
from artificial_human.neural.integration import (
    MultisensoryIntegrator,
    SensoryEstimate,
    estimate_from_percept,
)
from artificial_human.neural.sensory import (
    RetentionPolicy,
    SensoryEvent,
    SensoryModality,
    SensoryPermissionLedger,
    SensoryPipeline,
)


def _clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


def _encode_sensory_value(value: float) -> float:
    """Invert the population encoder's tanh normalization for physical values."""
    return math.atanh(_clamp(value, -0.95, 0.95))


def _require_finite(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class BodyState:
    tick: int
    position: float
    velocity: float
    goal_position: float


@dataclass(frozen=True, slots=True)
class BodyTransition:
    previous: BodyState
    current: BodyState
    command: float
    disturbance: float
    reached_goal: bool


class SimulatedBodyEnvironment:
    """A one-dimensional bounded plant; it cannot access physical devices."""

    def __init__(
        self,
        *,
        max_steps: int = 240,
        disturbance_shift_tick: int | None = None,
        initial_position: float = -0.75,
        goal_position: float = 0.65,
    ) -> None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 80 <= max_steps <= 500:
            raise ValueError("max_steps must be in [80, 500]")
        shift = max_steps // 3 if disturbance_shift_tick is None else disturbance_shift_tick
        if isinstance(shift, bool) or not isinstance(shift, int) or not 1 <= shift < max_steps:
            raise ValueError("disturbance_shift_tick must be within the episode")
        _require_finite(initial_position, "initial_position")
        _require_finite(goal_position, "goal_position")
        if not -1.0 <= initial_position <= 1.0 or not -1.0 <= goal_position <= 1.0:
            raise ValueError("body and goal positions must be in [-1, 1]")
        self.max_steps = max_steps
        self.disturbance_shift_tick = shift
        self.initial_position = float(initial_position)
        self.goal_position = float(goal_position)
        self._state = BodyState(0, self.initial_position, 0.0, self.goal_position)

    @property
    def state(self) -> BodyState:
        return self._state

    def reset(self) -> BodyState:
        self._state = BodyState(0, self.initial_position, 0.0, self.goal_position)
        return self._state

    def step(self, command: float) -> BodyTransition:
        _require_finite(command, "command")
        if not -1.0 <= command <= 1.0:
            raise ValueError("simulated motor command must be in [-1, 1]")
        if self._state.tick >= self.max_steps:
            raise RuntimeError("episode step budget exhausted")

        previous = self._state
        disturbance = 0.0 if previous.tick < self.disturbance_shift_tick else 0.1
        next_velocity = _clamp(
            0.82 * previous.velocity + 0.28 * command + disturbance
        )
        next_position = _clamp(previous.position + next_velocity)
        current = BodyState(
            tick=previous.tick + 1,
            position=next_position,
            velocity=next_velocity,
            goal_position=self.goal_position,
        )
        reached_goal = (
            abs(self.goal_position - next_position) <= 0.06
            and abs(next_velocity) <= 0.06
        )
        self._state = current
        return BodyTransition(
            previous=previous,
            current=current,
            command=float(command),
            disturbance=disturbance,
            reached_goal=reached_goal,
        )


@dataclass(frozen=True, slots=True)
class BodyTrace:
    tick: int
    context: str
    selected_action: str
    visual_position: float
    proprioceptive_position: float
    fused_position: float
    estimated_velocity: float
    motor_command: float
    predicted_next_velocity: float
    actual_position: float
    actual_velocity: float
    reward: float
    feedback_tick: int | None
    prediction_error: float | None
    neural_spikes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ClosedLoopTaskResult:
    seed: int
    steps: int
    success: bool
    disturbance_shift_tick: int
    initial_distance: float
    final_distance: float
    final_position: float
    final_velocity: float
    action_counts: tuple[tuple[str, int], ...]
    mean_abs_prediction_error: float
    mean_abs_prediction_error_after_shift: float
    max_abs_prediction_error: float
    learned_disturbance_bias: float
    sensory_retained_count: int
    neural_spike_count: int
    trace: tuple[BodyTrace, ...]


def _build_sensory_pipeline(max_steps: int) -> SensoryPipeline:
    ledger = SensoryPermissionLedger()
    for modality in (SensoryModality.VISION, SensoryModality.PROPRIOCEPTION):
        ledger.grant(
            modality,
            purpose="synthetic closed-loop body/environment experiment",
            granted_at=0,
            expires_at=max_steps + 2,
            retention=RetentionPolicy.EPHEMERAL,
            max_events=3 * max_steps + 10,
            max_samples=4,
            explicit_consent=True,
        )
    return SensoryPipeline(ledger=ledger, capacity=64)


def _require_estimate(value: SensoryEstimate | None, label: str) -> SensoryEstimate:
    if value is None:
        raise RuntimeError(f"{label} sensory estimate could not be decoded")
    return value


def _train_contextual_policy(seed: int) -> ActionSelectionCircuit:
    """Calibrate context/action values with bounded signed reward signals."""
    actions = SafeActionSpace(
        (
            ActionSpec("approach", "Move toward the simulated goal"),
            ActionSpec("hold", "Stabilize the current simulated position"),
            ActionSpec("recover", "Return toward the simulated goal after overshoot"),
        )
    )
    policy = ActionSelectionCircuit(
        actions,
        learning_rate=0.15,
        exploration_rate=0.0,
        exploration_bonus=0.0,
        seed=seed,
    )
    preferred = {
        "far": "approach",
        "at_goal": "hold",
        "overshot": "recover",
    }
    for context, preferred_action in preferred.items():
        for _ in range(40):
            for action_id in actions.identifiers:
                reward = 1.0 if action_id == preferred_action else -1.0
                policy.observe(context, action_id, reward, terminal=True)
    return policy


def _context_for_state(position: float, velocity: float, goal: float) -> str:
    if position > goal + 0.07:
        return "overshot"
    if abs(goal - position) <= 0.12 and abs(velocity) <= 0.08:
        return "at_goal"
    return "far"


def run_closed_loop_task(
    *,
    max_steps: int = 240,
    feedback_delay_steps: int = 2,
    sensor_noise_std: float = 0.01,
    seed: int = 23,
) -> ClosedLoopTaskResult:
    """Run a reproducible perception → action → outcome → learning episode."""
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 80 <= max_steps <= 500:
        raise ValueError("max_steps must be in [80, 500]")
    if (
        isinstance(feedback_delay_steps, bool)
        or not isinstance(feedback_delay_steps, int)
        or not 0 <= feedback_delay_steps <= AdaptiveForwardController.MAX_FEEDBACK_DELAY
    ):
        raise ValueError("feedback_delay_steps must be in [0, 32]")
    _require_finite(sensor_noise_std, "sensor_noise_std")
    if not 0.0 <= sensor_noise_std <= 0.05:
        raise ValueError("sensor_noise_std must be in [0, 0.05]")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    rng = random.Random(seed)
    environment = SimulatedBodyEnvironment(max_steps=max_steps)
    pipeline = _build_sensory_pipeline(max_steps)
    integrator = MultisensoryIntegrator()
    neural_simulation = build_recurrent_microcircuit()
    controller = AdaptiveForwardController(
        feedback_delay_steps=feedback_delay_steps,
        learning_rate=0.18,
    )
    policy = _train_contextual_policy(seed)

    initial_state = environment.state
    initial_distance = abs(initial_state.goal_position - initial_state.position)
    trace: list[BodyTrace] = []
    action_counts = {action: 0 for action in policy.action_space.identifiers}
    prediction_errors: list[tuple[int, float]] = []

    state = environment.state
    initial_velocity_event = SensoryEvent(
        modality=SensoryModality.PROPRIOCEPTION,
        values=(_encode_sensory_value(_clamp(state.velocity + rng.gauss(0.0, sensor_noise_std))),),
        timestamp=0,
        confidence=0.98,
    )
    initial_velocity_percept = pipeline.ingest(initial_velocity_event)
    current_velocity_estimate = _require_estimate(
        estimate_from_percept(
            initial_velocity_percept,
            source_id="simulated-body-velocity",
        ),
        "proprioceptive velocity",
    ).value

    for tick in range(max_steps):
        state = environment.state
        visual_value = _clamp(
            state.position + rng.gauss(0.0, sensor_noise_std)
        )
        proprioceptive_value = _clamp(
            state.position + rng.gauss(0.0, sensor_noise_std * 0.5)
        )
        vision_percept = pipeline.ingest(
            SensoryEvent(
                SensoryModality.VISION,
                (_encode_sensory_value(visual_value),),
                timestamp=tick,
                confidence=0.95,
            )
        )
        proprioception_percept = pipeline.ingest(
            SensoryEvent(
                SensoryModality.PROPRIOCEPTION,
                (_encode_sensory_value(proprioceptive_value),),
                timestamp=tick,
                confidence=0.98,
            )
        )
        vision_estimate = _require_estimate(
            estimate_from_percept(
                vision_percept,
                source_id="simulated-body-pose",
            ),
            "vision",
        )
        proprioception_estimate = _require_estimate(
            estimate_from_percept(
                proprioception_percept,
                source_id="simulated-body-pose",
            ),
            "proprioception",
        )
        integrated = integrator.integrate(
            (vision_estimate, proprioception_estimate),
            timestamp=tick,
            target_modality=SensoryModality.VISION,
            expected_modalities=(
                SensoryModality.VISION,
                SensoryModality.PROPRIOCEPTION,
            ),
        )
        fused_position = integrated.value
        velocity = current_velocity_estimate

        # The encoded/fused percept drives an actual recurrent LIF microcircuit.
        neural_step = neural_simulation.step(
            {"sensory": 160.0 + 20.0 * fused_position}
        )
        context = _context_for_state(
            fused_position,
            velocity,
            state.goal_position,
        )
        choice = policy.select(context, explore=False)
        action_counts[choice.action_id] += 1

        target_position = (
            fused_position if choice.action_id == "hold" else state.goal_position
        )
        motor = controller.command(
            fused_position,
            velocity,
            target_position,
            tick=tick,
        )
        transition = environment.step(motor.command)

        next_velocity_event = SensoryEvent(
            modality=SensoryModality.PROPRIOCEPTION,
            values=(
                _encode_sensory_value(
                    _clamp(
                        transition.current.velocity
                        + rng.gauss(0.0, sensor_noise_std * 0.5)
                    )
                ),
            ),
            timestamp=tick + 1,
            confidence=0.98,
        )
        next_velocity_percept = pipeline.ingest(next_velocity_event)
        next_velocity_estimate = _require_estimate(
            estimate_from_percept(
                next_velocity_percept,
                source_id="simulated-body-velocity",
            ),
            "next proprioceptive velocity",
        )
        prediction_update = controller.observe(
            velocity,
            motor.command,
            next_velocity_estimate.value,
            tick=tick,
        )

        distance_before = abs(state.goal_position - state.position)
        distance_after = abs(
            transition.current.goal_position - transition.current.position
        )
        progress = distance_before - distance_after
        stability = abs(state.velocity) - abs(transition.current.velocity)
        reward = _clamp(2.0 * progress + 0.15 * stability - 0.01)
        if context == "at_goal" and distance_after <= 0.08:
            reward = _clamp(reward + 0.15)
        policy.observe(context, choice.action_id, reward, terminal=True)

        feedback_tick: int | None = None
        prediction_error: float | None = None
        if prediction_update is not None:
            feedback_tick = prediction_update.predicted.timestamp
            prediction_error = prediction_update.residual[0]
            prediction_errors.append((feedback_tick, abs(prediction_error)))

        trace.append(
            BodyTrace(
                tick=tick,
                context=context,
                selected_action=choice.action_id,
                visual_position=round(visual_value, 8),
                proprioceptive_position=round(proprioceptive_value, 8),
                fused_position=round(fused_position, 8),
                estimated_velocity=round(velocity, 8),
                motor_command=motor.command,
                predicted_next_velocity=motor.predicted_next_velocity,
                actual_position=transition.current.position,
                actual_velocity=transition.current.velocity,
                reward=round(reward, 8),
                feedback_tick=feedback_tick,
                prediction_error=(
                    round(prediction_error, 8)
                    if prediction_error is not None else None
                ),
                neural_spikes=neural_step.spikes,
            )
        )
        current_velocity_estimate = next_velocity_estimate.value

    final_state = environment.state
    final_distance = abs(final_state.goal_position - final_state.position)
    error_values = [value for _, value in prediction_errors]
    post_shift_errors = [
        value for tick, value in prediction_errors
        if tick >= environment.disturbance_shift_tick
    ]
    learned_weights = controller.model.weights[0]
    return ClosedLoopTaskResult(
        seed=seed,
        steps=len(trace),
        success=final_distance <= 0.12 and abs(final_state.velocity) <= 0.12,
        disturbance_shift_tick=environment.disturbance_shift_tick,
        initial_distance=round(initial_distance, 8),
        final_distance=round(final_distance, 8),
        final_position=round(final_state.position, 8),
        final_velocity=round(final_state.velocity, 8),
        action_counts=tuple(sorted(action_counts.items())),
        mean_abs_prediction_error=round(
            sum(error_values) / len(error_values) if error_values else 0.0,
            8,
        ),
        mean_abs_prediction_error_after_shift=round(
            sum(post_shift_errors) / len(post_shift_errors)
            if post_shift_errors else 0.0,
            8,
        ),
        max_abs_prediction_error=round(max(error_values, default=0.0), 8),
        learned_disturbance_bias=round(learned_weights[0], 8),
        sensory_retained_count=pipeline.retained_count,
        neural_spike_count=sum(len(item.neural_spikes) for item in trace),
        trace=tuple(trace),
    )
