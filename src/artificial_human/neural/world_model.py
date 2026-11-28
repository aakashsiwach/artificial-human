"""Bounded deterministic predictive world-state model and error diagnostics.

The model is an online linear transition predictor over explicitly supplied
normalized state/action vectors. It is a computational baseline, not a claim
about biological predictive coding or a complete physical world model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence

from artificial_human.brain.retrieval import tokenize
from artificial_human.cognition.perception import Percept
from artificial_human.cognition.prediction import PredictionSystem


@dataclass(frozen=True, slots=True)
class WorldState:
    """Compact normalized state vector at an integer simulation tick."""

    values: tuple[float, ...]
    timestamp: int

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple) or not 1 <= len(self.values) <= 32:
            raise ValueError("world state must contain 1..32 values")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not -1.0 <= value <= 1.0
            for value in self.values
        ):
            raise ValueError("world-state values must be finite and in [-1, 1]")
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, int) or self.timestamp < 0:
            raise ValueError("world-state timestamp must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class WorldPrediction:
    """One-step state prediction with per-coordinate residual uncertainty."""

    state: WorldState
    standard_deviation: tuple[float, ...]
    model_version: str


@dataclass(frozen=True, slots=True)
class PredictionError:
    """Distinct sensory-change, transition-model and reward/goal error channels."""

    predicted: WorldState
    observed: WorldState
    residual: tuple[float, ...]
    model_error_mae: float
    sensory_surprise: float
    reward_goal_error: float
    uncertainty_before: float
    uncertainty_after: float
    updated: bool


@dataclass(frozen=True, slots=True)
class RecoveryEvaluation:
    first_error: float
    final_error: float
    uncertainty_before_shift: float
    uncertainty_after_shift: float
    uncertainty_after_recovery: float


@dataclass(frozen=True, slots=True)
class WorldModelComparison:
    training_samples: int
    heldout_samples: int
    frozen_model_mae: float
    learned_model_mae: float
    relative_improvement: float
    lexical_novelty_accuracy: float
    lexical_novelty_precision: float
    lexical_novelty_recall: float
    recovery: RecoveryEvaluation


class PredictiveWorldModel:
    """Online normalized-LMS transition model with bounded weights and residual variance."""

    VERSION = "bounded-linear-transition-v1"
    MIN_RESIDUAL_VARIANCE = 1e-6
    MAX_RESIDUAL_VARIANCE = 4.0
    MAX_ABS_WEIGHT = 4.0

    def __init__(
        self,
        state_dimensions: int,
        action_dimensions: int,
        *,
        learning_rate: float = 0.5,
        residual_decay: float = 0.9,
    ) -> None:
        if isinstance(state_dimensions, bool) or not isinstance(state_dimensions, int) or not 1 <= state_dimensions <= 32:
            raise ValueError("state_dimensions must be in [1, 32]")
        if isinstance(action_dimensions, bool) or not isinstance(action_dimensions, int) or not 1 <= action_dimensions <= 16:
            raise ValueError("action_dimensions must be in [1, 16]")
        if (
            isinstance(learning_rate, bool)
            or not math.isfinite(learning_rate)
            or not 0.001 <= learning_rate <= 1.0
        ):
            raise ValueError("learning_rate must be in [0.001, 1]")
        if (
            isinstance(residual_decay, bool)
            or not math.isfinite(residual_decay)
            or not 0.0 <= residual_decay < 1.0
        ):
            raise ValueError("residual_decay must be in [0, 1)")
        self.state_dimensions = state_dimensions
        self.action_dimensions = action_dimensions
        self.learning_rate = learning_rate
        self.residual_decay = residual_decay
        feature_count = 1 + state_dimensions + action_dimensions
        self.weights = [[0.0] * feature_count for _ in range(state_dimensions)]
        self.residual_variance = [0.25] * state_dimensions
        self.observations = 0

    def _features(self, state: WorldState, action: tuple[float, ...]) -> tuple[float, ...]:
        if len(state.values) != self.state_dimensions:
            raise ValueError("world-state dimension does not match model")
        if len(action) != self.action_dimensions:
            raise ValueError("action dimension does not match model")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not -1.0 <= value <= 1.0
            for value in action
        ):
            raise ValueError("action values must be finite and in [-1, 1]")
        return (1.0, *state.values, *action)

    def predict(
        self,
        state: WorldState,
        action: tuple[float, ...],
    ) -> WorldPrediction:
        """Predict the next state without changing model parameters."""
        features = self._features(state, action)
        predicted = tuple(
            min(
                1.0,
                max(
                    -1.0,
                    sum(weight * feature for weight, feature in zip(row, features, strict=True)),
                ),
            )
            for row in self.weights
        )
        return WorldPrediction(
            state=WorldState(predicted, state.timestamp + 1),
            standard_deviation=tuple(
                math.sqrt(value) for value in self.residual_variance
            ),
            model_version=self.VERSION,
        )

    def rollout(
        self,
        initial_state: WorldState,
        actions: Sequence[tuple[float, ...]],
    ) -> tuple[WorldPrediction, ...]:
        """Generate a multi-step open-loop rollout using predicted states as inputs."""
        current = initial_state
        predictions: list[WorldPrediction] = []
        accumulated_variance = [0.0] * self.state_dimensions
        for action in actions:
            prediction = self.predict(current, action)
            accumulated_uncertainty = tuple(
                math.sqrt(
                    accumulated_variance[index]
                    + prediction.standard_deviation[index] ** 2
                )
                for index in range(self.state_dimensions)
            )
            accumulated_variance = [
                value * value for value in accumulated_uncertainty
            ]
            predictions.append(
                WorldPrediction(
                    state=prediction.state,
                    standard_deviation=accumulated_uncertainty,
                    model_version=prediction.model_version,
                )
            )
            current = prediction.state
        return tuple(predictions)

    def observe(
        self,
        state: WorldState,
        action: tuple[float, ...],
        observed_next_state: WorldState,
        *,
        expected_reward: float = 0.0,
        observed_reward: float = 0.0,
        update: bool = True,
    ) -> PredictionError:
        """Measure errors before optional online learning from the observed transition.

        sensory_surprise is the normalized magnitude of observed state change
        (not a statistical surprise probability). model_error_mae is prediction
        residual magnitude. reward_goal_error is expected minus observed reward.
        """
        if len(observed_next_state.values) != self.state_dimensions:
            raise ValueError("observed state dimension does not match model")
        if observed_next_state.timestamp != state.timestamp + 1:
            raise ValueError("observed next-state timestamp must advance by one tick")
        if any(
            isinstance(value, bool) or not math.isfinite(value)
            for value in (expected_reward, observed_reward)
        ):
            raise ValueError("reward values must be finite")
        features = self._features(state, action)
        prediction = self.predict(state, action)
        residual = tuple(
            observed - predicted
            for observed, predicted in zip(
                observed_next_state.values, prediction.state.values, strict=True
            )
        )
        model_error = sum(abs(value) for value in residual) / self.state_dimensions
        sensory_surprise = sum(
            abs(after - before)
            for before, after in zip(
                state.values, observed_next_state.values, strict=True
            )
        ) / (2.0 * self.state_dimensions)
        uncertainty_before = sum(prediction.standard_deviation) / self.state_dimensions

        if update:
            norm = 1.0 + sum(value * value for value in features)
            for output_index, error in enumerate(residual):
                row = self.weights[output_index]
                for feature_index, feature in enumerate(features):
                    row[feature_index] = min(
                        self.MAX_ABS_WEIGHT,
                        max(
                            -self.MAX_ABS_WEIGHT,
                            row[feature_index]
                            + self.learning_rate * error * feature / norm,
                        ),
                    )
                variance = (
                    self.residual_decay * self.residual_variance[output_index]
                    + (1.0 - self.residual_decay) * error * error
                )
                self.residual_variance[output_index] = min(
                    self.MAX_RESIDUAL_VARIANCE,
                    max(self.MIN_RESIDUAL_VARIANCE, variance),
                )
            self.observations += 1

        uncertainty_after = sum(
            math.sqrt(value) for value in self.residual_variance
        ) / self.state_dimensions
        return PredictionError(
            predicted=prediction.state,
            observed=observed_next_state,
            residual=residual,
            model_error_mae=round(model_error, 10),
            sensory_surprise=round(sensory_surprise, 10),
            reward_goal_error=round(expected_reward - observed_reward, 10),
            uncertainty_before=round(uncertainty_before, 10),
            uncertainty_after=round(uncertainty_after, 10),
            updated=update,
        )

    def clone(self) -> PredictiveWorldModel:
        """Copy bounded model parameters for frozen-vs-adaptive comparisons."""
        cloned = PredictiveWorldModel(
            self.state_dimensions,
            self.action_dimensions,
            learning_rate=self.learning_rate,
            residual_decay=self.residual_decay,
        )
        cloned.weights = [row.copy() for row in self.weights]
        cloned.residual_variance = self.residual_variance.copy()
        cloned.observations = self.observations
        return cloned


def _clamp(value: float) -> float:
    return min(1.0, max(-1.0, value))


def _normal_transition(
    state: WorldState, action: tuple[float, ...]
) -> WorldState:
    position, velocity, target = state.values
    force = action[0]
    return WorldState(
        (
            _clamp(position + 0.25 * velocity + 0.12 * force),
            _clamp(0.72 * velocity + 0.20 * force),
            target,
        ),
        state.timestamp + 1,
    )


def _shifted_transition(
    state: WorldState, action: tuple[float, ...]
) -> WorldState:
    normal = _normal_transition(state, action)
    return WorldState(
        (_clamp(normal.values[0] + 0.45), normal.values[1], normal.values[2]),
        normal.timestamp,
    )


def _training_sample(index: int) -> tuple[WorldState, tuple[float, ...], WorldState]:
    state = WorldState(
        (
            _clamp(0.78 * math.sin(index * 0.173)),
            _clamp(0.42 * math.cos(index * 0.119)),
            _clamp(0.72 * math.sin(index * 0.071)),
        ),
        index,
    )
    action = (((index * 7) % 3) - 1) * 0.7
    return state, (action,), _normal_transition(state, (action,))


def _heldout_sample(index: int) -> tuple[WorldState, tuple[float, ...], WorldState, bool]:
    state = WorldState(
        (
            _clamp(0.74 * math.sin((index + 0.37) * 0.137)),
            _clamp(0.28 * math.cos((index + 0.19) * 0.103)),
            _clamp(0.68 * math.sin((index + 0.11) * 0.067)),
        ),
        10_000 + index,
    )
    action = ((((index * 11) % 3) - 1) * 0.55,)
    shifted = index % 5 == 0
    observed = (
        _shifted_transition(state, action)
        if shifted
        else _normal_transition(state, action)
    )
    return state, action, observed, shifted


def _lexical_novelty(label: str, index: int) -> float:
    percept = Percept(
        id=f"synthetic-transition-{index}",
        content=label,
        modality="language",
        source="deterministic-evaluation",
        observed_at=f"tick:{index}",
        tokens=tuple(tokenize(label)),
    )
    return PredictionSystem().assess(
        percept, ("agent moved slowly",)
    ).novelty


def _classification_metrics(
    expected: list[bool], predicted: list[bool]
) -> tuple[float, float, float]:
    true_positive = sum(actual and guess for actual, guess in zip(expected, predicted, strict=True))
    true_negative = sum(not actual and not guess for actual, guess in zip(expected, predicted, strict=True))
    false_positive = sum(not actual and guess for actual, guess in zip(expected, predicted, strict=True))
    false_negative = sum(actual and not guess for actual, guess in zip(expected, predicted, strict=True))
    total = len(expected)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    accuracy = (true_positive + true_negative) / total if total else 0.0
    return round(accuracy, 8), round(precision, 8), round(recall, 8)


def run_world_model_comparison(
    training_samples: int = 240,
    heldout_samples: int = 100,
) -> WorldModelComparison:
    """Compare frozen and learned dynamics, lexical novelty, and shift recovery."""
    if isinstance(training_samples, bool) or not isinstance(training_samples, int) or not 40 <= training_samples <= 10_000:
        raise ValueError("training_samples must be in [40, 10000]")
    if isinstance(heldout_samples, bool) or not isinstance(heldout_samples, int) or not 20 <= heldout_samples <= 2_000:
        raise ValueError("heldout_samples must be in [20, 2000]")

    frozen = PredictiveWorldModel(3, 1)
    learned = frozen.clone()
    for index in range(training_samples):
        state, action, observed = _training_sample(index)
        learned.observe(
            state, action, observed,
            expected_reward=0.0, observed_reward=0.0, update=True,
        )

    frozen_errors: list[float] = []
    learned_errors: list[float] = []
    lexical_scores: list[float] = []
    error_flags: list[bool] = []
    for index in range(heldout_samples):
        state, action, observed, shifted = _heldout_sample(index)
        frozen_prediction = frozen.predict(state, action)
        learned_prediction = learned.predict(state, action)
        frozen_error = sum(
            abs(a - b)
            for a, b in zip(frozen_prediction.state.values, observed.values, strict=True)
        ) / 3.0
        learned_error = sum(
            abs(a - b)
            for a, b in zip(learned_prediction.state.values, observed.values, strict=True)
        ) / 3.0
        frozen_errors.append(frozen_error)
        learned_errors.append(learned_error)
        error_flags.append(learned_error >= 0.10)
        delta = abs(observed.values[0] - state.values[0])
        label = "agent moved rapidly" if delta >= 0.35 else "agent moved slowly"
        lexical_scores.append(_lexical_novelty(label, index))

    lexical_predictions = [score >= 0.5 for score in lexical_scores]
    lexical_accuracy, lexical_precision, lexical_recall = _classification_metrics(
        error_flags, lexical_predictions
    )
    recovery_model = learned.clone()
    first_recovery_error = 0.0
    final_recovery_error = 0.0
    uncertainty_before_shift = sum(
        math.sqrt(value) for value in recovery_model.residual_variance
    ) / 3.0
    uncertainty_after_shift = uncertainty_before_shift
    uncertainty_after_recovery = uncertainty_before_shift
    for index in range(32):
        state = WorldState(
            (
                _clamp(0.3 * math.sin(index * 0.23)),
                _clamp(0.12 * math.cos(index * 0.17)),
                0.4,
            ),
            20_000 + index,
        )
        action = ((0.4 if index % 2 else -0.4),)
        observed = _shifted_transition(state, action)
        error = recovery_model.observe(
            state, action, observed,
            expected_reward=0.5, observed_reward=0.25,
            update=True,
        )
        if index == 0:
            first_recovery_error = error.model_error_mae
            uncertainty_after_shift = error.uncertainty_after
        if index >= 27:
            final_recovery_error += error.model_error_mae / 5.0
        if index == 31:
            uncertainty_after_recovery = error.uncertainty_after

    frozen_mae = sum(frozen_errors) / len(frozen_errors)
    learned_mae = sum(learned_errors) / len(learned_errors)
    return WorldModelComparison(
        training_samples=training_samples,
        heldout_samples=heldout_samples,
        frozen_model_mae=round(frozen_mae, 8),
        learned_model_mae=round(learned_mae, 8),
        relative_improvement=round(
            (frozen_mae - learned_mae) / frozen_mae if frozen_mae else 0.0,
            8,
        ),
        lexical_novelty_accuracy=lexical_accuracy,
        lexical_novelty_precision=lexical_precision,
        lexical_novelty_recall=lexical_recall,
        recovery=RecoveryEvaluation(
            first_error=round(first_recovery_error, 8),
            final_error=round(final_recovery_error, 8),
            uncertainty_before_shift=round(uncertainty_before_shift, 8),
            uncertainty_after_shift=round(uncertainty_after_shift, 8),
            uncertainty_after_recovery=round(uncertainty_after_recovery, 8),
        ),
    )
