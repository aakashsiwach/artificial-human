"""Bounded opponent-pathway action selection for deterministic simulations.

This is an inspectable computational hypothesis inspired by cortico-striatal
action-selection research. It is not a basal ganglia anatomy or dopamine model.
The module only selects identifiers from a fixed simulation-only allowlist; it
contains no actuator or real-world execution interface.
"""
from __future__ import annotations

import math
import random
from types import MappingProxyType
from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """One explicitly permitted simulated action."""

    identifier: str
    description: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.identifier, str)
            or not self.identifier
            or self.identifier.strip() != self.identifier
        ):
            raise ValueError("action identifier must be a non-empty trimmed string")
        if len(self.identifier) > 64:
            raise ValueError("action identifier must be at most 64 characters")
        if not self.description or not self.description.strip():
            raise ValueError("action description is required")


class SafeActionSpace:
    """Immutable allowlist. Learned values can never add or enable an action."""

    MAX_ACTIONS = 64

    def __init__(self, actions: Sequence[ActionSpec]) -> None:
        items = tuple(actions)
        if any(not isinstance(item, ActionSpec) for item in items):
            raise ValueError("action space entries must be ActionSpec instances")
        if not items or len(items) > self.MAX_ACTIONS:
            raise ValueError("action space must contain 1..64 actions")
        identifiers = [item.identifier for item in items]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("action identifiers must be unique")
        self._actions = items
        self._by_id = MappingProxyType({item.identifier: item for item in items})

    @property
    def actions(self) -> tuple[ActionSpec, ...]:
        return self._actions

    @property
    def identifiers(self) -> tuple[str, ...]:
        return tuple(item.identifier for item in self._actions)

    def require(self, identifier: str) -> ActionSpec:
        try:
            return self._by_id[identifier]
        except KeyError as exc:
            raise ValueError(f"action is not in the approved simulation allowlist: {identifier}") from exc


@dataclass(frozen=True, slots=True)
class ActionChoice:
    state_id: str
    action_id: str
    direct_activation: float
    indirect_suppression: float
    net_value: float
    uncertainty: float
    explored: bool


@dataclass(frozen=True, slots=True)
class ActionLearningUpdate:
    state_id: str
    action_id: str
    reward: float
    prediction_error: float
    value_before: float
    value_after: float
    direct_change: float
    indirect_change: float
    visit_count: int


@dataclass(frozen=True, slots=True)
class ActionSelectionEvaluation:
    training_episodes: int
    heldout_episodes: int
    learned_accuracy: float
    random_accuracy: float
    fixed_policy_accuracy: float
    learned_mean_reward: float
    seed: int


class ActionSelectionCircuit:
    """Tabular actor-value baseline with opponent direct/indirect channels.

    Positive reward-prediction errors strengthen the direct channel; negative
    errors strengthen the indirect channel. Action values are their difference.
    This is an intentionally reduced, falsifiable hypothesis, not a full
    cortico-basal-ganglia-thalamo-cortical loop.
    """

    VERSION = "opponent-pathway-action-selection-v1"
    MAX_CHANNEL_WEIGHT = 10.0
    MAX_STATE_ID_LENGTH = 128

    def __init__(
        self,
        action_space: SafeActionSpace,
        *,
        learning_rate: float = 0.15,
        discount_factor: float = 0.0,
        exploration_rate: float = 0.15,
        exploration_bonus: float = 0.2,
        max_state_action_pairs: int = 10_000,
        seed: int = 0,
    ) -> None:
        for name, value, lower, upper, upper_inclusive in (
            ("learning_rate", learning_rate, 0.001, 1.0, True),
            ("discount_factor", discount_factor, 0.0, 1.0, True),
            ("exploration_rate", exploration_rate, 0.0, 1.0, True),
            ("exploration_bonus", exploration_bonus, 0.0, 2.0, True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < lower
                or (value > upper if upper_inclusive else value >= upper)
            ):
                raise ValueError(f"{name} must be in [{lower}, {upper}]")
        if (
            isinstance(max_state_action_pairs, bool)
            or not isinstance(max_state_action_pairs, int)
            or not 1 <= max_state_action_pairs <= 1_000_000
        ):
            raise ValueError("max_state_action_pairs must be in [1, 1000000]")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")

        self.action_space = action_space
        self.learning_rate = float(learning_rate)
        self.discount_factor = float(discount_factor)
        self.exploration_rate = float(exploration_rate)
        self.exploration_bonus = float(exploration_bonus)
        self.max_state_action_pairs = max_state_action_pairs
        self.seed = seed
        self._rng = random.Random(seed)
        self._direct: dict[tuple[str, str], float] = {}
        self._indirect: dict[tuple[str, str], float] = {}
        self._visits: dict[tuple[str, str], int] = {}
        self.updates = 0

    @classmethod
    def _validate_state_id(cls, state_id: str) -> None:
        if (
            not isinstance(state_id, str)
            or not state_id
            or state_id.strip() != state_id
            or len(state_id) > cls.MAX_STATE_ID_LENGTH
        ):
            raise ValueError("state_id must be a trimmed non-empty string of at most 128 chars")

    def _value(self, key: tuple[str, str]) -> float:
        return self._direct.get(key, 0.0) - self._indirect.get(key, 0.0)

    def select(
        self,
        state_id: str,
        available_action_ids: Sequence[str] | None = None,
        *,
        explore: bool = True,
    ) -> ActionChoice:
        """Choose only from the allowlist; exploration is seeded and reproducible."""
        self._validate_state_id(state_id)
        if not isinstance(explore, bool):
            raise ValueError("explore must be a boolean")
        candidates = (
            self.action_space.identifiers
            if available_action_ids is None
            else tuple(available_action_ids)
        )
        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError("available actions must be a non-empty unique sequence")
        for action_id in candidates:
            self.action_space.require(action_id)

        explored = explore and self._rng.random() < self.exploration_rate
        if explored:
            selected_id = self._rng.choice(candidates)
        else:
            total_visits = sum(
                count for (known_state, _), count in self._visits.items()
                if known_state == state_id
            )

            def score(action_id: str) -> tuple[float, int]:
                key = (state_id, action_id)
                visits = self._visits.get(key, 0)
                bonus = (
                    self.exploration_bonus
                    * math.sqrt(math.log(total_visits + 2.0) / (visits + 1.0))
                    if explore else 0.0
                )
                return self._value(key) + bonus, -candidates.index(action_id)
            selected_id = max(candidates, key=score)

        key = (state_id, selected_id)
        visits = self._visits.get(key, 0)
        return ActionChoice(
            state_id=state_id,
            action_id=selected_id,
            direct_activation=round(self._direct.get(key, 0.0), 8),
            indirect_suppression=round(self._indirect.get(key, 0.0), 8),
            net_value=round(self._value(key), 8),
            uncertainty=round(1.0 / math.sqrt(visits + 1.0), 8),
            explored=explored,
        )

    def observe(
        self,
        state_id: str,
        action_id: str,
        reward: float,
        *,
        next_state_id: str | None = None,
        terminal: bool = False,
    ) -> ActionLearningUpdate:
        """Apply a bounded signed prediction-error update to one pathway channel."""
        self._validate_state_id(state_id)
        self.action_space.require(action_id)
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(reward)
            or not -1.0 <= reward <= 1.0
        ):
            raise ValueError("reward must be finite and in [-1, 1]")
        if not isinstance(terminal, bool):
            raise ValueError("terminal must be a boolean")
        if next_state_id is not None:
            self._validate_state_id(next_state_id)
        if not terminal and next_state_id is None:
            raise ValueError("next_state_id is required for a non-terminal transition")

        key = (state_id, action_id)
        if key not in self._visits and len(self._visits) >= self.max_state_action_pairs:
            raise RuntimeError("state-action capacity reached")
        value_before = self._value(key)
        next_value = 0.0
        if not terminal and next_state_id is not None:
            next_value = max(
                (self._value((next_state_id, candidate)) for candidate in self.action_space.identifiers),
                default=0.0,
            )
        prediction_error = float(reward) + (
            0.0 if terminal else self.discount_factor * next_value
        ) - value_before

        direct_change = 0.0
        indirect_change = 0.0
        if prediction_error > 0:
            old = self._direct.get(key, 0.0)
            new = min(
                self.MAX_CHANNEL_WEIGHT,
                old + self.learning_rate * prediction_error,
            )
            self._direct[key] = new
            direct_change = new - old
        elif prediction_error < 0:
            old = self._indirect.get(key, 0.0)
            new = min(
                self.MAX_CHANNEL_WEIGHT,
                old + self.learning_rate * abs(prediction_error),
            )
            self._indirect[key] = new
            indirect_change = new - old

        self._visits[key] = self._visits.get(key, 0) + 1
        self.updates += 1
        return ActionLearningUpdate(
            state_id=state_id,
            action_id=action_id,
            reward=float(reward),
            prediction_error=round(prediction_error, 8),
            value_before=round(value_before, 8),
            value_after=round(self._value(key), 8),
            direct_change=round(direct_change, 8),
            indirect_change=round(indirect_change, 8),
            visit_count=self._visits[key],
        )

    def snapshot(self) -> dict[str, object]:
        """Return inspectable numeric state; no persistence or action execution."""
        keys = sorted(self._visits)
        return {
            "version": self.VERSION,
            "state_action_pairs": len(keys),
            "updates": self.updates,
            "channels": [
                {
                    "state_id": state,
                    "action_id": action,
                    "direct_activation": round(self._direct.get((state, action), 0.0), 8),
                    "indirect_suppression": round(self._indirect.get((state, action), 0.0), 8),
                    "net_value": round(self._value((state, action)), 8),
                    "visits": self._visits[(state, action)],
                }
                for state, action in keys
            ],
        }

    def reset(self) -> None:
        """Reset learned in-memory state and the seeded exploration stream."""
        self._direct.clear()
        self._indirect.clear()
        self._visits.clear()
        self.updates = 0
        self._rng = random.Random(self.seed)


def run_action_selection_comparison(
    *,
    training_episodes: int = 4_000,
    heldout_episodes: int = 400,
    seed: int = 23,
) -> ActionSelectionEvaluation:
    """Compare learned contextual choice with random and fixed-policy controls."""
    if (
        isinstance(training_episodes, bool)
        or not isinstance(training_episodes, int)
        or not 400 <= training_episodes <= 100_000
    ):
        raise ValueError("training_episodes must be in [400, 100000]")
    if (
        isinstance(heldout_episodes, bool)
        or not isinstance(heldout_episodes, int)
        or not 40 <= heldout_episodes <= 20_000
    ):
        raise ValueError("heldout_episodes must be in [40, 20000]")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    actions = SafeActionSpace(
        (
            ActionSpec("advance", "Move toward the target in the simulation"),
            ActionSpec("withdraw", "Move away from the target in the simulation"),
            ActionSpec("observe", "Gather another simulated observation"),
            ActionSpec("stabilize", "Hold position and reduce simulated motion"),
        )
    )
    correct = {
        "approach": "advance",
        "hazard": "withdraw",
        "uncertain": "observe",
        "unstable": "stabilize",
    }
    agent = ActionSelectionCircuit(actions, seed=seed)
    training_rng = random.Random(seed + 1)
    states = tuple(correct)
    for episode in range(training_episodes):
        state_id = states[episode % len(states)]
        # Deterministic context order plus seeded exploration; every context is trained.
        choice = agent.select(state_id, explore=True)
        reward = 1.0 if choice.action_id == correct[state_id] else -1.0
        agent.observe(state_id, choice.action_id, reward, terminal=True)

    learned_correct = 0
    random_correct = 0
    fixed_correct = 0
    reward_total = 0.0
    for episode in range(heldout_episodes):
        state_id = states[episode % len(states)]
        choice = agent.select(state_id, explore=False)
        is_correct = choice.action_id == correct[state_id]
        learned_correct += int(is_correct)
        reward_total += 1.0 if is_correct else -1.0
        random_correct += int(training_rng.choice(actions.identifiers) == correct[state_id])
        fixed_correct += int(actions.identifiers[0] == correct[state_id])

    return ActionSelectionEvaluation(
        training_episodes=training_episodes,
        heldout_episodes=heldout_episodes,
        learned_accuracy=round(learned_correct / heldout_episodes, 8),
        random_accuracy=round(random_correct / heldout_episodes, 8),
        fixed_policy_accuracy=round(fixed_correct / heldout_episodes, 8),
        learned_mean_reward=round(reward_total / heldout_episodes, 8),
        seed=seed,
    )
