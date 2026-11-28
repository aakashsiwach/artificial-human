"""Deterministic, bounded, in-memory research organism vertical slice.

All inputs are explicit synthetic events. No database, LLM, device, network,
filesystem mutation, or physical actuator is used by this module.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from artificial_human.neural.population import (
    PopulationCode,
    PopulationEncoder as DistributedPopulationEncoder,
)
from artificial_human.neural.action_selection import (
    ActionSelectionCircuit,
    ActionSpec,
    SafeActionSpace,
)
from artificial_human.cognition.attention import (
    AttentionCandidate,
    AttentionSystem,
)
from artificial_human.cognition.working_memory import (
    WorkingMemory,
    WorkingMemoryEntry,
)
from artificial_human.development.program import (
    DevelopmentalModality as NeuralModality,
    DevelopmentalProgram as LIFDevelopmentalProgram,
)


def _default_neural_development() -> LIFDevelopmentalProgram:
    """Start the synthetic runtime from a documented postnatal test fixture."""
    program = LIFDevelopmentalProgram()
    program.advance_prenatal(280)
    program.advance_postnatal(500)
    return program


class Modality(StrEnum):
    VISION = "vision"
    AUDITION = "audition"
    TOUCH = "touch"
    PROPRIOCEPTION = "proprioception"
    INTEROCEPTION = "interoception"


@dataclass(frozen=True, slots=True)
class SensoryEvent:
    modality: Modality
    values: tuple[float, ...]
    timestamp: int
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.timestamp < 0 or not self.values or len(self.values) > 64:
            raise ValueError("invalid sensory event bounds")
        if not 0 <= self.confidence <= 1 or not all(map(math.isfinite, self.values)):
            raise ValueError("sensory values and confidence must be finite and bounded")


@dataclass(slots=True)
class PlasticSynapse:
    weight: float = 0.5
    minimum: float = 0.0
    maximum: float = 1.0
    potentiation: float = 0.02
    depression: float = 0.025
    decay: float = 0.9
    pre_trace: float = 0.0
    post_trace: float = 0.0

    def tick(self, pre_spike: bool, post_spike: bool, gate: float = 1.0) -> float:
        if not 0 <= gate <= 1 or not self.minimum <= self.weight <= self.maximum:
            raise ValueError("invalid synapse state or gate")
        self.pre_trace *= self.decay
        self.post_trace *= self.decay
        delta = 0.0
        if pre_spike:
            delta += self.potentiation * self.post_trace * gate
            self.pre_trace += 1.0
        if post_spike:
            delta -= self.depression * self.pre_trace * gate
            self.post_trace += 1.0
        self.weight = min(self.maximum, max(self.minimum, self.weight + delta))
        return self.weight


@dataclass(slots=True)
class Homeostat:
    target_rate: float = 0.2
    threshold: float = 0.5
    rate: float = 0.0

    def update(self, observed_rate: float) -> float:
        if not 0 <= observed_rate <= 1:
            raise ValueError("rate must be in [0, 1]")
        self.rate = 0.8 * self.rate + 0.2 * observed_rate
        self.threshold = min(2.0, max(0.1, self.threshold + 0.01 * (self.rate - self.target_rate)))
        return self.threshold


class PopulationEncoder:
    """OrganismLoop adapter around the explicit distributed numeric encoder."""

    def __init__(self, units: int = 16) -> None:
        if isinstance(units, bool) or not isinstance(units, int) or not 2 <= units <= 128:
            raise ValueError("units must be an integer in [2, 128]")
        self.units = units
        self._encoder = DistributedPopulationEncoder(units_per_feature=units)

    def encode(self, event: SensoryEvent) -> PopulationCode:
        return self._encoder.encode(
            event.values, timestamp=event.timestamp, confidence=event.confidence
        )

    def decode(self, code: PopulationCode) -> tuple[float | None, ...]:
        return self._encoder.decode(code)


@dataclass(slots=True)
class EpisodicMemory:
    capacity: int = 128
    episodes: deque[tuple[int, tuple[float, ...], str]] = field(default_factory=deque)

    def store(self, tick: int, vector: tuple[float, ...], outcome: str) -> None:
        if tick < 0 or not outcome:
            raise ValueError("invalid episode")
        if len(self.episodes) >= self.capacity:
            self.episodes.popleft()
        self.episodes.append((tick, vector, outcome))

    def recall(self, cue: tuple[float, ...]) -> str | None:
        if not cue or not self.episodes:
            return None
        def similarity(item: tuple[int, tuple[float, ...], str]) -> float:
            other = item[1]
            n = min(len(cue), len(other))
            dot = sum(a * b for a, b in zip(cue[:n], other[:n], strict=True))
            norm = math.sqrt(sum(a*a for a in cue[:n]) * sum(b*b for b in other[:n]))
            return dot / norm if norm else 0.0
        return max(self.episodes, key=similarity)[2]


@dataclass(slots=True)
class WorldModel:
    weights: list[float]
    learning_rate: float = 0.05

    @classmethod
    def create(cls, dimensions: int = 4) -> WorldModel:
        if not 1 <= dimensions <= 64:
            raise ValueError("dimensions must be in [1, 64]")
        return cls([0.0] * dimensions)

    def predict(self, state: tuple[float, ...]) -> float:
        if len(state) != len(self.weights) or not all(map(math.isfinite, state)):
            raise ValueError("invalid world state")
        return sum(w * x for w, x in zip(self.weights, state, strict=True))

    def observe(self, state: tuple[float, ...], outcome: float) -> float:
        error = outcome - self.predict(state)
        for i, value in enumerate(state):
            self.weights[i] = min(2.0, max(-2.0, self.weights[i] + self.learning_rate * error * value))
        return error


@dataclass(slots=True)
class SimulatedEnvironment:
    position: float = 0.0
    target: float = 1.0
    step_size: float = 0.1
    steps: int = 0
    max_steps: int = 100

    def act(self, action: int) -> float:
        if action not in (-1, 0, 1):
            raise ValueError("action outside simulated safe action space")
        if self.steps >= self.max_steps:
            raise RuntimeError("episode step limit reached")
        self.position = min(2.0, max(-2.0, self.position + action * self.step_size))
        self.steps += 1
        return -abs(self.target - self.position)

    def reset(self) -> None:
        self.position, self.steps = 0.0, 0


@dataclass(slots=True)
class ActionLearner:
    """Legacy directional learner retained as a comparison baseline only."""

    values: dict[int, float] = field(default_factory=lambda: {-1: 0.0, 0: 0.0, 1: 0.0})

    def choose(self, direction: float) -> int:
        return 1 if direction > 0 else -1 if direction < 0 else 0

    def learn(self, action: int, reward: float) -> None:
        if action not in self.values or not math.isfinite(reward):
            raise ValueError("invalid action/reward")
        self.values[action] += 0.2 * (reward - self.values[action])


@dataclass(slots=True)
class ForwardController:
    gain: float = 0.1

    def predict(self, action: int) -> float:
        return self.gain * action

    def adapt(self, action: int, observed_delta: float) -> float:
        error = observed_delta - self.predict(action)
        self.gain = min(1.0, max(0.0, self.gain + 0.1 * error * action))
        return error


@dataclass(frozen=True, slots=True)
class OrganismIntegrationConfig:
    """Explicit ablation switches for the deterministic integrated loop."""

    attention_enabled: bool = True
    working_memory_enabled: bool = True
    episodic_memory_enabled: bool = True
    neural_substrate_enabled: bool = True
    learned_action_selection_enabled: bool = True

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, bool)
            for value in (
                self.attention_enabled,
                self.working_memory_enabled,
                self.episodic_memory_enabled,
                self.neural_substrate_enabled,
                self.learned_action_selection_enabled,
            )
        ):
            raise ValueError("integration ablation switches must be booleans")


def _default_action_selection() -> ActionSelectionCircuit:
    """Build a fixed simulation-only action allowlist and seeded policy."""
    space = SafeActionSpace((
        ActionSpec("move_left", "Move the simulated body one step left"),
        ActionSpec("hold", "Keep the simulated body stationary"),
        ActionSpec("move_right", "Move the simulated body one step right"),
    ))
    return ActionSelectionCircuit(
        space,
        learning_rate=0.2,
        discount_factor=0.1,
        exploration_rate=0.1,
        exploration_bonus=0.1,
        max_state_action_pairs=256,
        seed=7,
    )


class DevelopmentStage(StrEnum):
    EARLY = "early"
    JUVENILE = "juvenile"
    MATURE = "mature"


@dataclass(slots=True)
class DevelopmentProgram:
    """Legacy stage-only baseline retained for compatibility tests."""

    tick: int = 0
    stage: DevelopmentStage = DevelopmentStage.EARLY

    def advance(self) -> DevelopmentStage:
        self.tick += 1
        if self.tick >= 300:
            self.stage = DevelopmentStage.MATURE
        elif self.tick >= 100:
            self.stage = DevelopmentStage.JUVENILE
        return self.stage


@dataclass(frozen=True, slots=True)
class ImprovementProposal:
    identifier: str
    hypothesis: str
    diff: str
    checks: tuple[str, ...]
    status: str = "awaiting_human_review"


class SafeSelfDevelopment:
    _secret = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")

    def propose(self, hypothesis: str, diff: str, checks: tuple[str, ...]) -> ImprovementProposal:
        if not hypothesis.strip() or not diff.strip() or len(diff) > 50_000:
            raise ValueError("invalid or oversized proposal")
        if self._secret.search(diff) or re.search(r"(?m)^\+\s*(?:eval\(|exec\(|os\.system|subprocess\.)", diff):
            raise ValueError("proposal contains a secret or prohibited execution primitive")
        allowed = {"ruff check .", "mypy src", "pytest -q"}
        if not checks or len(checks) > 20 or any(c not in allowed for c in checks):
            raise ValueError("checks must be non-empty and allowlisted")
        digest = hashlib.sha256((hypothesis + diff).encode()).hexdigest()[:16]
        return ImprovementProposal(digest, hypothesis, diff, checks)


@dataclass(slots=True)
class OrganismLoop:
    encoder: PopulationEncoder = field(default_factory=PopulationEncoder)
    memory: EpisodicMemory = field(default_factory=EpisodicMemory)
    world: WorldModel = field(default_factory=WorldModel.create)
    environment: SimulatedEnvironment = field(default_factory=SimulatedEnvironment)
    controller: ForwardController = field(default_factory=ForwardController)
    homeostat: Homeostat = field(default_factory=Homeostat)
    neural_development: LIFDevelopmentalProgram = field(default_factory=_default_neural_development)
    attention: AttentionSystem = field(default_factory=AttentionSystem)
    working_memory: WorkingMemory = field(default_factory=lambda: WorkingMemory(capacity=7))
    action_selection: ActionSelectionCircuit = field(default_factory=_default_action_selection)
    config: OrganismIntegrationConfig = field(default_factory=OrganismIntegrationConfig)
    tick_index: int = 0
    episode_count: int = 0
    paused: bool = False

    @staticmethod
    def _state_id(distance: float) -> str:
        direction = "ahead" if distance > 0.05 else "behind" if distance < -0.05 else "aligned"
        bucket = min(30, int(abs(distance) * 10))
        return f"goal:{direction}:{bucket}"

    def tick(self, event: SensoryEvent | None = None) -> dict[str, object]:
        if self.paused:
            return {"tick": self.tick_index, "paused": True}
        if self.environment.steps >= self.environment.max_steps:
            self.environment.reset()
            self.episode_count += 1
        event = event or SensoryEvent(
            Modality.PROPRIOCEPTION,
            (self.environment.position, self.environment.target),
            self.tick_index,
        )
        code = self.encoder.encode(event)
        decoded = self.encoder.decode(code)
        cue = tuple(value for value in decoded if value is not None)
        recalled = (
            self.memory.recall(cue)
            if self.config.episodic_memory_enabled
            else None
        )

        candidates = [
            AttentionCandidate(
                reference_id=f"sensory:{self.tick_index}",
                kind="sensory",
                relevance=event.confidence,
                salience=1.0 - code.uncertainty,
                channels=(event.modality.value,),
            )
        ]
        if recalled is not None:
            candidates.append(AttentionCandidate(
                reference_id=f"episode:{self.tick_index}",
                kind="episodic",
                relevance=0.7,
                salience=0.6,
                channels=("episodic_memory",),
            ))
        attended = (
            self.attention.select(candidates, limit=3)
            if self.config.attention_enabled
            else []
        )
        attended_weight = attended[0].weight if attended else (
            1.0 if not self.config.attention_enabled else 0.0
        )

        if self.config.working_memory_enabled:
            self.working_memory.remember(WorkingMemoryEntry(
                id=f"sensory:{self.tick_index}",
                content=",".join(f"{value:.6f}" for value in event.values),
                kind="percept",
                source="synthetic_sensory_event",
                salience=min(1.0, max(0.0, event.confidence * attended_weight)),
                created_at=f"simulation-tick:{self.tick_index:012d}",
            ))
            if recalled is not None:
                self.working_memory.remember(WorkingMemoryEntry(
                    id=f"recall:{self.tick_index}",
                    content=recalled,
                    kind="episodic_recall",
                    source="volatile_episodic_memory",
                    salience=0.6,
                    created_at=f"simulation-tick:{self.tick_index:012d}",
                ))

        neural_outcome: dict[str, object] = {
            "accepted": False,
            "spikes": (),
            "total_weight_change": 0.0,
            "simulation_step": self.neural_development.simulation.step_index,
        }
        if self.config.neural_substrate_enabled:
            modality_map = {
                Modality.VISION: NeuralModality.VISION,
                Modality.AUDITION: NeuralModality.AUDITORY,
                Modality.TOUCH: NeuralModality.TACTILE,
                Modality.PROPRIOCEPTION: NeuralModality.PROPRIOCEPTION,
                Modality.INTEROCEPTION: NeuralModality.INTEROCEPTION,
            }
            neural_intensity = min(
                1.0,
                max(0.0, event.confidence * (0.5 + 0.5 * attended_weight)),
            )
            outcome = self.neural_development.experience(
                modality_map[event.modality],
                intensity=neural_intensity,
                duration_steps=1,
            )
            self.neural_development.advance_simulation(1)
            neural_outcome = {
                "accepted": outcome.accepted,
                "spikes": outcome.spikes,
                "total_weight_change": outcome.total_weight_change,
                "simulation_step": self.neural_development.simulation.step_index,
            }

        distance = self.environment.target - self.environment.position
        state = (
            self.environment.position,
            self.environment.target,
            distance,
            attended_weight,
        )
        predicted = self.world.predict(state)
        state_id = self._state_id(distance)
        if self.config.learned_action_selection_enabled:
            choice = self.action_selection.select(state_id, explore=True)
            action_map = {"move_left": -1, "hold": 0, "move_right": 1}
            action = action_map[choice.action_id]
        else:
            choice = None
            action = 1 if distance > 0.05 else -1 if distance < -0.05 else 0

        before = self.environment.position
        distance_before = abs(self.environment.target - before)
        reward = self.environment.act(action)
        delta = self.environment.position - before
        distance_after = abs(self.environment.target - self.environment.position)
        reward_signal = max(
            -1.0,
            min(1.0, (distance_before - distance_after) / self.environment.step_size),
        )
        next_state_id = self._state_id(self.environment.target - self.environment.position)
        update = None
        if self.config.learned_action_selection_enabled and choice is not None:
            update = self.action_selection.observe(
                state_id,
                choice.action_id,
                reward_signal,
                next_state_id=next_state_id,
            )
        error = self.world.observe(state, reward)
        self.controller.adapt(action, delta)
        if self.config.episodic_memory_enabled:
            self.memory.store(self.tick_index, cue, f"action={action};reward={reward:.3f}")
        threshold = self.homeostat.update(min(1.0, abs(delta) * 2))
        stage = self.neural_development.stage
        result: dict[str, object] = {
            "tick": self.tick_index,
            "modality": event.modality.value,
            "code": code.values,
            "uncertainty": code.uncertainty,
            "attended": tuple(
                {
                    "reference_id": item.reference_id,
                    "kind": item.kind,
                    "weight": item.weight,
                    "channels": item.channels,
                }
                for item in attended
            ),
            "working_memory": self.working_memory.public_state(),
            "recalled": recalled,
            "prediction": predicted,
            "prediction_error": error,
            "action": action,
            "action_selection": (
                {
                    "state_id": choice.state_id,
                    "action_id": choice.action_id,
                    "net_value": choice.net_value,
                    "uncertainty": choice.uncertainty,
                    "explored": choice.explored,
                    "prediction_error": update.prediction_error if update else None,
                }
                if choice is not None else {"mode": "directional_ablation_baseline"}
            ),
            "reward": reward,
            "reward_signal": reward_signal,
            "position": self.environment.position,
            "threshold": threshold,
            "stage": stage.value,
            "neural": neural_outcome,
        }
        self.tick_index += 1
        return result

    def run(self, ticks: int) -> tuple[dict[str, object], ...]:
        if isinstance(ticks, bool) or not isinstance(ticks, int) or not 0 <= ticks <= 10_000:
            raise ValueError("ticks must be an integer in [0, 10000]")
        return tuple(self.tick() for _ in range(ticks))

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def reset(self) -> None:
        self.memory.episodes.clear()
        self.world.weights[:] = [0.0] * len(self.world.weights)
        self.environment.reset()
        self.working_memory = WorkingMemory(capacity=7)
        self.action_selection.reset()
        self.controller.gain = 0.1
        self.homeostat = Homeostat()
        self.neural_development = _default_neural_development()
        self.tick_index, self.episode_count, self.paused = 0, 0, False

    def snapshot(self) -> dict[str, object]:
        development = self.neural_development.snapshot()
        return {
            "tick": self.tick_index,
            "paused": self.paused,
            "position": self.environment.position,
            "episodes": len(self.memory.episodes),
            "episode_count": self.episode_count,
            "weights": tuple(self.world.weights),
            "working_memory": self.working_memory.public_state(),
            "action_selection": self.action_selection.snapshot(),
            "stage": development.stage.value,
            "developmental_time": {
                "prenatal_days": development.prenatal_days,
                "postnatal_units": development.postnatal_units,
            },
            "neural": {
                "simulation_step": development.simulation_step,
                "neuron_count": development.neuron_count,
                "synapse_count": development.synapse_count,
                "total_spikes": development.total_spikes,
                "experience_count": development.experience_count,
                "mean_conduction_delay_steps": development.mean_conduction_delay_steps,
            },
        }



def run_integration_ablation_comparison(
    *, ticks: int = 100
) -> tuple[dict[str, object], ...]:
    """Measure end-to-end subsystem ablations on identical deterministic tasks."""
    if isinstance(ticks, bool) or not isinstance(ticks, int) or not 1 <= ticks <= 1_000:
        raise ValueError("ticks must be an integer in [1, 1000]")

    conditions = (
        ("full", OrganismIntegrationConfig()),
        ("attention_disabled", OrganismIntegrationConfig(attention_enabled=False)),
        ("working_memory_disabled", OrganismIntegrationConfig(working_memory_enabled=False)),
        ("episodic_memory_disabled", OrganismIntegrationConfig(episodic_memory_enabled=False)),
        ("neural_substrate_disabled", OrganismIntegrationConfig(neural_substrate_enabled=False)),
        (
            "learned_action_selection_disabled",
            OrganismIntegrationConfig(learned_action_selection_enabled=False),
        ),
    )
    reports: list[dict[str, object]] = []
    for name, config in conditions:
        organism = OrganismLoop(config=config)
        trace = organism.run(ticks)
        snapshot = organism.snapshot()
        neural_spikes = 0
        attended_items = 0
        attention_weight_total = 0.0
        reward_total = 0.0
        prediction_error_total = 0.0
        for step in trace:
            attended = step.get("attended")
            if isinstance(attended, (tuple, list)):
                attended_items += len(attended)
                if attended and isinstance(attended[0], dict):
                    weight = attended[0].get("weight")
                    if isinstance(weight, (int, float)):
                        attention_weight_total += float(weight)
            neural = step.get("neural")
            if isinstance(neural, dict):
                spikes = neural.get("spikes")
                if isinstance(spikes, (tuple, list)):
                    neural_spikes += len(spikes)
            reward_value = step.get("reward")
            if isinstance(reward_value, (int, float)):
                reward_total += float(reward_value)
            error_value = step.get("prediction_error")
            if isinstance(error_value, (int, float)):
                prediction_error_total += abs(float(error_value))
        working_memory_state = snapshot.get("working_memory")
        working_memory_items = (
            working_memory_state.get("item_count", 0)
            if isinstance(working_memory_state, dict) else 0
        )
        action_state = snapshot.get("action_selection")
        action_updates = (
            action_state.get("updates", 0)
            if isinstance(action_state, dict) else 0
        )
        reports.append({
            "condition": name,
            "ticks": ticks,
            "cumulative_reward": round(reward_total, 8),
            "mean_absolute_prediction_error": round(prediction_error_total / ticks, 8),
            "neural_spikes": neural_spikes,
            "attended_items": attended_items,
            "mean_attention_weight": round(attention_weight_total / ticks, 8),
            "working_memory_items": working_memory_items,
            "episodic_memories": snapshot["episodes"],
            "action_learning_updates": action_updates,
        })
    return tuple(reports)
