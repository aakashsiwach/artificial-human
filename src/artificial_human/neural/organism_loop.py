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


@dataclass(frozen=True, slots=True)
class PopulationCode:
    values: tuple[float, ...]
    uncertainty: float


class PopulationEncoder:
    def __init__(self, units: int = 16) -> None:
        if not 2 <= units <= 128:
            raise ValueError("units must be in [2, 128]")
        self.units = units

    def encode(self, event: SensoryEvent) -> PopulationCode:
        vector = [0.0] * self.units
        for index, value in enumerate(event.values):
            vector[index % self.units] += math.tanh(value) / (1 + index // self.units)
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return PopulationCode(tuple(round(v / norm * event.confidence, 8) for v in vector),
                              round(1 - event.confidence, 8))


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
            dot = sum(a * b for a, b in zip(cue[:n], other[:n]))
            norm = math.sqrt(sum(a*a for a in cue[:n]) * sum(b*b for b in other[:n]))
            return dot / norm if norm else 0.0
        return max(self.episodes, key=similarity)[2]


@dataclass(slots=True)
class WorldModel:
    weights: list[float]
    learning_rate: float = 0.05

    @classmethod
    def create(cls, dimensions: int = 4) -> "WorldModel":
        if not 1 <= dimensions <= 64:
            raise ValueError("dimensions must be in [1, 64]")
        return cls([0.0] * dimensions)

    def predict(self, state: tuple[float, ...]) -> float:
        if len(state) != len(self.weights) or not all(map(math.isfinite, state)):
            raise ValueError("invalid world state")
        return sum(w * x for w, x in zip(self.weights, state))

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


class DevelopmentStage(StrEnum):
    EARLY = "early"
    JUVENILE = "juvenile"
    MATURE = "mature"


@dataclass(slots=True)
class DevelopmentProgram:
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
    actions: ActionLearner = field(default_factory=ActionLearner)
    controller: ForwardController = field(default_factory=ForwardController)
    homeostat: Homeostat = field(default_factory=Homeostat)
    development: DevelopmentProgram = field(default_factory=DevelopmentProgram)
    tick_index: int = 0
    paused: bool = False

    def tick(self, event: SensoryEvent | None = None) -> dict[str, object]:
        if self.paused:
            return {"tick": self.tick_index, "paused": True}
        event = event or SensoryEvent(Modality.PROPRIOCEPTION,
                                      (self.environment.position, self.environment.target),
                                      self.tick_index)
        code = self.encoder.encode(event)
        cue = code.values[:len(self.world.weights)]
        recalled = self.memory.recall(cue)
        state = (self.environment.position, self.environment.target,
                 self.environment.target - self.environment.position, 1.0)
        predicted = self.world.predict(state)
        action = self.actions.choose(self.environment.target - self.environment.position)
        before = self.environment.position
        reward = self.environment.act(action)
        delta = self.environment.position - before
        error = self.world.observe(state, reward)
        self.controller.adapt(action, delta)
        self.actions.learn(action, reward)
        self.memory.store(self.tick_index, code.values, f"action={action};reward={reward:.3f}")
        threshold = self.homeostat.update(min(1.0, abs(delta) * 2))
        stage = self.development.advance()
        result = {"tick": self.tick_index, "modality": event.modality.value, "code": code.values,
                  "uncertainty": code.uncertainty, "recalled": recalled, "prediction": predicted,
                  "prediction_error": error, "action": action, "reward": reward,
                  "position": self.environment.position, "threshold": threshold, "stage": stage.value}
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
        self.actions.values = {-1: 0.0, 0: 0.0, 1: 0.0}
        self.controller.gain = 0.1
        self.homeostat = Homeostat()
        self.development = DevelopmentProgram()
        self.tick_index, self.paused = 0, False

    def snapshot(self) -> dict[str, object]:
        return {"tick": self.tick_index, "paused": self.paused,
                "position": self.environment.position, "episodes": len(self.memory.episodes),
                "weights": tuple(self.world.weights), "stage": self.development.stage.value}
