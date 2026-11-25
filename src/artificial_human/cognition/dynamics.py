from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from math import sin, tanh
from threading import RLock
from typing import Final

from artificial_human.development.prenatal import POPULATIONS, PROJECTIONS


DYNAMICS_VERSION: Final[str] = "recurrent-population-dynamics-v1"
DEFAULT_CYCLES_PER_INPUT: Final[int] = 8
ACTIVITY_LEAK: Final[float] = 0.72
ACTIVITY_UPDATE: Final[float] = 0.28


@dataclass(frozen=True, slots=True)
class PopulationActivity:
    identifier: str
    activity: float
    external_drive: float


@dataclass(frozen=True, slots=True)
class DynamicsSnapshot:
    version: str
    inputs_processed: int
    integration_steps: int
    cycles_per_input: int
    population_count: int
    projection_count: int
    mean_activity: float
    active_population_count: int
    plasticity_enabled: bool
    intrinsic_activity_model: str
    idle_ticks: int
    state_storage: str
    populations: tuple[PopulationActivity, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "populations": [asdict(item) for item in self.populations],
        }


class RecurrentDynamicsEngine:
    """Small in-process recurrent population model for runtime cognitive activity.

    This is a computational experiment over abstract population and projection
    definitions. It is not a biological neural simulation. Text is converted to
    bounded channel drives; recurrent updates then propagate activity over a
    fixed directed graph. Raw input text is never retained by this engine.
    """

    _ASSOCIATION_CUES: Final[frozenset[str]] = frozenset({
        "associate", "association", "because", "compare", "connect",
        "connection", "concept", "explain", "how", "meaning", "relate",
        "relationship", "why",
    })
    _SEQUENCE_CUES: Final[frozenset[str]] = frozenset({
        "after", "before", "first", "next", "sequence", "step", "steps",
        "then", "timeline", "when",
    })
    _ACTION_CUES: Final[frozenset[str]] = frozenset({
        "build", "change", "create", "do", "execute", "implement", "make",
        "plan", "solve", "should", "task", "work",
    })
    _REGULATION_CUES: Final[frozenset[str]] = frozenset({
        "error", "overload", "risk", "safe", "safety", "slow", "stop",
        "unsafe", "urgent",
    })
    _INHIBITORY_CUES: Final[frozenset[str]] = frozenset({
        "avoid", "block", "cannot", "don't", "never", "no", "not",
        "prevent", "stop", "without",
    })

    def __init__(self, *, cycles_per_input: int = DEFAULT_CYCLES_PER_INPUT) -> None:
        if (
            isinstance(cycles_per_input, bool)
            or not isinstance(cycles_per_input, int)
            or not 1 <= cycles_per_input <= 32
        ):
            raise ValueError("cycles_per_input must be an integer between 1 and 32")
        self.cycles_per_input = cycles_per_input
        self._population_ids = tuple(item.identifier for item in POPULATIONS)
        self._activity = {identifier: 0.0 for identifier in self._population_ids}
        self._projection_edges = tuple(
            (
                item.source,
                item.target,
                float(item.initial_strength),
                -1.0 if item.polarity == "inhibitory" else 1.0,
            )
            for item in PROJECTIONS
        )
        self._inputs_processed = 0
        self._integration_steps = 0
        self._idle_ticks = 0
        self._last_drive = {identifier: 0.0 for identifier in self._population_ids}
        self._lock = RLock()

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9']+", text.casefold()))

    def _drive(self, text: str) -> dict[str, float]:
        tokens = self._tokens(text)
        token_count = len(tokens)
        if not tokens:
            raise ValueError("A non-empty text input is required")

        drive = {identifier: 0.0 for identifier in self._population_ids}
        # These are transparent text-feature encoders, not biological sensory codes.
        drive["sensory_input"] = min(0.70, 0.18 + 0.025 * token_count)
        drive["recurrent_integration"] = min(0.24, 0.035 + 0.008 * token_count)
        drive["association"] = 0.16 if tokens & self._ASSOCIATION_CUES else 0.035
        drive["sequence_context"] = 0.15 if tokens & self._SEQUENCE_CUES else 0.025
        drive["action_prediction"] = 0.14 if tokens & self._ACTION_CUES else 0.02
        drive["regulation"] = 0.12 if tokens & self._REGULATION_CUES else 0.025
        drive["inhibitory_control"] = (
            0.15 if tokens & self._INHIBITORY_CUES else 0.035
        )
        return drive

    def _integrate(self, drive: dict[str, float], *, steps: int) -> None:
        """Run synchronous bounded updates; caller holds the state lock."""
        for _ in range(steps):
            previous = self._activity.copy()
            incoming = {identifier: 0.0 for identifier in self._population_ids}
            for source, target, strength, polarity in self._projection_edges:
                incoming[target] += previous[source] * strength * polarity

            next_activity: dict[str, float] = {}
            for identifier in self._population_ids:
                net_input = drive[identifier] + incoming[identifier]
                target_activity = max(0.0, tanh(1.8 * net_input))
                updated = (
                    ACTIVITY_LEAK * previous[identifier]
                    + ACTIVITY_UPDATE * target_activity
                )
                next_activity[identifier] = min(1.0, max(0.0, updated))
            self._activity = next_activity
            self._integration_steps += 1

    def process(self, text: str) -> dict[str, object]:
        """Propagate one explicit text input through bounded recurrent updates."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("A non-empty text input is required")
        drive = self._drive(text)
        with self._lock:
            self._last_drive = drive
            self._integrate(drive, steps=self.cycles_per_input)
            self._inputs_processed += 1
            return self.snapshot().as_dict()

    def tick_idle(self) -> dict[str, object]:
        """Advance a deterministic low-amplitude resting-state approximation.

        This is an engineering activity loop, not biological spontaneous firing.
        It runs only while the hosting process is awake and stores no user input.
        """
        with self._lock:
            self._idle_ticks += 1
            drive = {
                identifier: 0.0025 + 0.0025 * (
                    0.5 + 0.5 * sin(self._idle_ticks * 0.11 + index * 0.73)
                )
                for index, identifier in enumerate(self._population_ids)
            }
            self._last_drive = drive
            self._integrate(drive, steps=1)
            return self.snapshot().as_dict()

    def snapshot(self) -> DynamicsSnapshot:
        with self._lock:
            populations = tuple(
                PopulationActivity(
                    identifier=identifier,
                    activity=round(self._activity[identifier], 6),
                    external_drive=round(self._last_drive[identifier], 6),
                )
                for identifier in self._population_ids
            )
            values = [item.activity for item in populations]
            return DynamicsSnapshot(
                version=DYNAMICS_VERSION,
                inputs_processed=self._inputs_processed,
                integration_steps=self._integration_steps,
                cycles_per_input=self.cycles_per_input,
                population_count=len(self._population_ids),
                projection_count=len(self._projection_edges),
                mean_activity=round(sum(values) / len(values), 6),
                active_population_count=sum(value >= 0.01 for value in values),
                plasticity_enabled=False,
                intrinsic_activity_model="deterministic_low_amplitude_approximation",
                idle_ticks=self._idle_ticks,
                state_storage="volatile_process_memory",
                populations=populations,
            )

    def reset(self) -> None:
        """Clear only transient activity, retaining the fixed source-code topology."""
        with self._lock:
            self._activity = {identifier: 0.0 for identifier in self._population_ids}
            self._last_drive = {identifier: 0.0 for identifier in self._population_ids}
            self._inputs_processed = 0
            self._integration_steps = 0
            self._idle_ticks = 0
