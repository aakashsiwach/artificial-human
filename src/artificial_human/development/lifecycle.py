"""Unified in-memory developmental programme on the LIF neural substrate.

The prenatal scaffold is retained as a migration/compatibility input. After
migration, development, experience, synaptogenesis and functional pruning all
operate on one deterministic NeuralSimulation instance. This is a computational
experiment, not an individual human-development or fetal-brain simulator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from artificial_human.development.prenatal import (
    MODALITY_ONSET_WEEK,
    MODALITY_PATHS,
    POPULATIONS,
    PROJECTIONS,
    PrenatalBrain,
    PrenatalModality,
)
from artificial_human.neural.plasticity import PairBasedSTDP, STDPParameters
from artificial_human.neural.simulation import (
    LIFNeuron,
    LIFParameters,
    NeuralSimulation,
    Polarity,
    SimulationStep,
    Synapse,
)


PRENATAL_TERM_DAYS = 280
MAX_MODEL_AGE_DAYS = 36_500


class DevelopmentStage(StrEnum):
    """Coarse model epochs; these are not human-age or clinical categories."""

    PRENATAL = "prenatal"
    EARLY_POSTNATAL = "early_postnatal"
    CHILDHOOD = "childhood"
    ADOLESCENCE = "adolescence"
    MATURE = "mature_model_regime"


class DevelopmentalModality(StrEnum):
    TACTILE = "tactile"
    VESTIBULAR = "vestibular"
    AUDITORY = "auditory"
    VISION = "vision"
    PROPRIOCEPTION = "proprioception"
    INTEROCEPTION = "interoception"


POSTNATAL_PATHS: dict[DevelopmentalModality, tuple[str, ...]] = {
    DevelopmentalModality.VISION: (
        "sensory_input", "recurrent_integration", "action_prediction", "association",
    ),
    DevelopmentalModality.PROPRIOCEPTION: (
        "sensory_input", "recurrent_integration", "action_prediction", "association",
    ),
    DevelopmentalModality.INTEROCEPTION: (
        "regulation", "recurrent_integration", "association",
    ),
}

POSTNATAL_ONSET_DAYS: dict[DevelopmentalModality, int] = {
    DevelopmentalModality.VISION: PRENATAL_TERM_DAYS + 1,
    DevelopmentalModality.PROPRIOCEPTION: PRENATAL_TERM_DAYS + 1,
    DevelopmentalModality.INTEROCEPTION: 0,
}

STAGED_CONNECTIONS: dict[DevelopmentStage, tuple[tuple[str, str, Polarity, float], ...]] = {
    DevelopmentStage.EARLY_POSTNATAL: (
        ("sequence_context", "action_prediction", Polarity.EXCITATORY, 105.0),
        ("action_prediction", "sequence_context", Polarity.EXCITATORY, 85.0),
    ),
    DevelopmentStage.CHILDHOOD: (
        ("sensory_input", "inhibitory_control", Polarity.EXCITATORY, 90.0),
        ("inhibitory_control", "recurrent_integration", Polarity.INHIBITORY, 100.0),
    ),
    DevelopmentStage.ADOLESCENCE: (
        ("association", "regulation", Polarity.INHIBITORY, 80.0),
        ("regulation", "sequence_context", Polarity.EXCITATORY, 75.0),
    ),
    DevelopmentStage.PRENATAL: (),
    DevelopmentStage.MATURE: (),
}

STAGE_BOUNDARIES: tuple[tuple[int, DevelopmentStage], ...] = (
    (PRENATAL_TERM_DAYS, DevelopmentStage.PRENATAL),
    (730, DevelopmentStage.EARLY_POSTNATAL),
    (3_650, DevelopmentStage.CHILDHOOD),
    (6_570, DevelopmentStage.ADOLESCENCE),
)


@dataclass(frozen=True, slots=True)
class DevelopmentalExposure:
    modality: str
    available: bool
    reason: str
    developmental_gate: float
    synapses_updated: int
    mean_path_weight_before: float
    mean_path_weight_after: float


@dataclass(frozen=True, slots=True)
class DevelopmentalSnapshot:
    version: str
    seed: int
    developmental_day: int
    stage: str
    population_maturity: tuple[tuple[str, float], ...]
    accepted_exposures: int
    plasticity_updates: int
    staged_synapses_created: tuple[str, ...]
    functionally_pruned_synapses: tuple[str, ...]
    synapse_usage: tuple[tuple[str, int], ...]
    network: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "seed": self.seed,
            "developmental_day": self.developmental_day,
            "stage": self.stage,
            "population_maturity": dict(self.population_maturity),
            "accepted_exposures": self.accepted_exposures,
            "plasticity_updates": self.plasticity_updates,
            "staged_synapses_created": list(self.staged_synapses_created),
            "functionally_pruned_synapses": list(self.functionally_pruned_synapses),
            "synapse_usage": dict(self.synapse_usage),
            "network": self.network,
            "limitations": [
                "Developmental days and stage boundaries are engineering time, not a person's age.",
                "Population nodes are functional abstractions, not cell counts or anatomical replicas.",
                "Functional pruning silences unused model connections; it does not model biological synapse elimination.",
                "No myelin, glial, endocrine, clinical or consciousness model is implied.",
            ],
        }


def stage_for_day(day: int) -> DevelopmentStage:
    if isinstance(day, bool) or not isinstance(day, int) or day < 0:
        raise ValueError("developmental day must be a non-negative integer")
    if day <= PRENATAL_TERM_DAYS:
        return DevelopmentStage.PRENATAL
    if day <= 730:
        return DevelopmentStage.EARLY_POSTNATAL
    if day <= 3_650:
        return DevelopmentStage.CHILDHOOD
    if day <= 6_570:
        return DevelopmentStage.ADOLESCENCE
    return DevelopmentStage.MATURE


class DevelopmentalNeuralSubstrate:
    """One LIF network whose bounded state changes across model development.

    The seven source-defined functional populations and ten prenatal projections
    are instantiated directly in NeuralSimulation. Postnatal stages add a small
    set of predeclared connections. Local STDP-inspired updates are gated by
    population maturity; unused staged connections may be functionally silenced
    in the mature model regime.
    """

    VERSION = "developmental-neural-substrate-v1"

    def __init__(self, *, seed: int = 0) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.seed = seed
        self.developmental_day = 0
        self.stage = DevelopmentStage.PRENATAL
        self.simulation = NeuralSimulation(
            dt_ms=1.0,
            max_neurons=32,
            max_synapses=128,
            max_pending_events=4_096,
            max_abs_external_current=1_000.0,
        )
        parameters = LIFParameters(
            resting_potential_mV=-65.0,
            reset_potential_mV=-65.0,
            threshold_mV=-50.0,
            membrane_time_constant_ms=20.0,
            membrane_resistance=1.0,
            refractory_period_ms=2.0,
        )
        self._maturity = {item.identifier: 0.0 for item in POPULATIONS}
        for item in POPULATIONS:
            self.simulation.add_neuron(
                LIFNeuron(item.identifier, parameters, input_gain=0.15)
            )

        self._projection_ids: dict[tuple[str, str], str] = {}
        self._usage: dict[str, int] = {}
        self._staged_ids: list[str] = []
        self._pruned_ids: list[str] = []
        self._accepted_exposures = 0
        self._plasticity_updates = 0
        for index, projection in enumerate(PROJECTIONS):
            identifier = f"prenatal-{index}-{projection.source}-{projection.target}"
            polarity = (
                Polarity.INHIBITORY
                if projection.polarity == "inhibitory"
                else Polarity.EXCITATORY
            )
            self.simulation.add_synapse(
                Synapse(
                    identifier,
                    projection.source,
                    projection.target,
                    120.0 + 120.0 * projection.initial_strength,
                    polarity,
                    delay_steps=2,
                    time_constant_ms=5.0,
                    maximum_weight=250.0,
                )
            )
            self._projection_ids[(projection.source, projection.target)] = identifier
            self._usage[identifier] = 0

        self._stdp = PairBasedSTDP(
            STDPParameters(potentiation=0.025, depression=0.02, trace_decay=0.92)
        )
        self._refresh_maturity()

    @classmethod
    def from_prenatal(
        cls, prenatal: PrenatalBrain, *, seed: int | None = None
    ) -> DevelopmentalNeuralSubstrate:
        """Migrate a prenatal scaffold snapshot into the LIF substrate.

        Population maturity and all ten projection efficacies are copied. The
        original PrenatalBrain remains available as a compatibility baseline;
        subsequent updates occur only in this unified neural substrate.
        """
        if not isinstance(prenatal, PrenatalBrain):
            raise TypeError("prenatal must be a PrenatalBrain")
        snapshot = prenatal.snapshot()
        result = cls(seed=prenatal.seed if seed is None else seed)
        result.developmental_day = snapshot.age_days
        result.stage = stage_for_day(snapshot.age_days)
        result._refresh_maturity()
        for population in snapshot.populations:
            result._maturity[population.identifier] = population.maturity
            result.simulation.neurons[population.identifier].set_input_gain(
                0.15 + 0.85 * population.maturity
            )
        for projection in snapshot.projections:
            identifier = result._projection_ids[(projection.source, projection.target)]
            result.simulation.update_synapse_weight(
                identifier, 120.0 + 120.0 * projection.strength
            )
        return result

    def _population_maturity(self, identifier: str, day: int) -> float:
        definition = next(item for item in POPULATIONS if item.identifier == identifier)
        if day <= PRENATAL_TERM_DAYS:
            week = day / 7.0
            value = (
                (week - definition.maturation_start_week)
                / (definition.maturation_end_week - definition.maturation_start_week)
            )
            return round(min(1.0, max(0.0, value)), 6)
        term_week = 40.0
        term_value = min(
            1.0,
            max(
                0.0,
                (term_week - definition.maturation_start_week)
                / (definition.maturation_end_week - definition.maturation_start_week),
            ),
        )
        postnatal_days = max(
            365.0, (definition.maturation_end_week - term_week) * 120.0
        )
        progress = min(1.0, max(0.0, (day - PRENATAL_TERM_DAYS) / postnatal_days))
        return round(term_value + (1.0 - term_value) * progress, 6)

    def _refresh_maturity(self) -> None:
        for identifier in self._maturity:
            maturity = self._population_maturity(identifier, self.developmental_day)
            self._maturity[identifier] = maturity
            self.simulation.neurons[identifier].set_input_gain(0.15 + 0.85 * maturity)

    def _add_stage_connections(self, stage: DevelopmentStage) -> None:
        for index, (source, target, polarity, weight) in enumerate(STAGED_CONNECTIONS[stage]):
            identifier = f"development-{stage.value}-{index}-{source}-{target}"
            self.simulation.add_synapse(
                Synapse(
                    identifier,
                    source,
                    target,
                    weight,
                    polarity,
                    delay_steps=1 if stage is DevelopmentStage.ADOLESCENCE else 2,
                    time_constant_ms=5.0,
                    maximum_weight=250.0,
                )
            )
            self._staged_ids.append(identifier)
            self._usage[identifier] = 0

    def _plasticity_gate(self) -> float:
        mean_maturity = sum(self._maturity.values()) / len(self._maturity)
        stage_factor = {
            DevelopmentStage.PRENATAL: 0.25,
            DevelopmentStage.EARLY_POSTNATAL: 0.7,
            DevelopmentStage.CHILDHOOD: 0.55,
            DevelopmentStage.ADOLESCENCE: 0.35,
            DevelopmentStage.MATURE: 0.15,
        }[self.stage]
        return min(1.0, max(0.0, mean_maturity * stage_factor))

    def _step(
        self,
        external_currents: dict[str, float] | None = None,
        *,
        learn: bool = True,
        count_experience_usage: bool = False,
    ) -> SimulationStep:
        result = self.simulation.step(external_currents)
        if count_experience_usage:
            for spike in result.spikes:
                for identifier, synapse in self.simulation.synapses.items():
                    if synapse.source == spike:
                        self._usage[identifier] = self._usage.get(identifier, 0) + 1
        if learn:
            before = {
                identifier: synapse.weight
                for identifier, synapse in self.simulation.synapses.items()
            }
            self._stdp.observe(
                self.simulation, result.spikes, gate=self._plasticity_gate()
            )
            self._plasticity_updates += sum(
                self.simulation.synapses[identifier].weight != weight
                for identifier, weight in before.items()
            )
        return result

    def advance(self, days: int = 1) -> None:
        """Advance model time; every day updates the same LIF network in RAM."""
        if isinstance(days, bool) or not isinstance(days, int) or not 0 <= days <= 10_000:
            raise ValueError("days must be an integer in [0, 10000]")
        if self.developmental_day + days > MAX_MODEL_AGE_DAYS:
            raise ValueError("developmental age exceeds the configured model horizon")
        for _ in range(days):
            self.developmental_day += 1
            self._refresh_maturity()
            next_stage = stage_for_day(self.developmental_day)
            if next_stage is not self.stage:
                self.stage = next_stage
                self._add_stage_connections(next_stage)
                if next_stage is DevelopmentStage.MATURE:
                    self._prune_unused_staged_connections()
            currents: dict[str, float] = {}
            if self.developmental_day >= 56:
                for index, identifier in enumerate(sorted(self._maturity)):
                    maturity = self._maturity[identifier]
                    phase = (self.seed % 997) * 0.001 + index * 0.73
                    wave = 0.5 + 0.5 * math.sin(self.developmental_day * 0.11 + phase)
                    currents[identifier] = 180.0 * maturity * wave
            self._step(currents, learn=True)

    def _prune_unused_staged_connections(self) -> None:
        """Silence never-used staged edges; base connectivity is never pruned."""
        for identifier in self._staged_ids:
            if self._usage.get(identifier, 0) == 0:
                self.simulation.update_synapse_weight(identifier, 0.0)
                self._pruned_ids.append(identifier)

    def expose(
        self,
        modality: DevelopmentalModality | PrenatalModality | str,
        *,
        intensity: float = 0.7,
        duration_seconds: float = 1.0,
    ) -> DevelopmentalExposure:
        """Process a bounded scalar cue; no raw sensory content is retained."""
        try:
            selected = DevelopmentalModality(modality)
        except ValueError as error:
            raise ValueError(f"unsupported modality: {modality}") from error
        if (
            isinstance(intensity, bool)
            or not isinstance(intensity, (int, float))
            or not math.isfinite(intensity)
            or not 0.0 <= intensity <= 1.0
        ):
            raise ValueError("intensity must be finite and in [0, 1]")
        if (
            isinstance(duration_seconds, bool)
            or not isinstance(duration_seconds, (int, float))
            or not math.isfinite(duration_seconds)
            or not 0.0 < duration_seconds <= 60.0
        ):
            raise ValueError("duration_seconds must be finite and in (0, 60]")

        if selected in POSTNATAL_PATHS:
            route = POSTNATAL_PATHS[selected]
            onset_day = POSTNATAL_ONSET_DAYS[selected]
        else:
            prenatal_modality = PrenatalModality(selected.value)
            route = MODALITY_PATHS[prenatal_modality]
            onset_day = int(MODALITY_ONSET_WEEK[prenatal_modality] * 7)
        before_weights = self._route_weights(route)
        if self.developmental_day < onset_day:
            return DevelopmentalExposure(
                selected.value, False, "modality_not_yet_available_in_model",
                0.0, 0, self._mean(before_weights), self._mean(before_weights),
            )
        gate = min(self._maturity[item] for item in route)
        if gate <= 0.0 or intensity == 0.0:
            return DevelopmentalExposure(
                selected.value, False, "target_population_not_mature_or_zero_signal",
                gate, 0, self._mean(before_weights), self._mean(before_weights),
            )

        steps = max(6, min(24, math.ceil(float(duration_seconds) * 6.0)))
        for index in range(steps):
            current = {route[0]: 900.0 * float(intensity)} if index == 0 else None
            self._step(current, learn=True, count_experience_usage=True)
        self._accepted_exposures += 1
        after_weights = self._route_weights(route)
        updated = sum(
            abs(after - before) > 1e-12
            for before, after in zip(before_weights, after_weights, strict=True)
        )
        return DevelopmentalExposure(
            selected.value, True, "processed_and_discarded",
            round(gate, 6), updated, self._mean(before_weights), self._mean(after_weights),
        )

    def _route_weights(self, route: tuple[str, ...]) -> list[float]:
        values = []
        for source, target in zip(route, route[1:], strict=False):
            identifier = self._projection_ids.get((source, target))
            if identifier is None:
                identifier = next(
                    (
                        item for item in self._staged_ids
                        if self.simulation.synapses[item].source == source
                        and self.simulation.synapses[item].target == target
                    ),
                    None,
                )
            if identifier is not None:
                values.append(self.simulation.synapses[identifier].weight)
        return values

    @staticmethod
    def _mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 8) if values else 0.0

    def probe(
        self, modality: DevelopmentalModality | PrenatalModality | str,
        *, intensity: float = 0.7,
    ) -> tuple[str, ...]:
        """Run a no-learning sensory probe and return emitted spikes."""
        selected = DevelopmentalModality(modality)
        if selected in POSTNATAL_PATHS:
            route = POSTNATAL_PATHS[selected]
            onset_day = POSTNATAL_ONSET_DAYS[selected]
        else:
            prenatal_modality = PrenatalModality(selected.value)
            route = MODALITY_PATHS[prenatal_modality]
            onset_day = int(MODALITY_ONSET_WEEK[prenatal_modality] * 7)
        if self.developmental_day < onset_day:
            return ()
        if isinstance(intensity, bool) or not math.isfinite(intensity) or not 0 <= intensity <= 1:
            raise ValueError("intensity must be finite and in [0, 1]")
        spikes: list[str] = []
        for index in range(12):
            current = {route[0]: 900.0 * intensity} if index == 0 else None
            spikes.extend(self._step(current, learn=False).spikes)
        return tuple(spikes)

    def snapshot(self) -> DevelopmentalSnapshot:
        network = self.simulation.snapshot().as_dict()
        return DevelopmentalSnapshot(
            version=self.VERSION,
            seed=self.seed,
            developmental_day=self.developmental_day,
            stage=self.stage.value,
            population_maturity=tuple(sorted(self._maturity.items())),
            accepted_exposures=self._accepted_exposures,
            plasticity_updates=self._plasticity_updates,
            staged_synapses_created=tuple(self._staged_ids),
            functionally_pruned_synapses=tuple(self._pruned_ids),
            synapse_usage=tuple(sorted(self._usage.items())),
            network=network,
        )

    def reset(self) -> None:
        """Reset all learned and developmental state to source-defined defaults."""
        fresh = DevelopmentalNeuralSubstrate(seed=self.seed)
        self.__dict__.clear()
        self.__dict__.update(fresh.__dict__)


def run_developmental_learning_comparison(
    *, seed: int = 17, training_exposures: int = 20
) -> dict[str, object]:
    """Compare repeated experience with a frozen-weight baseline.

    Both networks share a seed, developmental trajectory and simulation-step
    budget. The control receives matched quiet steps; only the trained network
    receives auditory input and plasticity. The held-out probe uses no learning.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if (
        isinstance(training_exposures, bool)
        or not isinstance(training_exposures, int)
        or not 1 <= training_exposures <= 100
    ):
        raise ValueError("training_exposures must be in [1, 100]")
    baseline = DevelopmentalNeuralSubstrate(seed=seed)
    trained = DevelopmentalNeuralSubstrate(seed=seed)
    baseline.advance(1_200)
    trained.advance(1_200)
    initial_weights = {
        identifier: synapse.weight
        for identifier, synapse in baseline.simulation.synapses.items()
    }
    for _ in range(training_exposures):
        trained.expose(DevelopmentalModality.AUDITORY, intensity=0.8)

    # Match every dynamic variable to the trained network, then restore only
    # the frozen baseline weights. This isolates the effect of learning rather
    # than comparing networks with different membrane/event-queue histories.
    baseline.simulation.restore(trained.simulation.state())
    for identifier, weight in initial_weights.items():
        baseline.simulation.update_synapse_weight(identifier, weight)

    route = ("sensory_input", "recurrent_integration", "association", "sequence_context")
    target = "sequence_context"

    def probe_metrics(network: DevelopmentalNeuralSubstrate) -> tuple[int, int | None]:
        emitted: list[str] = []
        latency: int | None = None
        for index in range(12):
            current = {route[0]: 720.0} if index == 0 else None
            step = network._step(current, learn=False)
            emitted.extend(step.spikes)
            if target in step.spikes and latency is None:
                latency = index
        return sum(item == target for item in emitted), latency

    baseline_target, baseline_latency = probe_metrics(baseline)
    trained_target, trained_latency = probe_metrics(trained)
    baseline_path = baseline._route_weights(route)
    trained_path = trained._route_weights(route)
    behavioural_improvement = (
        trained_target > baseline_target
        or (
            trained_latency is not None
            and (baseline_latency is None or trained_latency < baseline_latency)
        )
    )
    return {
        "seed": seed,
        "training_exposures": training_exposures,
        "baseline_target_spikes": baseline_target,
        "trained_target_spikes": trained_target,
        "baseline_target_latency_steps": baseline_latency,
        "trained_target_latency_steps": trained_latency,
        "baseline_path_weight": DevelopmentalNeuralSubstrate._mean(baseline_path),
        "trained_path_weight": DevelopmentalNeuralSubstrate._mean(trained_path),
        "behavioural_improvement": behavioural_improvement,
        "path_efficacy_improvement": DevelopmentalNeuralSubstrate._mean(trained_path)
        > DevelopmentalNeuralSubstrate._mean(baseline_path),
        "limitations": [
            "Synthetic one-dimensional pathway task, not a human developmental outcome.",
            "The STDP-inspired rule and stage boundaries are engineering hypotheses.",
            "No individual age, fetal, clinical, myelin or consciousness prediction is made.",
        ],
    }
