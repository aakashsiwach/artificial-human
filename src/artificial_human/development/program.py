"""One in-memory neural substrate with coarse prenatal and postnatal development.

Developmental units are engineering controls, not individual biological-age
predictions. The old aggregate PrenatalBrain is retained as a migration reference;
this program is the active neuron/synapse developmental path.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import math

from artificial_human.development.prenatal import (
    MODALITY_ONSET_WEEK,
    PrenatalModality,
    PrenatalPhase,
    phase_for_week,
)
from artificial_human.neural.homeostasis import HomeostaticController, HomeostasisParameters
from artificial_human.neural.plasticity import PairBasedSTDP, PlasticityController, STDPParameters
from artificial_human.neural.simulation import (
    LIFNeuron, LIFParameters, NeuralSimulation, Polarity, SimulationStep, Synapse,
)


class DevelopmentalStage(StrEnum):
    PRENATAL = "prenatal"
    EARLY_POSTNATAL = "early_postnatal"
    EXPERIENCE_REFINEMENT = "experience_refinement"
    LATE_MATURATION = "late_maturation"
    STABILIZED = "stabilized"


class DevelopmentalModality(StrEnum):
    TACTILE = "tactile"
    VESTIBULAR = "vestibular"
    AUDITORY = "auditory"
    VISION = "vision"
    PROPRIOCEPTION = "proprioception"
    INTEROCEPTION = "interoception"


@dataclass(frozen=True, slots=True)
class SynapseBlueprint:
    identifier: str
    source: str
    target: str
    weight: float
    delay_steps: int
    polarity: Polarity = Polarity.EXCITATORY
    prenatal_day: int = 281
    postnatal_unit: int = 0


@dataclass(frozen=True, slots=True)
class DevelopmentalExperienceOutcome:
    accepted: bool
    modality: str
    reason: str | None
    simulation_steps: int
    spikes: tuple[str, ...]
    total_weight_change: float


@dataclass(frozen=True, slots=True)
class DevelopmentalSnapshot:
    stage: DevelopmentalStage
    prenatal_days: int
    postnatal_units: int
    prenatal_phase: PrenatalPhase
    simulation_step: int
    maturity: float
    myelination_factor: float
    plasticity_gate: float
    available_modalities: tuple[DevelopmentalModality, ...]
    neuron_count: int
    synapse_count: int
    active_synapse_count: int
    pruned_synapses: tuple[str, ...]
    mean_active_weight: float
    mean_conduction_delay_steps: float
    experience_count: int
    total_spikes: int


@dataclass(frozen=True, slots=True)
class DevelopmentalComparison:
    training_trials: int
    developmental_prediction_spikes: int
    frozen_prediction_spikes: int
    mature_from_start_prediction_spikes: int
    developmental_association_prediction_weight: float
    frozen_association_prediction_weight: float
    mature_association_prediction_weight: float
    success: bool


_BASE = (
    SynapseBlueprint("tactile_to_integration", "sensory_tactile", "integration", 220.0, 3),
    SynapseBlueprint("integration_to_association", "integration", "association", 180.0, 3),
    SynapseBlueprint("association_recurrent", "association", "integration", 40.0, 4),
    SynapseBlueprint(
        "inhibition_to_association", "inhibitory", "association", 25.0, 4,
        polarity=Polarity.INHIBITORY,
    ),
)
_CANDIDATES = (
    SynapseBlueprint("vestibular_to_integration", "sensory_vestibular", "integration", 100.0, 4, prenatal_day=84),
    SynapseBlueprint("auditory_to_integration", "sensory_auditory", "integration", 100.0, 4, prenatal_day=168),
    SynapseBlueprint("visual_to_integration", "sensory_visual", "integration", 100.0, 4, postnatal_unit=1),
    SynapseBlueprint("proprioception_to_integration", "sensory_proprioceptive", "integration", 100.0, 3, postnatal_unit=1),
    SynapseBlueprint("interoception_to_integration", "sensory_interoceptive", "integration", 70.0, 4, postnatal_unit=1),
    SynapseBlueprint("integration_to_sequence", "integration", "sequence", 45.0, 4, postnatal_unit=100),
    SynapseBlueprint("sequence_to_association", "sequence", "association", 45.0, 4, postnatal_unit=100),
    SynapseBlueprint("association_to_prediction", "association", "prediction", 12.0, 3, postnatal_unit=500),
    SynapseBlueprint("auditory_to_prediction", "sensory_auditory", "prediction", 220.0, 1, postnatal_unit=500),
    SynapseBlueprint("prediction_to_motor", "prediction", "motor", 80.0, 4, postnatal_unit=500),
    SynapseBlueprint("association_to_motor", "association", "motor", 50.0, 4, postnatal_unit=1000),
    SynapseBlueprint("inhibition_to_integration", "inhibitory", "integration", 35.0, 3, Polarity.INHIBITORY, postnatal_unit=1000),
)
_NEURONS = (
    "sensory_tactile", "sensory_vestibular", "sensory_auditory",
    "sensory_visual", "sensory_proprioceptive", "sensory_interoceptive",
    "integration", "association", "sequence", "prediction", "inhibitory", "motor",
)
_MODALITY_NEURON = {
    DevelopmentalModality.TACTILE: "sensory_tactile",
    DevelopmentalModality.VESTIBULAR: "sensory_vestibular",
    DevelopmentalModality.AUDITORY: "sensory_auditory",
    DevelopmentalModality.VISION: "sensory_visual",
    DevelopmentalModality.PROPRIOCEPTION: "sensory_proprioceptive",
    DevelopmentalModality.INTEROCEPTION: "sensory_interoceptive",
}
_PROTECTED = {
    "tactile_to_integration", "integration_to_association", "association_recurrent",
    "association_to_prediction", "auditory_to_prediction", "prediction_to_motor",
}


class DevelopmentalProgram:
    """Developmental controller for a single bounded LIF neuron/synapse substrate."""

    MAX_PRENATAL_DAYS = 280
    MAX_POSTNATAL_UNITS = 10000

    def __init__(
        self,
        *,
        plasticity_enabled: bool = True,
        mature_from_start: bool = False,
    ) -> None:
        if not isinstance(plasticity_enabled, bool) or not isinstance(mature_from_start, bool):
            raise ValueError("development flags must be boolean")
        self.plasticity_enabled = plasticity_enabled
        self.mature_from_start = mature_from_start
        self.prenatal_days = 0
        self.postnatal_units = 0
        self.experience_count = 0
        self.total_spikes = 0
        self._usage: dict[str, int] = {}
        self._pruned: set[str] = set()
        self._added: set[str] = set()
        self._initial_delays: dict[str, int] = {}
        self._blueprints = {item.identifier: item for item in (*_BASE, *_CANDIDATES)}

        self.simulation = NeuralSimulation(
            max_neurons=16, max_synapses=32, max_pending_events=8192
        )
        immature = LIFParameters(
            membrane_time_constant_ms=24.0,
            refractory_period_ms=3.0,
        )
        for identifier in _NEURONS:
            self.simulation.add_neuron(LIFNeuron(identifier, parameters=immature))
        for blueprint in _BASE:
            self._add_synapse(blueprint)

        self.plasticity = PlasticityController(
            self.simulation,
            PairBasedSTDP(STDPParameters(
                potentiation=2.5,
                depression=1.6,
                trace_decay=0.9,
            )),
        )
        self.homeostasis = HomeostaticController(
            self.simulation,
            HomeostasisParameters(
                observation_window_steps=100,
                target_spike_rate=0.12,
                rate_ema_decay=0.8,
                adaptation_rate=0.05,
                deadband=0.04,
                minimum_input_gain=0.1,
                maximum_input_gain=1.0,
            ),
        )
        if mature_from_start:
            self.prenatal_days = self.MAX_PRENATAL_DAYS
            self.postnatal_units = 2000
            self._apply_development()

    @property
    def stage(self) -> DevelopmentalStage:
        if self.prenatal_days < self.MAX_PRENATAL_DAYS:
            return DevelopmentalStage.PRENATAL
        if self.postnatal_units < 100:
            return DevelopmentalStage.EARLY_POSTNATAL
        if self.postnatal_units < 1000:
            return DevelopmentalStage.EXPERIENCE_REFINEMENT
        if self.postnatal_units < 2000:
            return DevelopmentalStage.LATE_MATURATION
        return DevelopmentalStage.STABILIZED

    @property
    def maturity(self) -> float:
        prenatal = self.prenatal_days / self.MAX_PRENATAL_DAYS
        postnatal = min(1.0, self.postnatal_units / 2000.0)
        return min(1.0, 0.3 * prenatal + 0.7 * postnatal)

    @property
    def myelination_factor(self) -> float:
        prenatal = self.prenatal_days / self.MAX_PRENATAL_DAYS
        postnatal = min(1.0, self.postnatal_units / 2000.0)
        return min(1.0, 0.2 * prenatal + 0.8 * postnatal)

    @property
    def plasticity_gate(self) -> float:
        return max(0.05, 1.0 - 0.85 * self.maturity)

    @property
    def available_modalities(self) -> tuple[DevelopmentalModality, ...]:
        available: list[DevelopmentalModality] = []
        week = self.prenatal_days / 7.0
        for modality in (
            DevelopmentalModality.TACTILE,
            DevelopmentalModality.VESTIBULAR,
            DevelopmentalModality.AUDITORY,
        ):
            onset = MODALITY_ONSET_WEEK[PrenatalModality(modality.value)]
            if self.prenatal_days == self.MAX_PRENATAL_DAYS or week >= onset:
                available.append(modality)
        if self.postnatal_units >= 1:
            available.extend((
                DevelopmentalModality.VISION,
                DevelopmentalModality.PROPRIOCEPTION,
                DevelopmentalModality.INTEROCEPTION,
            ))
        return tuple(available)

    def advance_prenatal(self, days: int) -> DevelopmentalSnapshot:
        if isinstance(days, bool) or not isinstance(days, int) or days < 0:
            raise ValueError("prenatal days must be a non-negative integer")
        if self.postnatal_units:
            raise ValueError("prenatal time cannot advance after the birth boundary")
        if self.prenatal_days + days > self.MAX_PRENATAL_DAYS:
            raise ValueError("prenatal time cannot exceed 280 simulated days")
        self.prenatal_days += days
        self._apply_development()
        return self.snapshot()

    def advance_postnatal(self, units: int) -> DevelopmentalSnapshot:
        if self.prenatal_days != self.MAX_PRENATAL_DAYS:
            raise ValueError("postnatal development requires the simulated birth boundary")
        if isinstance(units, bool) or not isinstance(units, int) or units < 0:
            raise ValueError("postnatal units must be a non-negative integer")
        if self.postnatal_units + units > self.MAX_POSTNATAL_UNITS:
            raise ValueError("postnatal units exceed the configured budget")
        self.postnatal_units += units
        self._apply_development()
        return self.snapshot()

    def _add_synapse(self, item: SynapseBlueprint) -> None:
        self.simulation.add_synapse(Synapse(
            item.identifier, item.source, item.target, item.weight,
            polarity=item.polarity,
            delay_steps=item.delay_steps,
            maximum_weight=250.0,
        ))
        self._added.add(item.identifier)
        self._usage[item.identifier] = 0
        self._initial_delays[item.identifier] = item.delay_steps

    def _apply_development(self) -> None:
        for item in _CANDIDATES:
            if item.identifier in self._added:
                continue
            prenatal_ready = item.postnatal_unit == 0 and item.prenatal_day <= self.prenatal_days
            postnatal_ready = item.postnatal_unit > 0 and item.postnatal_unit <= self.postnatal_units
            if prenatal_ready or postnatal_ready:
                self._add_synapse(item)

        maturity = self.maturity
        tau = 24.0 - 4.0 * maturity
        refractory = 3.0 - maturity
        for neuron in self.simulation.neurons.values():
            neuron.parameters = replace(
                neuron.parameters,
                membrane_time_constant_ms=tau,
                refractory_period_ms=refractory,
            )

        myelination = self.myelination_factor
        for identifier in self._added:
            initial_delay = self._initial_delays[identifier]
            delay = max(1, round(initial_delay * (1.0 - 0.5 * myelination)))
            self.simulation.update_synapse_delay(identifier, delay)

    def _observe_step(self, step: SimulationStep) -> None:
        self.total_spikes += len(step.spikes)
        fired = set(step.spikes)
        for identifier, synapse in self.simulation.synapses.items():
            if synapse.source in fired:
                self._usage[identifier] = self._usage.get(identifier, 0) + 1
        if self.plasticity_enabled:
            self.plasticity.observe_step(step.spikes, gate=self.plasticity_gate)
        self.homeostasis.observe_step(step)
        for identifier in self._pruned:
            self.simulation.update_synapse_weight(identifier, 0.0)

    def advance_simulation(self, steps: int) -> tuple[SimulationStep, ...]:
        if isinstance(steps, bool) or not isinstance(steps, int) or not 0 <= steps <= 10000:
            raise ValueError("simulation steps must be an integer in [0, 10000]")
        results: list[SimulationStep] = []
        for _ in range(steps):
            step = self.simulation.step()
            self._observe_step(step)
            results.append(step)
        return tuple(results)

    def experience(
        self,
        modality: DevelopmentalModality | str,
        *,
        intensity: float = 0.8,
        duration_steps: int = 1,
    ) -> DevelopmentalExperienceOutcome:
        try:
            selected = DevelopmentalModality(modality)
        except (TypeError, ValueError):
            raise ValueError("unknown developmental sensory modality") from None
        if (
            isinstance(intensity, bool)
            or not isinstance(intensity, (int, float))
            or not math.isfinite(intensity)
            or not 0.0 <= intensity <= 1.0
        ):
            raise ValueError("experience intensity must be finite and in [0, 1]")
        if (
            isinstance(duration_steps, bool)
            or not isinstance(duration_steps, int)
            or not 1 <= duration_steps <= 64
        ):
            raise ValueError("duration_steps must be an integer in [1, 64]")
        if selected not in self.available_modalities:
            return DevelopmentalExperienceOutcome(
                False, selected.value,
                "modality_not_available_at_current_developmental_stage",
                0, (), 0.0,
            )

        neuron_id = _MODALITY_NEURON[selected]
        before = {
            key: value.weight for key, value in self.simulation.synapses.items()
        }
        spikes: list[str] = []
        for _ in range(duration_steps):
            step = self.simulation.step({neuron_id: float(intensity) * 500.0})
            self._observe_step(step)
            spikes.extend(step.spikes)
        self.experience_count += 1
        weight_change = sum(
            abs(self.simulation.synapses[key].weight - weight)
            for key, weight in before.items()
        )
        return DevelopmentalExperienceOutcome(
            True, selected.value, None, duration_steps,
            tuple(spikes), round(weight_change, 8),
        )

    def prune_unused(self, *, minimum_usage: int = 2) -> tuple[str, ...]:
        if self.stage not in {
            DevelopmentalStage.LATE_MATURATION,
            DevelopmentalStage.STABILIZED,
        }:
            raise ValueError("functional pruning is only enabled in late maturation")
        if isinstance(minimum_usage, bool) or not isinstance(minimum_usage, int) or minimum_usage < 0:
            raise ValueError("minimum_usage must be a non-negative integer")
        pruned: list[str] = []
        for item in _CANDIDATES:
            identifier = item.identifier
            if (
                identifier in self._added
                and identifier not in _PROTECTED
                and self._usage.get(identifier, 0) < minimum_usage
            ):
                self.simulation.cancel_pending_events(synapse_id=identifier)
                self.simulation.update_synapse_weight(identifier, 0.0)
                self._pruned.add(identifier)
                pruned.append(identifier)
        return tuple(sorted(pruned))

    def snapshot(self) -> DevelopmentalSnapshot:
        synapses = tuple(self.simulation.synapses.values())
        active = tuple(item for item in synapses if item.weight > 0)
        week = min(40.0, self.prenatal_days / 7.0)
        return DevelopmentalSnapshot(
            stage=self.stage,
            prenatal_days=self.prenatal_days,
            postnatal_units=self.postnatal_units,
            prenatal_phase=phase_for_week(week),
            simulation_step=self.simulation.step_index,
            maturity=round(self.maturity, 8),
            myelination_factor=round(self.myelination_factor, 8),
            plasticity_gate=round(self.plasticity_gate, 8),
            available_modalities=self.available_modalities,
            neuron_count=len(self.simulation.neurons),
            synapse_count=len(synapses),
            active_synapse_count=len(active),
            pruned_synapses=tuple(sorted(self._pruned)),
            mean_active_weight=round(
                sum(item.weight for item in active) / len(active) if active else 0.0,
                8,
            ),
            mean_conduction_delay_steps=round(
                sum(item.delay_steps for item in active) / len(active) if active else 0.0,
                8,
            ),
            experience_count=self.experience_count,
            total_spikes=self.total_spikes,
        )


def run_developmental_comparison(
    *, training_trials: int = 60,
) -> DevelopmentalComparison:
    """Test cue-only prediction after paired experience vs frozen baselines."""
    if (
        isinstance(training_trials, bool)
        or not isinstance(training_trials, int)
        or not 1 <= training_trials <= 250
    ):
        raise ValueError("training_trials must be an integer in [1, 250]")

    developmental = DevelopmentalProgram(plasticity_enabled=True)
    frozen = DevelopmentalProgram(plasticity_enabled=False)
    mature = DevelopmentalProgram(plasticity_enabled=False, mature_from_start=True)
    for program in (developmental, frozen):
        program.advance_prenatal(280)
        program.advance_postnatal(500)

    programs = (developmental, frozen, mature)
    for _ in range(training_trials):
        for program in programs:
            program.experience(DevelopmentalModality.TACTILE, intensity=0.95)
            program.advance_simulation(2)
            program.experience(DevelopmentalModality.AUDITORY, intensity=0.95)
            program.advance_simulation(3)
    for program in programs:
        program.advance_simulation(20)

    developmental.plasticity_enabled = False

    def cue_probe(program: DevelopmentalProgram) -> int:
        before = program.simulation.neurons["prediction"].spike_count
        program.experience(DevelopmentalModality.TACTILE, intensity=0.95)
        program.advance_simulation(14)
        return program.simulation.neurons["prediction"].spike_count - before

    developmental_spikes = cue_probe(developmental)
    frozen_spikes = cue_probe(frozen)
    mature_spikes = cue_probe(mature)
    learned_weight = developmental.simulation.synapses["association_to_prediction"].weight
    frozen_weight = frozen.simulation.synapses["association_to_prediction"].weight
    mature_weight = mature.simulation.synapses["association_to_prediction"].weight
    return DevelopmentalComparison(
        training_trials,
        developmental_spikes,
        frozen_spikes,
        mature_spikes,
        round(learned_weight, 8),
        round(frozen_weight, 8),
        round(mature_weight, 8),
        developmental_spikes > frozen_spikes
        and developmental_spikes > mature_spikes
        and learned_weight > frozen_weight,
    )
