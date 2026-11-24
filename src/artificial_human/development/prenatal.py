"""A small, inspectable model of prenatal neural development.

This is an engineering experiment, not a biological fetal-brain simulator.
It represents a stable abstract circuit scaffold whose population maturity and
connection strengths change over a 40-week gestational timeline. Individual
stimuli are not retained: only aggregate counters and plastic connection
strengths remain in the in-memory object.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite, sin
from typing import Final


GESTATION_DAYS: Final[int] = 280
GESTATION_WEEKS: Final[int] = 40
BLUEPRINT_VERSION: Final[str] = "prenatal-scaffold-v1"


class PrenatalPhase(StrEnum):
    """Broad gestational phases using obstetric (LMP) dating."""

    EARLY_EMBRYONIC = "early_embryonic"
    NEURULATION = "neurulation_and_early_patterning"
    REGIONAL_PATTERNING = "regional_patterning"
    PROLIFERATION_AND_MIGRATION = "proliferation_and_migration"
    EARLY_CIRCUIT_ASSEMBLY = "early_circuit_assembly"
    SENSORY_REFINEMENT = "sensory_refinement"
    LATE_GESTATION = "late_gestation"


class PrenatalModality(StrEnum):
    """Explicit, low-dimensional simulated sensory channels."""

    TACTILE = "tactile"
    VESTIBULAR = "vestibular"
    AUDITORY = "auditory"


@dataclass(frozen=True, slots=True)
class PopulationDefinition:
    """An abstract neural population, not a count of biological neurons."""

    identifier: str
    role: str
    capacity_units: int
    maturation_start_week: float
    maturation_end_week: float


@dataclass(frozen=True, slots=True)
class ProjectionDefinition:
    """A fixed directed connection whose efficacy may change through plasticity."""

    source: str
    target: str
    initial_strength: float
    learning_rate: float
    polarity: str = "excitatory"


@dataclass(frozen=True, slots=True)
class DevelopmentalMilestone:
    week: int
    title: str
    description: str


@dataclass(frozen=True, slots=True)
class PopulationSnapshot:
    identifier: str
    role: str
    capacity_units: int
    maturity: float
    activity: float


@dataclass(frozen=True, slots=True)
class ProjectionSnapshot:
    source: str
    target: str
    polarity: str
    strength: float
    update_count: int


@dataclass(frozen=True, slots=True)
class ExposureOutcome:
    modality: str
    available: bool
    reason: str
    plasticity_gate: float
    synapses_updated: int
    mean_strength_before: float
    mean_strength_after: float


@dataclass(frozen=True, slots=True)
class PrenatalSnapshot:
    age_days: int
    gestational_week: float
    completed_week: int
    phase: str
    blueprint_version: str
    population_count: int
    projection_count: int
    accepted_exposures: int
    total_plasticity_updates: int
    mean_projection_strength: float
    populations: tuple[PopulationSnapshot, ...]
    projections: tuple[ProjectionSnapshot, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible, content-free inspection snapshot."""
        return {
            "age_days": self.age_days,
            "gestational_week": self.gestational_week,
            "completed_week": self.completed_week,
            "phase": self.phase,
            "blueprint_version": self.blueprint_version,
            "population_count": self.population_count,
            "projection_count": self.projection_count,
            "accepted_exposures": self.accepted_exposures,
            "total_plasticity_updates": self.total_plasticity_updates,
            "mean_projection_strength": self.mean_projection_strength,
            "populations": [asdict(item) for item in self.populations],
            "projections": [asdict(item) for item in self.projections],
        }


@dataclass(slots=True)
class _PopulationState:
    definition: PopulationDefinition
    maturity: float = 0.0
    activity: float = 0.0


@dataclass(slots=True)
class _ProjectionState:
    definition: ProjectionDefinition
    strength: float
    update_count: int = 0


# Functional population abstractions; not one-to-one anatomical regions.
POPULATIONS: Final[tuple[PopulationDefinition, ...]] = (
    PopulationDefinition("sensory_input", "sensory input and relay", 12, 4.0, 24.0),
    PopulationDefinition("recurrent_integration", "recurrent integration", 16, 5.0, 32.0),
    PopulationDefinition("association", "distributed association", 18, 8.0, 36.0),
    PopulationDefinition("sequence_context", "sequence and context", 12, 11.0, 40.0),
    PopulationDefinition("action_prediction", "sensorimotor prediction", 14, 12.0, 40.0),
    PopulationDefinition("regulation", "state regulation", 10, 5.0, 32.0),
    PopulationDefinition("inhibitory_control", "local inhibitory control", 8, 8.0, 36.0),
)

# The topology is stable. Learning changes efficacy, never nodes or endpoints.
PROJECTIONS: Final[tuple[ProjectionDefinition, ...]] = (
    ProjectionDefinition("sensory_input", "recurrent_integration", 0.12, 0.12),
    ProjectionDefinition("recurrent_integration", "sensory_input", 0.08, 0.05),
    ProjectionDefinition("recurrent_integration", "association", 0.10, 0.12),
    ProjectionDefinition("association", "recurrent_integration", 0.08, 0.05),
    ProjectionDefinition("association", "sequence_context", 0.08, 0.10),
    ProjectionDefinition("sequence_context", "association", 0.08, 0.07),
    ProjectionDefinition("recurrent_integration", "action_prediction", 0.07, 0.09),
    ProjectionDefinition("action_prediction", "association", 0.07, 0.07),
    ProjectionDefinition("regulation", "recurrent_integration", 0.10, 0.04),
    ProjectionDefinition("inhibitory_control", "association", 0.10, 0.05, "inhibitory"),
)

MODALITY_ONSET_WEEK: Final[dict[PrenatalModality, float]] = {
    PrenatalModality.TACTILE: 10.0,
    PrenatalModality.VESTIBULAR: 12.0,
    PrenatalModality.AUDITORY: 24.0,
}

MODALITY_PATHS: Final[dict[PrenatalModality, tuple[str, ...]]] = {
    PrenatalModality.TACTILE: (
        "sensory_input", "recurrent_integration", "association"
    ),
    PrenatalModality.VESTIBULAR: (
        "sensory_input", "recurrent_integration", "action_prediction", "association"
    ),
    PrenatalModality.AUDITORY: (
        "sensory_input", "recurrent_integration", "association", "sequence_context"
    ),
}

MILESTONES: Final[tuple[DevelopmentalMilestone, ...]] = (
    DevelopmentalMilestone(
        5, "Neural induction",
        "The neural plate and early neural patterning begin in obstetric gestational dating.",
    ),
    DevelopmentalMilestone(
        6, "Neurulation",
        "Neural tube closure and early rostrocaudal patterning occur during early embryonic development.",
    ),
    DevelopmentalMilestone(
        8, "Regional patterning",
        "Early brain divisions and forebrain patterning continue; structures remain immature.",
    ),
    DevelopmentalMilestone(
        10, "Progenitor expansion",
        "Neural progenitor proliferation expands; timing and rate differ across cell populations.",
    ),
    DevelopmentalMilestone(
        12, "Neuronal migration",
        "Large-scale neuronal migration and early cortical organisation are underway.",
    ),
    DevelopmentalMilestone(
        17, "Mid-gestation growth",
        "Some cortical progenitor populations show high proliferation around mid-gestation.",
    ),
    DevelopmentalMilestone(
        20, "Cortical organisation",
        "Migration, layering and the organisation of developing cortical circuits continue.",
    ),
    DevelopmentalMilestone(
        22, "Early circuit formation",
        "Axonal and dendritic growth and early cortical synaptogenesis become increasingly prominent.",
    ),
    DevelopmentalMilestone(
        24, "Sensory pathway maturation",
        "Auditory pathways continue maturing; fetal responses to sound are reported in this period.",
    ),
    DevelopmentalMilestone(
        28, "Increasing sensory responsiveness",
        "Auditory and other sensory processing continue to develop; in-utero signals are filtered.",
    ),
    DevelopmentalMilestone(
        36, "Late gestation",
        "Network organisation continues. Birth does not mark an adult-like or fully mature brain.",
    ),
)


def phase_for_week(week: float) -> PrenatalPhase:
    """Map gestational age to a coarse developmental phase."""
    if not isfinite(week) or not 0.0 <= week <= float(GESTATION_WEEKS):
        raise ValueError("Gestational week must be between 0 and 40")
    if week < 5:
        return PrenatalPhase.EARLY_EMBRYONIC
    if week < 8:
        return PrenatalPhase.NEURULATION
    if week < 12:
        return PrenatalPhase.REGIONAL_PATTERNING
    if week < 22:
        return PrenatalPhase.PROLIFERATION_AND_MIGRATION
    if week < 28:
        return PrenatalPhase.EARLY_CIRCUIT_ASSEMBLY
    if week < 36:
        return PrenatalPhase.SENSORY_REFINEMENT
    return PrenatalPhase.LATE_GESTATION


def milestones_reached(week: float) -> tuple[DevelopmentalMilestone, ...]:
    """Return approximate reference milestones reached by this gestational week."""
    if not isfinite(week) or not 0.0 <= week <= float(GESTATION_WEEKS):
        raise ValueError("Gestational week must be between 0 and 40")
    return tuple(item for item in MILESTONES if item.week <= week)


class PrenatalBrain:
    """An in-memory developmental network with fixed topology and plastic weights.

    Advancing time runs a coarse developmental programme and deterministic
    intrinsic activity. Exposures process one explicit low-dimensional cue.
    The cue is discarded; only connection efficacy, current population activity,
    and aggregate counters remain in this object.
    """

    def __init__(self, *, seed: int = 0) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Seed must be an integer")
        self.seed = seed
        self._age_days = 0
        self._accepted_exposures = 0
        self._total_plasticity_updates = 0
        self._populations = {
            item.identifier: _PopulationState(item) for item in POPULATIONS
        }
        self._projections = {
            (item.source, item.target): _ProjectionState(item, item.initial_strength)
            for item in PROJECTIONS
        }
        self._topology_signature = self.structure_signature()
        self._refresh_maturity()

    @property
    def age_days(self) -> int:
        return self._age_days

    @property
    def gestational_week(self) -> float:
        return round(self._age_days / 7.0, 4)

    @property
    def phase(self) -> PrenatalPhase:
        return phase_for_week(self.gestational_week)

    def structure_signature(self) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
        """Return the immutable population and connection layout."""
        return (
            tuple(item.identifier for item in POPULATIONS),
            tuple((item.source, item.target) for item in PROJECTIONS),
        )

    def advance(self, days: int = 1) -> None:
        """Advance by whole gestational days without crossing the term boundary."""
        if isinstance(days, bool) or not isinstance(days, int) or days < 0:
            raise ValueError("Days must be a non-negative integer")
        if self._age_days + days > GESTATION_DAYS:
            raise ValueError("Cannot advance beyond 280 gestational days")
        for _ in range(days):
            self._age_days += 1
            self._refresh_maturity()
            self._run_intrinsic_activity()

    def _refresh_maturity(self) -> None:
        week = self.gestational_week
        for state in self._populations.values():
            definition = state.definition
            span = definition.maturation_end_week - definition.maturation_start_week
            state.maturity = round(
                max(0.0, min(1.0, (week - definition.maturation_start_week) / span)),
                6,
            )

    def _run_intrinsic_activity(self) -> None:
        """Apply a deterministic, low-amplitude spontaneous-activity approximation."""
        if self.gestational_week < 8.0:
            return
        for index, state in enumerate(self._populations.values()):
            phase = (self.seed % 997) * 0.001 + index * 0.73
            wave = 0.5 + 0.5 * sin(self._age_days * 0.11 + phase)
            state.activity = round(state.maturity * (0.08 + 0.14 * wave), 6)
        for projection in self._projections.values():
            source = self._populations[projection.definition.source]
            target = self._populations[projection.definition.target]
            gate = min(source.maturity, target.maturity) * 0.025
            self._strengthen(projection, gate)

    def _strengthen(self, projection: _ProjectionState, gate: float) -> bool:
        definition = projection.definition
        source = self._populations[definition.source]
        target = self._populations[definition.target]
        if gate <= 0.0 or source.activity <= 0.0 or target.activity <= 0.0:
            return False
        # Bounded Hebbian-style efficacy update. This is an engineering rule,
        # not a claim that one scalar reproduces biological synaptic plasticity.
        delta = (
            definition.learning_rate
            * min(1.0, gate)
            * source.activity
            * target.activity
            * (1.0 - projection.strength)
        )
        if delta <= 1e-12:
            return False
        projection.strength = round(min(1.0, projection.strength + delta), 8)
        projection.update_count += 1
        self._total_plasticity_updates += 1
        return True

    def expose(
        self,
        modality: PrenatalModality | str,
        *,
        intensity: float = 0.5,
        duration_seconds: float = 1.0,
    ) -> ExposureOutcome:
        """Process one explicit simulated cue, then discard it.

        No text, waveform, image, stimulus identifier, or exposure history is
        accepted or retained. A modality before its model onset is ignored.
        """
        try:
            selected_modality = PrenatalModality(modality)
        except ValueError as error:
            raise ValueError(f"Unsupported prenatal modality: {modality}") from error
        if (
            isinstance(intensity, bool)
            or not isinstance(intensity, (int, float))
            or not isfinite(intensity)
            or not 0.0 <= intensity <= 1.0
        ):
            raise ValueError("Intensity must be a finite number between 0 and 1")
        if (
            isinstance(duration_seconds, bool)
            or not isinstance(duration_seconds, (int, float))
            or not isfinite(duration_seconds)
            or not 0.0 < duration_seconds <= 60.0
        ):
            raise ValueError("Duration must be greater than 0 and at most 60 seconds")

        week = self.gestational_week
        route = MODALITY_PATHS[selected_modality]
        projections = [
            self._projections[(source, target)]
            for source, target in zip(route, route[1:], strict=False)
        ]
        before = self._mean_strength(projections)
        onset = MODALITY_ONSET_WEEK[selected_modality]
        if week < onset:
            return ExposureOutcome(
                selected_modality.value, False,
                "modality_not_yet_available_in_this_model", 0.0, 0, before, before,
            )

        maturity = min(self._populations[item].maturity for item in route)
        readiness = min(1.0, (week - onset + 1.0) / 4.0)
        gate = round(maturity * readiness, 6)
        if gate <= 0.0 or intensity == 0.0:
            return ExposureOutcome(
                selected_modality.value, False,
                "target_circuit_not_mature_or_signal_is_zero", gate, 0, before, before,
            )

        signal = float(intensity) * min(1.0, float(duration_seconds) / 3.0)
        self._populations[route[0]].activity = max(
            self._populations[route[0]].activity, signal
        )
        for source_id, target_id in zip(route, route[1:], strict=False):
            source = self._populations[source_id]
            target = self._populations[target_id]
            projection = self._projections[(source_id, target_id)]
            target_activity = (
                signal
                * (0.35 + 0.65 * projection.strength)
                * (0.30 + 0.70 * target.maturity)
            )
            target.activity = max(target.activity, min(1.0, target_activity))

        count_before = self._total_plasticity_updates
        for projection in projections:
            self._strengthen(projection, gate)
        updated = self._total_plasticity_updates - count_before
        self._accepted_exposures += 1
        after = self._mean_strength(projections)
        return ExposureOutcome(
            selected_modality.value, True, "processed_and_discarded",
            gate, updated, before, after,
        )

    @staticmethod
    def _mean_strength(projections: list[_ProjectionState]) -> float:
        if not projections:
            return 0.0
        return round(sum(item.strength for item in projections) / len(projections), 8)

    def snapshot(self) -> PrenatalSnapshot:
        """Return inspectable state without stimulus content."""
        populations = tuple(
            PopulationSnapshot(
                identifier=item.identifier,
                role=item.role,
                capacity_units=item.capacity_units,
                maturity=state.maturity,
                activity=state.activity,
            )
            for item in POPULATIONS
            for state in (self._populations[item.identifier],)
        )
        projections = tuple(
            ProjectionSnapshot(
                source=item.source,
                target=item.target,
                polarity=item.polarity,
                strength=self._projections[(item.source, item.target)].strength,
                update_count=self._projections[(item.source, item.target)].update_count,
            )
            for item in PROJECTIONS
        )
        return PrenatalSnapshot(
            age_days=self._age_days,
            gestational_week=self.gestational_week,
            completed_week=self._age_days // 7,
            phase=self.phase.value,
            blueprint_version=BLUEPRINT_VERSION,
            population_count=len(populations),
            projection_count=len(projections),
            accepted_exposures=self._accepted_exposures,
            total_plasticity_updates=self._total_plasticity_updates,
            mean_projection_strength=round(
                sum(item.strength for item in projections) / len(projections), 8
            ),
            populations=populations,
            projections=projections,
        )
