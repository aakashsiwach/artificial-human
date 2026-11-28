"""Deterministic, in-memory neural simulation primitives."""

from artificial_human.neural.circuits import build_recurrent_microcircuit
from artificial_human.neural.simulation import (
    LIFNeuron,
    LIFParameters,
    NeuronSnapshot,
    NeuralSimulation,
    NeuralSimulationState,
    SimulationCapacityError,
    Polarity,
    SimulationSnapshot,
    SimulationStep,
    Synapse,
)
from artificial_human.neural.population import (
    FeatureVectorBaselineEncoder,
    OneHotBaselineEncoder,
    PopulationAssociativeMemory,
    PopulationCode,
    PopulationEncoder,
    PopulationProjector,
    run_population_coding_comparison,
)
from artificial_human.neural.sensory import (
    AlignedPercepts,
    AuditionEncoder,
    EncodedPercept,
    InteroceptionEncoder,
    PermissionDeniedError,
    ProprioceptionEncoder,
    RetentionPolicy,
    SensoryEncoderSuite,
    SensoryEvent,
    SensoryModality,
    SensoryPermission,
    SensoryPermissionLedger,
    SensoryPipeline,
    SensoryProvenance,
    TouchEncoder,
    VestibularEncoder,
    VisionEncoder,
    project_percept,
)
from artificial_human.neural.homeostasis import (
    HomeostaticController,
    HomeostasisParameters,
    HomeostasisSnapshot,
    InternalModulatorySignals,
    run_homeostasis_comparison,
)

__all__ = [
    "LIFNeuron", "LIFParameters", "NeuronSnapshot", "NeuralSimulation",
    "NeuralSimulationState", "Polarity", "SimulationCapacityError",
    "SimulationSnapshot", "SimulationStep", "Synapse",
    "build_recurrent_microcircuit", "HomeostaticController",
    "HomeostasisParameters", "HomeostasisSnapshot", "InternalModulatorySignals",
    "run_homeostasis_comparison",
    "PopulationCode", "PopulationEncoder", "OneHotBaselineEncoder",
    "FeatureVectorBaselineEncoder", "PopulationAssociativeMemory",
    "PopulationProjector", "run_population_coding_comparison",
    "SensoryModality", "SensoryProvenance", "RetentionPolicy", "SensoryEvent",
    "SensoryPermission", "SensoryPermissionLedger", "PermissionDeniedError",
    "EncodedPercept", "AlignedPercepts", "SensoryEncoderSuite", "SensoryPipeline",
    "VisionEncoder", "AuditionEncoder", "TouchEncoder", "VestibularEncoder",
    "ProprioceptionEncoder", "InteroceptionEncoder", "project_percept",
]
