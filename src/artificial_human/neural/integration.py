"""Deterministic uncertainty-aware multisensory state estimation.

This is a bounded computational model for controlled synthetic tasks. It uses
reliability-weighted cue fusion and a simple Bayesian common-cause arbitration
hypothesis; it is not a biological or clinical perception model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from artificial_human.neural.population import PopulationCode, PopulationEncoder, PopulationProjector
from artificial_human.neural.sensory import EncodedPercept, SensoryModality
from artificial_human.neural.simulation import NeuralSimulation, SimulationStep


@dataclass(frozen=True, slots=True)
class SensoryEstimate:
    """A scalar estimate decoded from one modality's population representation."""

    modality: SensoryModality
    value: float
    variance: float
    timestamp: int
    source_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.modality, SensoryModality):
            raise ValueError("modality must be a SensoryModality")
        if isinstance(self.value, bool) or not math.isfinite(self.value) or not -1.0 <= self.value <= 1.0:
            raise ValueError("estimate value must be finite and in [-1, 1]")
        if isinstance(self.variance, bool) or not math.isfinite(self.variance) or not 1e-6 <= self.variance <= 4.0:
            raise ValueError("variance must be finite and in [1e-6, 4]")
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, int) or self.timestamp < 0:
            raise ValueError("timestamp must be a non-negative integer")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id is required")


def estimate_from_percept(
    percept: EncodedPercept,
    *,
    source_id: str,
    feature_index: int = 0,
    variance_floor: float = 0.01,
    uncertainty_scale: float = 0.5,
) -> SensoryEstimate | None:
    """Decode a normalized feature and convert percept uncertainty to variance.

    The variance mapping is an explicit engineering assumption for this
    synthetic task, not a statistically calibrated sensor-noise estimate.
    """
    if isinstance(feature_index, bool) or not isinstance(feature_index, int) or not 0 <= feature_index < percept.code.feature_count:
        raise ValueError("feature_index is out of range")
    if (
        isinstance(variance_floor, bool)
        or not math.isfinite(variance_floor)
        or not 1e-6 <= variance_floor <= 4.0
        or isinstance(uncertainty_scale, bool)
        or not math.isfinite(uncertainty_scale)
        or not 0.0 <= uncertainty_scale <= 4.0
    ):
        raise ValueError("variance mapping parameters are outside supported bounds")
    value = percept.code.decode()[feature_index]
    if value is None:
        return None
    variance = min(4.0, variance_floor + uncertainty_scale * percept.code.uncertainty)
    return SensoryEstimate(
        modality=percept.modality,
        value=value,
        variance=max(1e-6, variance),
        timestamp=percept.timestamp,
        source_id=source_id,
    )


@dataclass(frozen=True, slots=True)
class MultisensoryParameters:
    common_cause_prior: float = 0.8
    spatial_scale: float = 0.25
    temporal_scale: float = 2.0
    max_modalities: int = 6
    population_units: int = 9

    def __post_init__(self) -> None:
        if (
            isinstance(self.common_cause_prior, bool)
            or not math.isfinite(self.common_cause_prior)
            or not 0.0 <= self.common_cause_prior <= 1.0
        ):
            raise ValueError("common_cause_prior must be in [0, 1]")
        if isinstance(self.spatial_scale, bool) or not math.isfinite(self.spatial_scale) or not 1e-6 <= self.spatial_scale <= 2.0:
            raise ValueError("spatial_scale must be in [1e-6, 2]")
        if isinstance(self.temporal_scale, bool) or not math.isfinite(self.temporal_scale) or not 1e-6 <= self.temporal_scale <= 10_000:
            raise ValueError("temporal_scale must be in [1e-6, 10000]")
        if isinstance(self.max_modalities, bool) or not isinstance(self.max_modalities, int) or not 1 <= self.max_modalities <= 6:
            raise ValueError("max_modalities must be in [1, 6]")
        if isinstance(self.population_units, bool) or not isinstance(self.population_units, int) or not 2 <= self.population_units <= 128:
            raise ValueError("population_units must be in [2, 128]")


@dataclass(frozen=True, slots=True)
class MultisensoryEstimate:
    timestamp: int
    value: float
    variance: float
    common_cause_probability: float
    fused_value: float | None
    fused_variance: float | None
    segregated_value: float
    naive_value: float
    modality_values: tuple[tuple[SensoryModality, float], ...]
    missing_modalities: tuple[SensoryModality, ...]
    population_code: PopulationCode

    def __post_init__(self) -> None:
        if not 0.0 <= self.common_cause_probability <= 1.0:
            raise ValueError("common-cause probability must be in [0, 1]")
        if not math.isfinite(self.value) or not -1.0 <= self.value <= 1.0:
            raise ValueError("integrated value must be in [-1, 1]")


class MultisensoryIntegrator:
    """Reliability-weighted fusion gated by temporal and spatial common-cause evidence."""

    def __init__(self, parameters: MultisensoryParameters | None = None) -> None:
        self.parameters = parameters or MultisensoryParameters()
        self.encoder = PopulationEncoder(units_per_feature=self.parameters.population_units)

    def integrate(
        self,
        estimates: tuple[SensoryEstimate, ...],
        *,
        timestamp: int,
        target_modality: SensoryModality,
        expected_modalities: tuple[SensoryModality, ...] | None = None,
    ) -> MultisensoryEstimate:
        if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
            raise ValueError("timestamp must be a non-negative integer")
        if not isinstance(target_modality, SensoryModality):
            raise ValueError("target_modality must be a SensoryModality")
        if not estimates:
            raise ValueError("at least one sensory estimate is required")
        if len(estimates) > self.parameters.max_modalities:
            raise ValueError("estimate count exceeds configured modality capacity")
        modalities = tuple(item.modality for item in estimates)
        if len(set(modalities)) != len(modalities):
            raise ValueError("only one estimate per modality is allowed")
        if target_modality not in modalities:
            raise ValueError("target modality must be present to define the segregated baseline")
        if any(item.timestamp > timestamp for item in estimates):
            raise ValueError("future-dated evidence cannot update the requested state")
        expected = expected_modalities or tuple(SensoryModality)
        if len(set(expected)) != len(expected) or any(not isinstance(item, SensoryModality) for item in expected):
            raise ValueError("expected_modalities must be unique modality values")
        if len(expected) > self.parameters.max_modalities:
            raise ValueError("expected modality count exceeds configured capacity")
        if not set(modalities).issubset(set(expected)):
            raise ValueError("received modality is not in expected_modalities")

        target = next(item for item in estimates if item.modality is target_modality)
        precision_sum = sum(1.0 / item.variance for item in estimates)
        fused_variance = 1.0 / precision_sum
        fused_value = sum(item.value / item.variance for item in estimates) / precision_sum
        naive_value = sum(item.value for item in estimates) / len(estimates)

        if len(estimates) == 1:
            common_probability = 0.0
        elif len({item.source_id for item in estimates}) > 1:
            common_probability = 0.0
        else:
            weighted_disparity = sum(
                (item.value - fused_value) ** 2 / item.variance for item in estimates
            ) / len(estimates)
            temporal_span = max(item.timestamp for item in estimates) - min(
                item.timestamp for item in estimates
            )
            spatial_likelihood = math.exp(
                -0.5 * weighted_disparity / (self.parameters.spatial_scale ** 2)
            )
            temporal_likelihood = math.exp(
                -0.5 * (temporal_span / self.parameters.temporal_scale) ** 2
            )
            common_likelihood = spatial_likelihood * temporal_likelihood
            prior = self.parameters.common_cause_prior
            numerator = prior * common_likelihood
            denominator = numerator + (1.0 - prior)
            common_probability = numerator / denominator if denominator > 0.0 else 0.0

        integrated_value = (
            common_probability * fused_value
            + (1.0 - common_probability) * target.value
        )
        integrated_value = min(1.0, max(-1.0, integrated_value))
        integrated_variance = (
            common_probability * fused_variance
            + (1.0 - common_probability) * target.variance
        )
        missing = tuple(item for item in expected if item not in modalities)
        code = self.encoder.encode(
            (integrated_value, common_probability, 1.0 / (1.0 + integrated_variance)),
            timestamp=timestamp,
            confidence=max(0.0, min(1.0, 1.0 - integrated_variance / 4.0)),
        )
        return MultisensoryEstimate(
            timestamp=timestamp,
            value=integrated_value,
            variance=integrated_variance,
            common_cause_probability=common_probability,
            fused_value=fused_value if len(estimates) > 1 else None,
            fused_variance=fused_variance if len(estimates) > 1 else None,
            segregated_value=target.value,
            naive_value=naive_value,
            modality_values=tuple((item.modality, item.value) for item in estimates),
            missing_modalities=missing,
            population_code=code,
        )


def project_integrated_state(
    estimate: MultisensoryEstimate,
    simulation: NeuralSimulation,
    neuron_ids: tuple[str, ...],
    *,
    current_scale: float = 450.0,
) -> SimulationStep:
    """Project the integrated state representation into the configured LIF circuit."""
    return PopulationProjector(neuron_ids, current_scale=current_scale).drive(
        estimate.population_code, simulation
    )


@dataclass(frozen=True, slots=True)
class MultisensoryComparison:
    trials: int
    independent_mae: float
    naive_concatenation_mae: float
    reliability_fusion_mae: float
    causal_integration_mae: float
    missing_modality_trials: int
    conflicting_modality_trials: int


def run_multisensory_comparison(trials: int = 200) -> MultisensoryComparison:
    """Compare baselines on a deterministic two-cue shared/segregated source task."""
    if isinstance(trials, bool) or not isinstance(trials, int) or not 20 <= trials <= 10_000:
        raise ValueError("trials must be an integer in [20, 10000]")
    integrator = MultisensoryIntegrator()
    errors_independent: list[float] = []
    errors_naive: list[float] = []
    errors_fusion: list[float] = []
    errors_causal: list[float] = []
    missing_count = 0
    conflicting_count = 0

    for index in range(trials):
        truth = -0.8 + 1.6 * ((index * 37) % trials) / max(1, trials - 1)
        vision_noise = (((index * 17) % 11) - 5) * 0.012
        proprio_noise = (((index * 13) % 9) - 4) * 0.02
        vision_value = min(1.0, max(-1.0, truth + vision_noise))
        proprio_value = min(1.0, max(-1.0, truth + proprio_noise))
        conflict = index % 5 == 0
        missing = index % 7 == 0
        if conflict:
            proprio_value = min(1.0, max(-1.0, proprio_value + (0.65 if index % 2 else -0.65)))
            conflicting_count += 1
        if missing:
            missing_count += 1

        estimates = [SensoryEstimate(
            SensoryModality.VISION, vision_value, 0.025, index, "shared-source"
        )]
        if not missing:
            estimates.append(SensoryEstimate(
                SensoryModality.PROPRIOCEPTION,
                proprio_value,
                0.08,
                index + (5 if index % 9 == 0 else 0),
                "shared-source",
            ))
        result = integrator.integrate(
            tuple(estimates),
            timestamp=index + 5,
            target_modality=SensoryModality.VISION,
            expected_modalities=(SensoryModality.VISION, SensoryModality.PROPRIOCEPTION),
        )
        errors_independent.append(abs(vision_value - truth))
        errors_naive.append(abs(result.naive_value - truth))
        errors_fusion.append(abs((result.fused_value if result.fused_value is not None else vision_value) - truth))
        errors_causal.append(abs(result.value - truth))

    def mean_absolute_error(values: list[float]) -> float:
        return round(sum(values) / len(values), 8)

    return MultisensoryComparison(
        trials=trials,
        independent_mae=mean_absolute_error(errors_independent),
        naive_concatenation_mae=mean_absolute_error(errors_naive),
        reliability_fusion_mae=mean_absolute_error(errors_fusion),
        causal_integration_mae=mean_absolute_error(errors_causal),
        missing_modality_trials=missing_count,
        conflicting_modality_trials=conflicting_count,
    )
