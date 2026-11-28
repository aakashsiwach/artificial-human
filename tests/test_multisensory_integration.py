from __future__ import annotations

import pytest

from artificial_human.neural.integration import (
    MultisensoryIntegrator,
    MultisensoryParameters,
    SensoryEstimate,
    estimate_from_percept,
    project_integrated_state,
    run_multisensory_comparison,
)
from artificial_human.neural.population import PopulationEncoder
from artificial_human.neural.sensory import (
    EncodedPercept,
    RetentionPolicy,
    SensoryEvent,
    SensoryModality,
    SensoryPermissionLedger,
    SensoryPipeline,
)
from artificial_human.neural.simulation import LIFNeuron, NeuralSimulation


def estimate(
    modality: SensoryModality,
    value: float,
    *,
    variance: float = 0.04,
    timestamp: int = 10,
    source_id: str = "same-event",
) -> SensoryEstimate:
    return SensoryEstimate(modality, value, variance, timestamp, source_id)


def test_estimate_is_decoded_from_permission_gated_population_percept() -> None:
    ledger = SensoryPermissionLedger()
    ledger.grant(
        SensoryModality.VISION,
        purpose="synthetic shared-target task",
        granted_at=0,
        expires_at=20,
        retention=RetentionPolicy.EPHEMERAL,
        explicit_consent=True,
    )
    pipeline = SensoryPipeline(ledger=ledger)
    percept = pipeline.ingest(
        SensoryEvent(SensoryModality.VISION, (0.35, 0.35, 0.35), 3, 0.9)
    )
    decoded = estimate_from_percept(percept, source_id="target-3")
    assert decoded is not None
    assert decoded.modality is SensoryModality.VISION
    assert decoded.value == pytest.approx(percept.code.decode()[0])
    assert decoded.variance > 0.0
    assert decoded.timestamp == 3
    assert decoded.source_id == "target-3"


def test_common_cause_fuses_cues_by_inverse_variance_not_equal_average() -> None:
    integrator = MultisensoryIntegrator()
    vision = estimate(SensoryModality.VISION, 0.2, variance=0.01)
    proprioception = estimate(SensoryModality.PROPRIOCEPTION, 0.8, variance=0.09)
    result = integrator.integrate(
        (vision, proprioception),
        timestamp=10,
        target_modality=SensoryModality.VISION,
    )
    expected = (0.2 / 0.01 + 0.8 / 0.09) / (1 / 0.01 + 1 / 0.09)
    assert result.fused_value == pytest.approx(expected)
    assert result.common_cause_probability > 0.99
    assert result.value == pytest.approx(expected, abs=0.01)
    assert result.naive_value == pytest.approx(0.5)
    assert result.value != pytest.approx(result.naive_value)


def test_large_spatial_conflict_reduces_fusion_and_keeps_target_modality() -> None:
    integrator = MultisensoryIntegrator(
        MultisensoryParameters(common_cause_prior=0.9, spatial_scale=0.12)
    )
    result = integrator.integrate(
        (
            estimate(SensoryModality.VISION, -0.7, variance=0.02),
            estimate(SensoryModality.PROPRIOCEPTION, 0.7, variance=0.02),
        ),
        timestamp=10,
        target_modality=SensoryModality.VISION,
    )
    assert result.common_cause_probability < 0.01
    assert result.value == pytest.approx(-0.7, abs=0.01)
    assert result.fused_value == pytest.approx(0.0)


def test_temporal_mismatch_reduces_common_cause_probability() -> None:
    integrator = MultisensoryIntegrator()
    synchronized = integrator.integrate(
        (
            estimate(SensoryModality.VISION, 0.1, timestamp=10),
            estimate(SensoryModality.PROPRIOCEPTION, 0.12, timestamp=10),
        ),
        timestamp=10,
        target_modality=SensoryModality.VISION,
    )
    asynchronous = integrator.integrate(
        (
            estimate(SensoryModality.VISION, 0.1, timestamp=1),
            estimate(SensoryModality.PROPRIOCEPTION, 0.12, timestamp=10),
        ),
        timestamp=10,
        target_modality=SensoryModality.VISION,
    )
    assert synchronized.common_cause_probability > asynchronous.common_cause_probability
    assert asynchronous.value == pytest.approx(0.1, abs=0.02)


def test_distinct_source_ids_are_segregated_even_when_values_match() -> None:
    result = MultisensoryIntegrator().integrate(
        (
            estimate(SensoryModality.VISION, 0.3, source_id="object-a"),
            estimate(SensoryModality.PROPRIOCEPTION, 0.31, source_id="object-b"),
        ),
        timestamp=10,
        target_modality=SensoryModality.VISION,
    )
    assert result.common_cause_probability == 0.0
    assert result.value == pytest.approx(0.3)


def test_missing_modality_is_explicit_and_single_cue_is_not_fabricated() -> None:
    result = MultisensoryIntegrator().integrate(
        (estimate(SensoryModality.VISION, 0.4),),
        timestamp=10,
        target_modality=SensoryModality.VISION,
        expected_modalities=(SensoryModality.VISION, SensoryModality.PROPRIOCEPTION),
    )
    assert result.missing_modalities == (SensoryModality.PROPRIOCEPTION,)
    assert result.common_cause_probability == 0.0
    assert result.fused_value is None
    assert result.value == pytest.approx(0.4)


def test_integrated_state_projects_into_lif_neural_substrate() -> None:
    result = MultisensoryIntegrator().integrate(
        (
            estimate(SensoryModality.VISION, 0.2),
            estimate(SensoryModality.PROPRIOCEPTION, 0.25),
        ),
        timestamp=10,
        target_modality=SensoryModality.VISION,
    )
    neuron_ids = tuple(f"multisensory-{index}" for index in range(27))
    simulation = NeuralSimulation(max_neurons=27, max_synapses=1, max_pending_events=1)
    for identifier in neuron_ids:
        simulation.add_neuron(LIFNeuron(identifier))
    step = project_integrated_state(result, simulation, neuron_ids)
    assert step.step_index == 1
    assert step.spikes
    assert set(step.spikes).issubset(neuron_ids)


def test_comparison_is_deterministic_and_exercises_missing_and_conflicting_inputs() -> None:
    first = run_multisensory_comparison(100)
    second = run_multisensory_comparison(100)
    assert first == second
    assert first.trials == 100
    assert first.missing_modality_trials > 0
    assert first.conflicting_modality_trials > 0
    assert first.reliability_fusion_mae > first.independent_mae
    assert first.causal_integration_mae < first.independent_mae
    assert first.causal_integration_mae < first.reliability_fusion_mae


def test_invalid_duplicate_future_and_over_capacity_evidence_is_rejected() -> None:
    integrator = MultisensoryIntegrator()
    vision = estimate(SensoryModality.VISION, 0.1)
    with pytest.raises(ValueError, match="one estimate per modality"):
        integrator.integrate(
            (vision, vision),
            timestamp=10,
            target_modality=SensoryModality.VISION,
        )
    with pytest.raises(ValueError, match="future-dated"):
        integrator.integrate(
            (estimate(SensoryModality.VISION, 0.1, timestamp=11),),
            timestamp=10,
            target_modality=SensoryModality.VISION,
        )
    too_many = tuple(
        estimate(modality, 0.1)
        for modality in SensoryModality
    )
    with pytest.raises(ValueError, match="capacity"):
        integrator.integrate(
            too_many + (vision,),
            timestamp=10,
            target_modality=SensoryModality.VISION,
        )


def test_estimate_conversion_rejects_invalid_variance_mapping_and_feature_index() -> None:
    code = PopulationEncoder().encode((0.1, 0.2), timestamp=1)
    percept = EncodedPercept(
        SensoryModality.VISION, code, 1, code_scheme_provenance(), 0, "test"
    )
    with pytest.raises(ValueError, match="feature_index"):
        estimate_from_percept(percept, source_id="source", feature_index=2)
    with pytest.raises(ValueError, match="variance mapping"):
        estimate_from_percept(percept, source_id="source", variance_floor=0.0)


def code_scheme_provenance():
    from artificial_human.neural.sensory import SensoryProvenance

    return SensoryProvenance.SYNTHETIC
