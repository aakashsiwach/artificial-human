from __future__ import annotations

from dataclasses import fields

import pytest

from artificial_human.neural.population import (
    FeatureVectorBaselineEncoder,
    OneHotBaselineEncoder,
    PopulationAssociativeMemory,
    PopulationEncoder,
    PopulationProjector,
    run_population_coding_comparison,
)
from artificial_human.neural.simulation import LIFNeuron, NeuralSimulation


def test_gaussian_population_code_is_distributed_decodable_and_generalizes_smoothly() -> None:
    encoder = PopulationEncoder(units_per_feature=9)
    near = encoder.encode((0.0, 0.5), timestamp=7, confidence=0.8)
    nearby = encoder.encode((0.2, 0.5), timestamp=8, confidence=0.8)
    distant = encoder.encode((0.8, 0.5), timestamp=9, confidence=0.8)

    assert len(near.activations) == 18
    assert all(0.0 <= value <= 1.0 for value in near.activations)
    assert near.cosine_similarity(nearby) > near.cosine_similarity(distant)
    decoded = encoder.decode(near)
    assert decoded[0] == pytest.approx(0.0, abs=1e-8)
    assert decoded[1] == pytest.approx(0.456, abs=0.01)
    assert near.timestamp == 7
    assert near.uncertainty == pytest.approx(0.2)
    assert near.scheme == "gaussian"


def test_code_preserves_uncertainty_and_time_without_retaining_raw_features() -> None:
    source = (0.25, -0.75)
    code = PopulationEncoder().encode(source, timestamp=42, confidence=0.6)

    names = {item.name for item in fields(code)}
    assert "activations" in names
    assert "timestamp" in names
    assert "uncertainty" in names
    assert "features" not in names
    assert "raw_features" not in names
    assert code.timestamp == 42
    assert code.uncertainty == pytest.approx(0.4)
    assert not hasattr(code, "source_event")


def test_one_hot_quantization_collision_causes_interference_dense_code_disambiguates() -> None:
    dense = PopulationEncoder()
    one_hot = OneHotBaselineEncoder()
    pattern_a = dense.encode((0.01, -0.8), timestamp=1)
    pattern_b = dense.encode((0.03, 0.8), timestamp=2)
    one_hot_a = one_hot.encode((0.01, -0.8), timestamp=1)
    one_hot_b = one_hot.encode((0.03, 0.8), timestamp=2)

    assert one_hot_a.activations[:9] == one_hot_b.activations[:9]
    assert pattern_a.activations[:9] != pattern_b.activations[:9]

    dense_memory = PopulationAssociativeMemory(capacity=2)
    dense_memory.store("pattern-a", pattern_a)
    dense_memory.store("pattern-b", pattern_b)
    dense_result = dense_memory.complete(pattern_b.mask_features(1))
    assert dense_result is not None
    assert dense_result.identifier == "pattern-b"

    one_hot_memory = PopulationAssociativeMemory(capacity=2)
    one_hot_memory.store("pattern-a", one_hot_a)
    one_hot_memory.store("pattern-b", one_hot_b)
    one_hot_result = one_hot_memory.complete(one_hot_b.mask_features(1))
    assert one_hot_result is not None
    # Identical observed one-hot bins tie; stable insertion order returns the distractor.
    assert one_hot_result.identifier == "pattern-a"


def test_top_k_separation_reduces_overlap_for_documented_moderate_difference() -> None:
    encoder = PopulationEncoder()
    first = encoder.encode((0.0, 0.5))
    second = encoder.encode((0.2, 0.5))

    dense_overlap = first.cosine_similarity(second)
    sparse_overlap = first.sparsify(top_k=2).cosine_similarity(
        second.sparsify(top_k=2)
    )
    assert 0.0 < sparse_overlap < dense_overlap < 1.0


def test_associative_completion_recovers_masked_feature_from_partial_population_cue() -> None:
    encoder = PopulationEncoder()
    target = encoder.encode((0.7, -0.8), timestamp=5)
    distractor = encoder.encode((-0.7, 0.8), timestamp=6)
    cue = target.mask_features(1)

    assert cue.observed_features == (True, False)
    assert cue.decode()[1] is None

    memory = PopulationAssociativeMemory(capacity=2)
    memory.store("target", target)
    memory.store("distractor", distractor)
    result = memory.complete(cue)

    assert result is not None
    assert result.identifier == "target"
    assert result.completed_features[1] is not None
    assert result.completed_features[1] == pytest.approx(encoder.decode(target)[1])
    # Zero-fill control leaves the masked coordinate unavailable rather than completing it.
    assert cue.decode()[1] is None


def test_population_code_projects_to_structured_lif_spike_activity() -> None:
    code = PopulationEncoder(units_per_feature=9).encode((0.0,))
    neuron_ids = tuple(f"unit-{index}" for index in range(len(code.activations)))
    simulation = NeuralSimulation(
        max_neurons=len(neuron_ids),
        max_synapses=1,
        max_pending_events=1,
    )
    for identifier in neuron_ids:
        simulation.add_neuron(LIFNeuron(identifier))

    step = PopulationProjector(neuron_ids).drive(code, simulation)
    assert step.spikes
    assert set(step.spikes) <= set(neuron_ids)
    assert "unit-4" in step.spikes
    assert step.pending_events == 0


def test_hand_coded_feature_baseline_is_direct_and_same_numeric_domain() -> None:
    baseline = FeatureVectorBaselineEncoder().encode(
        (0.2, -0.5), timestamp=3, confidence=0.75
    )
    assert baseline.values == pytest.approx((0.1973753, -0.4621172))
    assert baseline.timestamp == 3
    assert baseline.uncertainty == pytest.approx(0.25)


def test_memory_bounds_duplicate_identifiers_and_empty_cues_are_explicit() -> None:
    encoder = PopulationEncoder()
    code = encoder.encode((0.1, 0.2))
    memory = PopulationAssociativeMemory(capacity=1)

    assert memory.complete(code.mask_features(0, 1)) is None
    memory.store("one", code)
    with pytest.raises(ValueError, match="duplicate"):
        memory.store("one", code)
    with pytest.raises(RuntimeError, match="capacity"):
        memory.store("two", code)


def test_encoder_and_code_reject_invalid_inputs_and_geometry() -> None:
    with pytest.raises(ValueError, match="units_per_feature"):
        PopulationEncoder(units_per_feature=1)
    with pytest.raises(ValueError, match="tuning_width"):
        PopulationEncoder(tuning_width=0.0)
    with pytest.raises(ValueError, match="features"):
        PopulationEncoder().encode((float("nan"),))
    with pytest.raises(ValueError, match="confidence"):
        PopulationEncoder().encode((0.0,), confidence=1.1)
    with pytest.raises(ValueError, match="timestamp"):
        PopulationEncoder().encode((0.0,), timestamp=-1)

    encoder = PopulationEncoder()
    code = encoder.encode((0.1, 0.2))
    with pytest.raises(ValueError, match="out of range"):
        code.mask_features(2)
    with pytest.raises(ValueError, match="unique"):
        code.mask_features(0, 0)
    with pytest.raises(ValueError, match="compatible geometry"):
        code.cosine_similarity(PopulationEncoder(units_per_feature=8).encode((0.1, 0.2)))


def test_projector_rejects_mismatched_neural_topology() -> None:
    code = PopulationEncoder(units_per_feature=3).encode((0.0,))
    simulation = NeuralSimulation()
    simulation.add_neuron(LIFNeuron("only-one"))
    with pytest.raises(ValueError, match="neuron count"):
        PopulationProjector(("only-one",)).drive(code, simulation)


def test_reproducible_comparison_reports_distinct_hypotheses_and_baselines() -> None:
    first = run_population_coding_comparison()
    second = run_population_coding_comparison()

    assert first == second
    assert first["generalization"]["near_is_more_similar"] is True
    assert first["one_hot_interference"]["same_bin_collision"] is True
    assert first["one_hot_interference"]["dense_retrieval"] == "pattern-b"
    assert first["one_hot_interference"]["one_hot_retrieval"] == "pattern-a"
    assert first["pattern_separation"]["overlap_reduced"] is True
    assert first["pattern_completion"]["retrieved_identifier"] == "target-pattern"
    assert first["pattern_completion"]["missing_feature_recovered"] is True
    assert first["lif_projection"]["spike_count"] > 0
