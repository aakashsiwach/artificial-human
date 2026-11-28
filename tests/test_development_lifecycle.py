from __future__ import annotations

from artificial_human.development.lifecycle import (
    DevelopmentStage,
    DevelopmentalModality,
    DevelopmentalNeuralSubstrate,
    run_developmental_learning_comparison,
    stage_for_day,
)
from artificial_human.development.prenatal import PrenatalBrain


def test_model_stage_boundaries_are_explicit_and_not_person_age_claims() -> None:
    assert stage_for_day(0) is DevelopmentStage.PRENATAL
    assert stage_for_day(280) is DevelopmentStage.PRENATAL
    assert stage_for_day(281) is DevelopmentStage.EARLY_POSTNATAL
    assert stage_for_day(731) is DevelopmentStage.CHILDHOOD
    assert stage_for_day(3_651) is DevelopmentStage.ADOLESCENCE
    assert stage_for_day(6_571) is DevelopmentStage.MATURE


def test_prenatal_scaffold_migrates_into_the_same_lif_population_and_synapse_state() -> None:
    prenatal = PrenatalBrain(seed=31)
    prenatal.advance(140)
    source = prenatal.snapshot()

    substrate = DevelopmentalNeuralSubstrate.from_prenatal(prenatal)
    target = substrate.snapshot()

    assert target.developmental_day == source.age_days
    assert dict(target.population_maturity) == {
        item.identifier: item.maturity for item in source.populations
    }
    assert target.network["neuron_count"] == source.population_count
    assert target.network["synapse_count"] == source.projection_count
    for projection in source.projections:
        identifier = substrate._projection_ids[(projection.source, projection.target)]
        assert substrate.simulation.synapses[identifier].weight == (
            120.0 + 120.0 * projection.strength
        )


def test_postnatal_maturation_adds_connections_without_replacing_the_substrate() -> None:
    substrate = DevelopmentalNeuralSubstrate(seed=7)
    network = substrate.simulation
    initial = substrate.snapshot()
    substrate.advance(1_200)
    later = substrate.snapshot()

    assert substrate.simulation is network
    assert later.stage == DevelopmentStage.CHILDHOOD.value
    assert later.developmental_day == 1_200
    assert len(later.staged_synapses_created) == 4
    assert later.network["neuron_count"] == initial.network["neuron_count"] == 7
    assert later.network["synapse_count"] == initial.network["synapse_count"] + 4
    assert all(
        dict(later.population_maturity)[name] >= value
        for name, value in initial.population_maturity
    )


def test_sensory_capability_is_gated_by_model_development() -> None:
    substrate = DevelopmentalNeuralSubstrate()
    early = substrate.expose(DevelopmentalModality.AUDITORY)
    assert not early.available
    assert early.reason == "modality_not_yet_available_in_model"

    substrate.advance(1_200)
    auditory = substrate.expose(DevelopmentalModality.AUDITORY)
    vision = substrate.expose(DevelopmentalModality.VISION)
    interoception = substrate.expose(DevelopmentalModality.INTEROCEPTION)
    assert auditory.available
    assert vision.available
    assert interoception.available
    assert substrate.snapshot().accepted_exposures == 3


def test_experience_changes_existing_pathway_efficacy_without_retaining_stimuli() -> None:
    substrate = DevelopmentalNeuralSubstrate(seed=13)
    substrate.advance(1_200)
    route = ("sensory_input", "recurrent_integration", "association", "sequence_context")
    before = substrate._route_weights(route)
    outcomes = [
        substrate.expose(DevelopmentalModality.AUDITORY, intensity=0.8)
        for _ in range(12)
    ]
    after = substrate._route_weights(route)

    assert all(item.available for item in outcomes)
    assert substrate.snapshot().accepted_exposures == 12
    assert after != before
    assert all(0.0 <= item.weight <= 250.0 for item in substrate.simulation.synapses.values())
    assert not hasattr(substrate, "stimulus_history")
    assert "content" not in str(substrate.snapshot().as_dict()).lower()


def test_unused_staged_connections_are_silenced_but_base_scaffold_is_preserved() -> None:
    substrate = DevelopmentalNeuralSubstrate(seed=5)
    substrate.advance(6_571)
    snapshot = substrate.snapshot()

    assert snapshot.stage == DevelopmentStage.MATURE.value
    assert len(snapshot.staged_synapses_created) == 6
    assert len(snapshot.functionally_pruned_synapses) > 0
    assert all(
        substrate.simulation.synapses[item].weight == 0.0
        for item in snapshot.functionally_pruned_synapses
    )
    assert all(
        substrate.simulation.synapses[item].weight > 0.0
        for item in substrate._projection_ids.values()
    )


def test_developmental_comparison_is_reproducible_and_reports_frozen_baseline() -> None:
    first = run_developmental_learning_comparison(seed=17, training_exposures=12)
    second = run_developmental_learning_comparison(seed=17, training_exposures=12)

    assert first == second
    assert first["training_exposures"] == 12
    assert first["path_efficacy_improvement"] is True
    assert "behavioural_improvement" in first
    assert isinstance(first["baseline_target_spikes"], int)
    assert isinstance(first["trained_target_spikes"], int)


def test_reset_returns_the_unified_substrate_to_prenatal_source_defaults() -> None:
    substrate = DevelopmentalNeuralSubstrate(seed=23)
    substrate.advance(1_200)
    substrate.expose(DevelopmentalModality.AUDITORY)
    substrate.reset()

    snapshot = substrate.snapshot()
    assert snapshot.developmental_day == 0
    assert snapshot.stage == DevelopmentStage.PRENATAL.value
    assert snapshot.accepted_exposures == 0
    assert snapshot.network["synapse_count"] == 10
    assert snapshot.network["step_index"] == 0
