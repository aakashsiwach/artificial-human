from __future__ import annotations

import pytest

from artificial_human.development.prenatal import phase_for_week
from artificial_human.development.program import (
    DevelopmentalModality,
    DevelopmentalProgram,
    DevelopmentalStage,
    run_developmental_comparison,
)


def test_developmental_clock_is_separate_from_neural_simulation_clock() -> None:
    program = DevelopmentalProgram()
    program.advance_prenatal(280)
    program.advance_postnatal(500)

    snapshot = program.snapshot()
    assert snapshot.prenatal_days == 280
    assert snapshot.postnatal_units == 500
    assert snapshot.simulation_step == 0
    assert snapshot.stage is DevelopmentalStage.EXPERIENCE_REFINEMENT
    assert snapshot.maturity == pytest.approx(0.475)


def test_prenatal_phase_and_modality_onsets_match_reference_schedule() -> None:
    program = DevelopmentalProgram()
    previous = 0
    for day in (0, 35, 56, 84, 168, 280):
        program.advance_prenatal(day - previous)
        previous = day
        assert program.snapshot().prenatal_phase is phase_for_week(day / 7.0)

    assert DevelopmentalModality.TACTILE in program.available_modalities
    assert DevelopmentalModality.AUDITORY in program.available_modalities


def test_postnatal_maturation_adds_connections_and_reduces_conduction_delay() -> None:
    program = DevelopmentalProgram()
    initial_count = program.snapshot().synapse_count
    initial_delay = program.simulation.synapses["tactile_to_integration"].delay_steps
    initial_tau = program.simulation.neurons["integration"].parameters.membrane_time_constant_ms

    program.advance_prenatal(280)
    birth_count = program.snapshot().synapse_count
    program.advance_postnatal(2000)
    mature = program.snapshot()

    assert birth_count > initial_count
    assert mature.synapse_count > birth_count
    assert mature.stage is DevelopmentalStage.STABILIZED
    assert mature.myelination_factor == 1.0
    assert program.simulation.synapses["tactile_to_integration"].delay_steps < initial_delay
    assert program.simulation.neurons["integration"].parameters.membrane_time_constant_ms < initial_tau


def test_sensory_capabilities_are_gated_by_developmental_stage() -> None:
    program = DevelopmentalProgram()
    before = program.experience(DevelopmentalModality.AUDITORY)
    assert not before.accepted
    assert program.simulation.step_index == 0

    program.advance_prenatal(168)
    auditory = program.experience(DevelopmentalModality.AUDITORY)
    assert auditory.accepted
    assert auditory.simulation_steps == 1

    program.advance_prenatal(112)
    program.advance_postnatal(1)
    vision = program.experience(DevelopmentalModality.VISION)
    assert vision.accepted


def test_experience_updates_live_synapses_and_is_bounded() -> None:
    program = DevelopmentalProgram()
    program.advance_prenatal(280)
    program.advance_postnatal(500)
    initial = program.simulation.synapses["association_to_prediction"].weight

    for _ in range(20):
        program.experience(DevelopmentalModality.TACTILE, intensity=0.95)
        program.advance_simulation(2)
        program.experience(DevelopmentalModality.AUDITORY, intensity=0.95)
        program.advance_simulation(3)

    assert program.simulation.synapses["association_to_prediction"].weight > initial
    assert program.experience_count == 40
    assert program.snapshot().total_spikes > 0


def test_functional_pruning_only_runs_in_late_maturation() -> None:
    program = DevelopmentalProgram()
    with pytest.raises(ValueError, match="late maturation"):
        program.prune_unused()

    program.advance_prenatal(280)
    program.advance_postnatal(1000)
    pruned = program.prune_unused(minimum_usage=1)

    assert "visual_to_integration" in pruned
    assert program.simulation.synapses["visual_to_integration"].weight == 0.0
    assert "visual_to_integration" in program.snapshot().pruned_synapses


def test_developmental_learning_beats_frozen_and_mature_from_start_baselines() -> None:
    result = run_developmental_comparison(training_trials=60)

    assert result.success
    assert result.developmental_prediction_spikes > result.frozen_prediction_spikes
    assert result.developmental_prediction_spikes > result.mature_from_start_prediction_spikes
    assert (
        result.developmental_association_prediction_weight
        > result.frozen_association_prediction_weight
    )


def test_developmental_clock_and_experience_arguments_are_validated() -> None:
    program = DevelopmentalProgram()
    with pytest.raises(ValueError, match="birth boundary"):
        program.advance_postnatal(1)
    with pytest.raises(ValueError, match="280"):
        program.advance_prenatal(281)
    with pytest.raises(ValueError, match="intensity"):
        program.experience(DevelopmentalModality.TACTILE, intensity=1.5)
