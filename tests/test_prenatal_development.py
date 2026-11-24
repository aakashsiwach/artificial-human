from __future__ import annotations

import json

import pytest

from artificial_human.cli import main
from artificial_human.development.prenatal import (
    GESTATION_DAYS,
    MODALITY_PATHS,
    MILESTONES,
    PrenatalBrain,
    PrenatalModality,
    PrenatalPhase,
    milestones_reached,
    phase_for_week,
)


def _path_strength(brain: PrenatalBrain, modality: PrenatalModality) -> float:
    snapshot = brain.snapshot()
    strengths = {(item.source, item.target): item.strength for item in snapshot.projections}
    path = MODALITY_PATHS[modality]
    values = [strengths[edge] for edge in zip(path, path[1:], strict=False)]
    return sum(values) / len(values)


def test_new_brain_starts_with_empty_experience_and_fixed_scaffold() -> None:
    brain = PrenatalBrain(seed=11)
    snapshot = brain.snapshot()
    assert snapshot.age_days == 0
    assert snapshot.gestational_week == 0
    assert snapshot.accepted_exposures == 0
    assert snapshot.total_plasticity_updates == 0
    assert snapshot.population_count == 7
    assert snapshot.projection_count == 10
    assert brain.structure_signature()[0]
    assert len(brain.structure_signature()[1]) == 10


def test_developmental_phase_and_milestone_timeline() -> None:
    assert phase_for_week(0) is PrenatalPhase.EARLY_EMBRYONIC
    assert phase_for_week(6) is PrenatalPhase.NEURULATION
    assert phase_for_week(9) is PrenatalPhase.REGIONAL_PATTERNING
    assert phase_for_week(16) is PrenatalPhase.PROLIFERATION_AND_MIGRATION
    assert phase_for_week(24) is PrenatalPhase.EARLY_CIRCUIT_ASSEMBLY
    assert phase_for_week(31) is PrenatalPhase.SENSORY_REFINEMENT
    assert phase_for_week(38) is PrenatalPhase.LATE_GESTATION
    assert [item.week for item in milestones_reached(24)] == [
        item.week for item in MILESTONES if item.week <= 24
    ]
    with pytest.raises(ValueError):
        phase_for_week(41)


def test_advance_matures_populations_and_preserves_structure() -> None:
    brain = PrenatalBrain(seed=3)
    initial_signature = brain.structure_signature()
    initial = brain.snapshot()
    brain.advance(140)
    middle = brain.snapshot()
    brain.advance(140)
    term = brain.snapshot()
    assert initial.populations[0].maturity == 0.0
    assert middle.populations[0].maturity > initial.populations[0].maturity
    assert term.age_days == GESTATION_DAYS
    assert term.phase == PrenatalPhase.LATE_GESTATION.value
    assert brain.structure_signature() == initial_signature
    assert term.total_plasticity_updates > 0


def test_early_auditory_exposure_is_not_processed() -> None:
    brain = PrenatalBrain()
    brain.advance(20 * 7)
    before = _path_strength(brain, PrenatalModality.AUDITORY)
    outcome = brain.expose(PrenatalModality.AUDITORY, intensity=0.8)
    after = _path_strength(brain, PrenatalModality.AUDITORY)
    assert outcome.available is False
    assert outcome.reason == "modality_not_yet_available_in_this_model"
    assert after == before
    assert brain.snapshot().accepted_exposures == 0


def test_repeated_experience_changes_connection_strength_not_topology() -> None:
    brain = PrenatalBrain(seed=5)
    brain.advance(28 * 7)
    initial_signature = brain.structure_signature()
    before = _path_strength(brain, PrenatalModality.AUDITORY)
    outcomes = [
        brain.expose("auditory", intensity=0.9, duration_seconds=3.0)
        for _ in range(20)
    ]
    after = _path_strength(brain, PrenatalModality.AUDITORY)
    assert all(item.available for item in outcomes)
    assert after > before
    assert all(0.0 <= item.strength <= 1.0 for item in brain.snapshot().projections)
    assert brain.structure_signature() == initial_signature
    assert brain.snapshot().accepted_exposures == 20


def test_tactile_and_vestibular_have_separate_developmental_routes() -> None:
    brain = PrenatalBrain(seed=2)
    brain.advance(18 * 7)
    tactile = brain.expose("tactile", intensity=0.6)
    vestibular = brain.expose("vestibular", intensity=0.6)
    auditory = brain.expose("auditory", intensity=0.6)
    assert tactile.available
    assert vestibular.available
    assert not auditory.available


def test_exposure_content_is_not_retained() -> None:
    brain = PrenatalBrain()
    brain.advance(30 * 7)
    brain.expose("auditory", intensity=0.7, duration_seconds=2.0)
    snapshot = brain.snapshot().as_dict()
    serialised = json.dumps(snapshot)
    assert "stimulus" not in serialised
    assert "melody" not in serialised
    assert "content" not in serialised
    assert not hasattr(brain, "exposure_history")
    assert not hasattr(brain, "stimuli")


def test_fixed_seed_is_reproducible() -> None:
    first = PrenatalBrain(seed=41)
    second = PrenatalBrain(seed=41)
    first.advance(175)
    second.advance(175)
    first.expose("auditory", intensity=0.75)
    second.expose("auditory", intensity=0.75)
    assert first.snapshot().as_dict() == second.snapshot().as_dict()


@pytest.mark.parametrize(
    ("modality", "intensity", "duration"),
    [
        ("unsupported", 0.5, 1.0),
        ("auditory", -0.1, 1.0),
        ("auditory", 1.1, 1.0),
        ("auditory", 0.5, 0.0),
        ("auditory", 0.5, 61.0),
    ],
)
def test_invalid_exposure_inputs_are_rejected(
    modality: str, intensity: float, duration: float
) -> None:
    brain = PrenatalBrain()
    with pytest.raises(ValueError):
        brain.expose(modality, intensity=intensity, duration_seconds=duration)


def test_advance_cannot_regress_or_exceed_term() -> None:
    brain = PrenatalBrain()
    with pytest.raises(ValueError):
        brain.advance(-1)
    with pytest.raises(ValueError):
        brain.advance(GESTATION_DAYS + 1)
    brain.advance(GESTATION_DAYS)
    with pytest.raises(ValueError):
        brain.advance(1)


def test_cli_prenatal_simulation_and_exposure(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([
        "prenatal", "--days", "280", "--seed", "7",
        "--exposure", "25:auditory:0.7",
        "--exposure", "30:tactile:0.4",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["snapshot"]["age_days"] == GESTATION_DAYS
    assert result["snapshot"]["accepted_exposures"] == 2
    assert len(result["exposures"]) == 2


def test_cli_rejects_exposure_after_selected_stop_day(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["prenatal", "--days", "100", "--exposure", "25:auditory:0.5"]) == 2
    assert "before --days" in capsys.readouterr().err
