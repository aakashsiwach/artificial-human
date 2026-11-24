import pytest

from artificial_human.core.models import HealthSignal, Incident, RepairProposal, Severity
from artificial_human.immune.cells import (
    BCellMemory,
    DendriticCell,
    MacrophageCell,
    NeutrophilCell,
    RepairEvidence,
    TCellVerifier,
)


def make_incident() -> Incident:
    signal = HealthSignal(
        name="heartbeat_age_seconds", value=150, threshold=60,
        direction="above", source="test",
    )
    return Incident(
        signal=signal, severity=Severity.CRITICAL, summary="Heartbeat stale",
        containment_recommendation="Pause background work",
        investigation_recommendation="Inspect worker health",
    )


def test_neutrophil_recommends_containment_but_never_executes() -> None:
    plan = NeutrophilCell().recommend_containment(make_incident())
    assert plan.requires_operator_approval is True
    assert plan.auto_executable is False
    assert "Escalate the incident as critical." in plan.actions


def test_macrophage_builds_evidence_bundle_from_explicit_observations() -> None:
    from artificial_human.core.models import Observation

    observation = Observation(source="manual-note", subject="worker", value="stale")
    report = MacrophageCell().investigate(make_incident(), [observation])
    assert report.observation_ids == (observation.id,)
    assert report.evidence_sources == ("manual-note",)


def test_b_cell_learns_only_reviewed_verified_patterns() -> None:
    cells = BCellMemory()
    with pytest.raises(ValueError):
        cells.learn("timeout", "restart worker", reviewer="owner", verified=False)
    cells.learn("timeout", "scale worker pool", reviewer="owner", verified=True)
    assert cells.recall("timeout") is not None


def test_t_cell_requires_sandbox_tests_regression_and_diff() -> None:
    proposal = RepairProposal(
        incident_id="incident-1", summary="Repair timeout", steps=("test",),
        safety_checks=("review",),
    )
    verifier = TCellVerifier()
    rejected = verifier.verify(
        proposal,
        RepairEvidence(
            sandboxed=False, tests_passed=True,
            regression_test_added=False, diff_digest="",
        ),
    )
    assert rejected.ready_for_human_review is False
    assert len(rejected.failed_checks) == 3

    accepted = verifier.verify(
        proposal,
        RepairEvidence(
            sandboxed=True, tests_passed=True,
            regression_test_added=True, diff_digest="sha256:abc",
        ),
    )
    assert accepted.ready_for_human_review is True


def test_dendritic_cell_classifies_signal() -> None:
    cell = DendriticCell()
    normal = HealthSignal("api_error_rate", 0.01, 0.05)
    abnormal = HealthSignal("heartbeat_age_seconds", 150, 60)
    assert cell.classify(normal) == "normal"
    assert cell.classify(abnormal) == "alert"
