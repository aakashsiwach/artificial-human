from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
from uuid import uuid4

from artificial_human.core.models import HealthSignal, Incident, Observation, RepairProposal, Severity


class DendriticCell:
    """Classifies a signal and presents an immune alert to the organism."""

    def classify(self, signal: HealthSignal) -> str:
        if not signal.abnormal:
            return "normal"
        return "critical_alert" if signal.threshold > 0 and signal.direction == "below" else "alert"


@dataclass(frozen=True, slots=True)
class ContainmentPlan:
    incident_id: str
    actions: tuple[str, ...]
    requires_operator_approval: bool = True
    auto_executable: bool = False
    id: str = field(default_factory=lambda: str(uuid4()))


class NeutrophilCell:
    """Produces rapid containment recommendations without applying them."""

    def recommend_containment(self, incident: Incident) -> ContainmentPlan:
        actions = [
            "Pause non-essential background work.",
            "Preserve logs and diagnostic evidence.",
            "Notify the owner or designated operator.",
        ]
        if incident.severity == Severity.CRITICAL:
            actions.insert(0, "Escalate the incident as critical.")
        return ContainmentPlan(incident_id=incident.id, actions=tuple(actions))


@dataclass(frozen=True, slots=True)
class InvestigationReport:
    incident_id: str
    observation_ids: tuple[str, ...]
    evidence_sources: tuple[str, ...]
    next_steps: tuple[str, ...]


class MacrophageCell:
    """Correlates explicitly supplied observations into an auditable evidence bundle."""

    def investigate(
        self, incident: Incident, observations: Sequence[Observation]
    ) -> InvestigationReport:
        ordered = sorted(observations, key=lambda item: item.observed_at)
        return InvestigationReport(
            incident_id=incident.id,
            observation_ids=tuple(item.id for item in ordered),
            evidence_sources=tuple(sorted({item.source for item in ordered})),
            next_steps=(
                "Review evidence timestamps and source trust.",
                "Compare the incident window with recent changes.",
                "Record a root-cause hypothesis with confidence and supporting evidence.",
            ),
        )


@dataclass(frozen=True, slots=True)
class KnownRepair:
    pattern_key: str
    remediation: str
    reviewer: str
    verified: bool = True


class BCellMemory:
    """Retains only human-reviewed, verified repair patterns."""

    def __init__(self) -> None:
        self._patterns: dict[str, KnownRepair] = {}

    def learn(
        self, pattern_key: str, remediation: str, *, reviewer: str, verified: bool
    ) -> KnownRepair:
        if not verified:
            raise ValueError("Only verified repair outcomes can be learned")
        if not pattern_key.strip() or not remediation.strip() or not reviewer.strip():
            raise ValueError("pattern_key, remediation, and reviewer are required")
        record = KnownRepair(pattern_key, remediation, reviewer)
        self._patterns[pattern_key] = record
        return record

    def recall(self, pattern_key: str) -> KnownRepair | None:
        return self._patterns.get(pattern_key)


@dataclass(frozen=True, slots=True)
class RepairEvidence:
    sandboxed: bool
    tests_passed: bool
    regression_test_added: bool
    diff_digest: str


@dataclass(frozen=True, slots=True)
class TCellVerdict:
    ready_for_human_review: bool
    failed_checks: tuple[str, ...]


class TCellVerifier:
    """Challenges repair proposals against evidence before they reach human review."""

    def verify(self, proposal: RepairProposal, evidence: RepairEvidence) -> TCellVerdict:
        failures: list[str] = []
        if proposal.status != "pending_review":
            failures.append("Proposal must be pending review.")
        if not evidence.sandboxed:
            failures.append("Repair was not prepared in an isolated sandbox.")
        if not evidence.tests_passed:
            failures.append("Required tests have not passed.")
        if not evidence.regression_test_added:
            failures.append("No regression test was added.")
        if not evidence.diff_digest.strip():
            failures.append("An immutable diff digest is required.")
        return TCellVerdict(
            ready_for_human_review=not failures,
            failed_checks=tuple(failures),
        )
