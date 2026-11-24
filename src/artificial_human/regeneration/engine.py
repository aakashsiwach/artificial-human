from __future__ import annotations

from dataclasses import replace

from artificial_human.core.models import Incident, RepairProposal


class RepairEngine:
    """Creates and tracks proposals; it never executes commands or modifies source code."""

    def __init__(self) -> None:
        self._proposals: dict[str, RepairProposal] = {}

    def propose(self, incident: Incident) -> RepairProposal:
        proposal = RepairProposal(
            incident_id=incident.id,
            summary=f"Investigate and address: {incident.summary}",
            steps=(
                "Reproduce the issue in an isolated environment.",
                "Identify the smallest responsible component and root cause.",
                "Create a proposed change with a regression test.",
                "Run the relevant test suite and security checks.",
                "Present evidence and a diff for human review.",
            ),
            safety_checks=(
                "No production mutation.",
                "No unrestricted shell execution.",
                "No secrets copied into the repair workspace.",
                "No merge or deployment without explicit approval.",
            ),
        )
        self._proposals[proposal.id] = proposal
        return proposal

    def approve(self, proposal_id: str, *, reviewer: str) -> RepairProposal:
        if not reviewer.strip():
            raise ValueError("A reviewer identity is required")
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(proposal_id)
        if proposal.status != "pending_review":
            raise ValueError(f"Cannot approve proposal in state: {proposal.status}")
        updated = replace(proposal, status=f"approved_by:{reviewer}")
        self._proposals[proposal_id] = updated
        return updated

    def list(self) -> tuple[RepairProposal, ...]:
        return tuple(self._proposals.values())
