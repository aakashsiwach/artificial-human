from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from artificial_human import __version__
from artificial_human.brain.planner import Planner
from artificial_human.circulation.scheduler import WorkScheduler
from artificial_human.core.events import EventBus
from artificial_human.identity.profile import IdentityProfile
from artificial_human.learning import LearningSystem
from artificial_human.immune.engine import ImmuneSystem
from artificial_human.memory.repository import MemoryRepository
from artificial_human.metabolism.governor import ResourceGovernor
from artificial_human.regeneration.engine import RepairEngine
from artificial_human.sensory.manual import ManualSensor
from artificial_human.sleep.consolidation import SleepSystem


class ArtificialOrganism:
    """Composition root for the first local artificial-organism runtime."""

    def __init__(
        self, *, owner_id: str | None = None, database_path: str | Path | None = None
    ) -> None:
        self.owner_id = owner_id or os.getenv("ARTIFICIAL_HUMAN_OWNER_ID", "local-owner")
        self.identity = IdentityProfile(owner_id=self.owner_id)
        self.memory = MemoryRepository(database_path)
        self.learning = LearningSystem(self.memory)
        self.brain = Planner()
        self.senses = ManualSensor()
        self.immune = ImmuneSystem()
        self.regeneration = RepairEngine()
        self.sleep = SleepSystem()
        self.circulation = WorkScheduler()
        self.metabolism = ResourceGovernor()
        self.nervous_system = EventBus()

    def status(self) -> dict[str, Any]:
        return {
            "name": "Artificial Human",
            "version": __version__,
            "owner_id": self.owner_id,
            "state": "operational",
            "systems": {
                "brain": "ready",
                "identity": "ready",
                "memory": "ready_consent_required",
                "sensory": "manual_input_only",
                "immune": "ready",
                "regeneration": "proposal_only",
                "sleep": "review_only",
                "circulation": "ready",
                "metabolism": "bounded_declared_work",
                "nervous_system": "in_process",
            },
            "immune_cells": [
                "dendritic", "neutrophil", "macrophage", "b_cell_memory", "t_cell_verifier"
            ],
            "work_queue_depth": self.circulation.depth(),
            "open_incidents": len(self.immune.incidents()),
            "repair_proposals": len(self.regeneration.list()),
        }

    def evaluate_health(self, name: str, value: float, source: str = "manual") -> dict[str, Any]:
        signal, incident = self.immune.evaluate(name, value, source=source)
        if incident is not None:
            containment = self.immune.neutrophil.recommend_containment(incident)
            self.nervous_system.publish("immune.incident_detected", incident)
            proposal = self.regeneration.propose(incident)
            self.nervous_system.publish("regeneration.proposal_created", proposal)
            return {
                "signal": {"name": signal.name, "value": signal.value, "abnormal": True},
                "incident": {
                    "id": incident.id,
                    "severity": incident.severity.value,
                    "summary": incident.summary,
                },
                "immune_response": {
                    "dendritic_classification": self.immune.dendritic.classify(signal),
                    "neutrophil_plan": {
                        "actions": containment.actions,
                        "requires_operator_approval": containment.requires_operator_approval,
                        "auto_executable": containment.auto_executable,
                    },
                },
                "repair_proposal": {
                    "id": proposal.id,
                    "status": proposal.status,
                    "summary": proposal.summary,
                },
            }
        return {
            "signal": (
                {"name": signal.name, "value": signal.value, "abnormal": False}
                if signal
                else {"name": name, "value": value, "configured": False}
            ),
            "incident": None,
            "repair_proposal": None,
        }
