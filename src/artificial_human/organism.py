from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from artificial_human import __version__
from artificial_human.brain.planner import Planner
from artificial_human.brain.providers.base import ChatProvider
from artificial_human.brain.providers.openai_compatible import OpenAICompatibleProvider
from artificial_human.brain.retrieval import MemoryRetriever
from artificial_human.brain.service import BrainService
from artificial_human.circulation.scheduler import WorkScheduler
from artificial_human.core.events import EventBus
from artificial_human.identity.repository import IdentityRepository
from artificial_human.immune.engine import ImmuneSystem
from artificial_human.learning import LearningSystem
from artificial_human.memory.repository import MemoryRepository
from artificial_human.metabolism.governor import ResourceGovernor
from artificial_human.regeneration.engine import RepairEngine
from artificial_human.sensory.manual import ManualSensor
from artificial_human.sleep.consolidation import SleepSystem


class ArtificialOrganism:
    """Composition root for the local artificial-organism runtime."""

    def __init__(
        self,
        *,
        owner_id: str | None = None,
        database_path: str | Path | None = None,
        identity_path: str | Path | None = None,
        brain_provider: ChatProvider | None = None,
    ) -> None:
        self.owner_id: str = (\n            owner_id or os.getenv("ARTIFICIAL_HUMAN_OWNER_ID") or "local-owner"\n        )
        self.identity_repository = IdentityRepository(self.owner_id, identity_path)
        self.identity = self.identity_repository.load()
        self.memory = MemoryRepository(database_path)
        self.learning = LearningSystem(self.memory)
        self.brain = Planner()
        self.chat_provider = brain_provider or OpenAICompatibleProvider()
        self.cognitive_brain = BrainService(
            self.chat_provider,
            self.learning,
            MemoryRetriever(),
        )
        self.senses = ManualSensor()
        self.immune = ImmuneSystem()
        self.regeneration = RepairEngine()
        self.sleep = SleepSystem()
        self.circulation = WorkScheduler()
        self.metabolism = ResourceGovernor()
        self.nervous_system = EventBus()

    def reasoning_status(self) -> dict[str, object]:
        if isinstance(self.chat_provider, OpenAICompatibleProvider):
            return self.chat_provider.status()
        return {
            "provider": "custom",
            "configured": True,
            "authentication_configured": None,
        }

    def status(self) -> dict[str, Any]:
        reasoning = self.reasoning_status()
        return {
            "name": "Artificial Human",
            "version": __version__,
            "owner_id": self.owner_id,
            "state": "operational",
            "systems": {
                "brain": "ready" if reasoning["configured"] else "provider_not_configured",
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
            "reasoning_provider": reasoning,
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
            if signal is None:
                raise RuntimeError("Immune system returned an incident without its signal.")
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
                    "dendritic_classification": self.immune.dendritic.classify(
                        signal, incident.severity
                    ),
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
