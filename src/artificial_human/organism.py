from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from artificial_human import __version__
from artificial_human.brain.planner import Planner
from artificial_human.brain.providers.base import ChatProvider
from artificial_human.brain.providers.openai_compatible import OpenAICompatibleProvider
from artificial_human.brain.retrieval import MemoryRetriever
from artificial_human.brain.service import BrainService
from artificial_human.circulation.scheduler import WorkScheduler
from artificial_human.cognition.architecture import CognitiveArchitecture
from artificial_human.core.events import EventBus
from artificial_human.identity.constitution import core_identity_profile
from artificial_human.identity.repository import IdentityRepository
from artificial_human.immune.engine import ImmuneSystem
from artificial_human.learning import LearningSystem
from artificial_human.memory.embeddings import EmbeddingProvider, embedding_provider_from_env
from artificial_human.memory.repository import MemoryRepository
from artificial_human.memory.search import SearchConfig
from artificial_human.memory.system import LongTermMemory
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
        embedding_provider: EmbeddingProvider | Literal["env"] | None = "env",
        search_config: SearchConfig | None = None,
    ) -> None:
        self.owner_id: str = (
            owner_id or os.getenv("ARTIFICIAL_HUMAN_OWNER_ID") or "local-owner"
        )
        public_deployment = os.getenv("ARTIFICIAL_HUMAN_PUBLIC_DEPLOYMENT", "").lower() in {
            "1", "true", "yes",
        }
        requested_storage = os.getenv("ARTIFICIAL_HUMAN_STORAGE_MODE", "").strip().lower()
        self.volatile_runtime = public_deployment or requested_storage == "volatile"
        if self.volatile_runtime:
            # No file-backed identity, SQLite database, or external vector service
            # is used in the public runtime. Foundational identity lives in source.
            self.identity_repository: IdentityRepository | None = None
            self.identity = core_identity_profile(self.owner_id)
            self.memory = MemoryRepository(":memory:")
            self.embedding_provider = None
        else:
            self.identity_repository = IdentityRepository(self.owner_id, identity_path)
            self.identity = self.identity_repository.load()
            self.memory = MemoryRepository(database_path)
            # Semantic indexing is off unless explicitly configured; see docs/MEMORY.md.
            self.embedding_provider = (
                embedding_provider_from_env() if embedding_provider == "env" else embedding_provider
            )
        self.learning = LearningSystem(self.memory)
        self.long_term_memory = LongTermMemory(
            self.memory, self.embedding_provider, search_config or SearchConfig.from_env()
        )
        self.brain = Planner()
        self.chat_provider = brain_provider or OpenAICompatibleProvider()
        retriever = MemoryRetriever()
        self.cognitive_brain = BrainService(
            self.chat_provider,
            self.learning,
            retriever,
            architecture=CognitiveArchitecture(
                retriever=retriever,
                memory_search=self.long_term_memory.search_engine,
                learning=self.learning,
            ),
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
                "brain": "cognitive_cycle_ready" if reasoning["configured"] else "cognitive_cycle_ready_provider_not_configured",
                "identity": "ready",
                "memory": (
                    "volatile_in_process_no_external_database"
                    if self.volatile_runtime
                    else "ready_consent_required"
                ),
                "storage": (
                    "volatile_process_memory_restart_resets_state"
                    if self.volatile_runtime
                    else "local_persistent_storage"
                ),
                "memory_retrieval": (
                    "hybrid_lexical_semantic_temporal"
                    if self.embedding_provider is not None
                    else "lexical_temporal_semantic_not_configured"
                ),
                "sensory": "manual_input_only",
                "immune": "ready",
                "regeneration": "proposal_only",
                "sleep": "review_only",
                "circulation": "ready",
                "metabolism": "bounded_declared_work",
                "nervous_system": "in_process",
            },
            "reasoning_provider": reasoning,
            "runtime": {
                "storage_mode": "volatile_in_process" if self.volatile_runtime else "local_persistent",
                "external_database_connected": False,
                "personal_experience_persists_restart": not self.volatile_runtime,
                "identity_source": "version_controlled_source_code" if self.volatile_runtime else "local_owner_profile",
            },
            "cognitive_systems": self.cognitive_brain.architecture.status(),
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
