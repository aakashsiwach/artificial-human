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
from artificial_human.cognition.dynamics import RecurrentDynamicsEngine
from artificial_human.cognition.schema_extraction import SchemaExtractionSystem
from artificial_human.core.events import EventBus
from artificial_human.identity.constitution import core_identity_profile
from artificial_human.identity.repository import IdentityRepository
from artificial_human.immune.engine import ImmuneSystem
from artificial_human.learning import LearningSystem
from artificial_human.memory.embeddings import EmbeddingProvider, embedding_provider_from_env
from artificial_human.memory.cognitive import InMemoryCognitiveMemory
from artificial_human.memory.volatile_repository import InMemoryMemoryRepository
from artificial_human.memory.search import SearchConfig
from artificial_human.memory.system import LongTermMemory
from artificial_human.metabolism.governor import ResourceGovernor
from artificial_human.regeneration.engine import RepairEngine
from artificial_human.sensory.manual import ManualSensor
from artificial_human.sleep.consolidation import SleepSystem
from artificial_human.neural.organism_loop import OrganismLoop
from artificial_human.runtime import ContinuousOrganismRuntime


class ArtificialOrganism:
    """Composition root for the local artificial-organism runtime."""

    def __init__(
        self,
        *,
        owner_id: str | None = None,
        database_path: str | Path | None = None,
        identity_path: str | Path | None = None,
        brain_provider: ChatProvider | None = None,
        embedding_provider: EmbeddingProvider | Literal["env"] | None = None,
        search_config: SearchConfig | None = None,
    ) -> None:
        self.owner_id: str = (
            owner_id or os.getenv("ARTIFICIAL_HUMAN_OWNER_ID") or "local-owner"
        )
        # The active organism is always RAM-only. Legacy path arguments remain
        # accepted for compatibility but are intentionally ignored.
        self.volatile_runtime = True
        self.identity_repository: IdentityRepository | None = None
        self.identity = core_identity_profile(self.owner_id)
        self.memory = InMemoryMemoryRepository()
        self.cognitive_memory = InMemoryCognitiveMemory()
        if embedding_provider == "env":
            self.embedding_provider = embedding_provider_from_env()
        else:
            self.embedding_provider = embedding_provider
        self.learning = LearningSystem(self.memory)
        self.long_term_memory = LongTermMemory(
            self.memory, self.embedding_provider, search_config or SearchConfig.from_env()
        )
        self.brain = Planner()
        self.chat_provider = brain_provider or OpenAICompatibleProvider()
        self.neural_dynamics = RecurrentDynamicsEngine()
        # One composition-owned organism loop and its developmental LIF network.
        # API, CLI and deterministic tests all attach to this same runtime.
        self.organism_loop = OrganismLoop()
        self.neural_simulation = self.organism_loop.neural_development.simulation
        self.runtime = ContinuousOrganismRuntime(self.organism_loop)
        retriever = MemoryRetriever()
        self.cognitive_brain = BrainService(
            self.chat_provider,
            self.learning,
            retriever,
            dynamics=self.neural_dynamics,
            architecture=CognitiveArchitecture(
                retriever=retriever,
                memory_search=self.long_term_memory.search_engine,
                cognitive_memory=self.cognitive_memory,
                learning=self.learning,
            ),
            working_memory_sessions=self.cognitive_memory.working,
        )
        self.senses = ManualSensor()
        self.immune = ImmuneSystem()
        self.regeneration = RepairEngine()
        self.sleep = SleepSystem()
        self.schema_extraction = SchemaExtractionSystem()
        self.circulation = WorkScheduler()
        self.metabolism = ResourceGovernor()
        self.nervous_system = EventBus()

    def reset_volatile_state(self) -> None:
        """Clear all process-local personal memory and derived cognitive state."""
        self.memory.reset()
        self.cognitive_memory.reset()
        self.sleep.clear_owner(self.owner_id)
        self.schema_extraction.clear_owner(self.owner_id)

    def reasoning_status(self) -> dict[str, object]:
        if isinstance(self.chat_provider, OpenAICompatibleProvider):
            provider_status = self.chat_provider.status()
        else:
            provider_status = {
                "provider": "custom",
                "configured": True,
                "authentication_configured": None,
            }
        provider_status["neural_dynamics"] = self.neural_dynamics.snapshot().as_dict()
        provider_status["neural_simulation"] = self.neural_simulation.snapshot().as_dict()
        return provider_status

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
                "memory": "bounded_in_memory_consent_scoped",
                "storage": "bounded_process_memory_restart_resets_state",
                "memory_retrieval": (
                    "hybrid_lexical_semantic_temporal"
                    if self.embedding_provider is not None
                    else "lexical_temporal_semantic_not_configured"
                ),
                "sensory": "manual_input_only",
                "immune": "ready",
                "regeneration": "proposal_only",
                "sleep": "replay_pattern_detection_owner_review",
                "schema_extraction": "annotated_event_pattern_owner_review",
                "circulation": "ready",
                "metabolism": "bounded_declared_work",
                "nervous_system": "in_process",
            },
            "reasoning_provider": reasoning,
            "continuous_runtime": self.runtime.snapshot().as_dict(),
            "runtime": {
                "storage_mode": "in_memory_no_database",
                "external_database_connected": False,
                "personal_experience_persists_restart": False,
                "identity_source": "version_controlled_source_code",
            },
            "cognitive_systems": self.cognitive_brain.architecture.status(),
            "neural_dynamics": self.neural_dynamics.snapshot().as_dict(),
            "neural_simulation": self.neural_simulation.snapshot().as_dict(),
            "associative_memory": self.cognitive_brain.knowledge_network.snapshot(),
            "relational_memory": self.cognitive_brain.architecture.relational_memory.snapshot(),
            "cognitive_memory": self.cognitive_memory.snapshot(),
            "episodic_memory": self.cognitive_brain.architecture.episodic_memory.snapshot(),
            "sleep_consolidation": self.sleep.snapshot(self.owner_id),
            "schema_extraction": self.schema_extraction.snapshot(self.owner_id),
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
