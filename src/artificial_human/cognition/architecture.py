from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Sequence

from artificial_human.brain.planner import GoalPlan
from artificial_human.brain.retrieval import MemoryMatch, MemoryRetriever, rank_texts
from artificial_human.cognition.attention import AttentionCandidate, AttentionSystem
from artificial_human.cognition.executive import ExecutiveControlSystem, ExecutiveState
from artificial_human.cognition.perception import Percept, PerceptionSystem
from artificial_human.cognition.prediction import PredictionAssessment, PredictionSystem
from artificial_human.cognition.working_memory import (
    WorkingMemory,
    new_working_memory_entry,
)
from artificial_human.learning.system import LearningSystem
from artificial_human.memory.repository import LearningExample, MemoryRecord
from artificial_human.memory.search import HybridMemorySearch


@dataclass(frozen=True, slots=True)
class CognitiveCycle:
    percept: Percept
    memories: tuple[MemoryMatch, ...]
    demonstrations: tuple[tuple[LearningExample, float], ...]
    executive: ExecutiveState
    prediction: PredictionAssessment
    attended: tuple[tuple[str, str, float, tuple[str, ...]], ...]
    working_memory_state: dict[str, object]
    working_memory_context: tuple[dict[str, str], ...]
    retrieval: dict[str, Any] = field(default_factory=dict)

    @property
    def plan(self) -> GoalPlan:
        return self.executive.plan

    def trace(self) -> dict[str, Any]:
        return {
            "architecture": "neuroscience_informed_cognitive_cycle_v1",
            "notice": (
                "These are explicit software mechanisms informed by cognitive science; "
                "they are not simulations of neurons or proof of biological equivalence."
            ),
            "perception": {
                "percept_id": self.percept.id,
                "modality": self.percept.modality,
                "source": self.percept.source,
                "token_count": len(self.percept.tokens),
            },
            "attention": {
                "selected_count": len(self.attended),
                "selected": [
                    {
                        "reference_id": item_id, "kind": kind, "weight": weight,
                        "retrieval_channels": list(channels),
                    }
                    for item_id, kind, weight, channels in self.attended
                ],
            },
            "working_memory": self.working_memory_state,
            "prediction": {
                "comparable": self.prediction.comparable,
                "novelty": self.prediction.novelty,
                "overlapping_terms": self.prediction.overlapping_terms,
                "context_term_count": self.prediction.context_term_count,
            },
            "executive_control": {
                "goal_id": self.plan.id,
                "step_count": len(self.plan.steps),
                "approval_required": self.executive.approval_required,
            },
            "memory": {
                "attended_count": len(self.memories),
                "memory_ids": [item.memory.id for item in self.memories],
                "retrieval": self.retrieval,
            },
            "learning": {
                "attended_demonstration_count": len(self.demonstrations),
                "demonstration_ids": [item.id for item, _ in self.demonstrations],
            },
        }


class CognitiveArchitecture:
    """Coordinates a modular cognitive cycle around explicit subsystem boundaries."""

    def __init__(
        self,
        *,
        perception: PerceptionSystem | None = None,
        attention: AttentionSystem | None = None,
        prediction: PredictionSystem | None = None,
        executive: ExecutiveControlSystem | None = None,
        retriever: MemoryRetriever | None = None,
        memory_search: HybridMemorySearch | None = None,
        learning: LearningSystem,
    ) -> None:
        self.perception = perception or PerceptionSystem()
        self.attention = attention or AttentionSystem()
        self.prediction = prediction or PredictionSystem()
        self.executive = executive or ExecutiveControlSystem()
        self.retriever = retriever or MemoryRetriever()
        self.memory_search = memory_search
        self.learning = learning

    @staticmethod
    def _normalise_scores(items: Sequence[tuple[Any, float]]) -> list[tuple[Any, float]]:
        maximum = max((score for _, score in items), default=0.0)
        if maximum <= 0:
            return []
        return [(item, min(1.0, score / maximum)) for item, score in items]

    def status(self) -> dict[str, str]:
        return {
            "perception": "explicit_language_input",
            "attention": "bounded_relevance_and_salience_heuristic",
            "working_memory": "capacity_limited_volatile_workspace",
            "prediction": "lexical_novelty_error_heuristic",
            "executive_control": "goal_decomposition_with_approval_gate",
            "memory": (
                "consent_filtered_hybrid_lexical_semantic_temporal_retrieval"
                if self.memory_search is not None and self.memory_search.index is not None
                and self.memory_search.index.configured
                else "consent_filtered_lexical_temporal_retrieval"
                if self.memory_search is not None
                else "consent_filtered_lexical_retrieval"
            ),
            "learning": "owner_approved_demonstrations",
            "language_model": "bounded_generation_and_deliberation_interface",
        }

    @staticmethod
    def _explicit_salience(metadata: dict[str, Any]) -> float:
        value = metadata.get("salience", 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0.0
        return float(value) if 0.0 <= value <= 1.0 else 0.0

    def _retrieve(
        self,
        owner_id: str,
        query: str,
        memories: Sequence[MemoryRecord] | None,
        max_memories: int,
    ) -> tuple[list[MemoryMatch], dict[str, Any]]:
        """Retrieve long-term memory candidates before attention selection.

        With a configured long-term memory search and no caller-supplied
        records, the owner-scoped hybrid search runs and returns twice the
        memory budget so attention can choose among lexical, semantic and
        temporal candidates. A caller-supplied record list keeps the previous
        lexical behaviour over exactly those records.
        """
        if memories is not None or self.memory_search is None:
            matches = self.retriever.retrieve(query, memories or (), limit=max_memories)
            return matches, {"strategy": "lexical_bm25_over_supplied_records"}
        response = self.memory_search.search(
            owner_id, query, top_k=min(max(max_memories * 2, 1), 100)
        )
        matches = [
            MemoryMatch(
                memory=item.memory, score=item.score, channels=item.channels,
                explanation=item.explanation,
            )
            for item in response.results
        ]
        return matches, response.explain()

    def run(
        self,
        *,
        owner_id: str,
        query: str,
        memories: Sequence[MemoryRecord] | None = None,
        workspace: WorkingMemory,
        max_memories: int = 5,
        max_demonstrations: int = 3,
        attention_limit: int = 8,
    ) -> CognitiveCycle:
        percept = self.perception.perceive(query)
        prior_context = workspace.context(limit=workspace.capacity)
        prediction = self.prediction.assess(
            percept, [entry.content for entry in prior_context]
        )

        retrieved, retrieval = self._retrieve(owner_id, query, memories, max_memories)
        examples = self.learning.examples(owner_id, status="approved", limit=500)
        ranked_examples = rank_texts(
            query,
            [
                (
                    item.id,
                    f"{item.task_category} {item.input_text} {item.target_text}",
                )
                for item in examples
            ],
            limit=max_demonstrations,
        )
        examples_by_id = {item.id: item for item in examples}
        ranked_demonstrations = [
            (examples_by_id[item_id], score)
            for item_id, score in ranked_examples
            if item_id in examples_by_id
        ]

        candidates: list[AttentionCandidate] = [
            AttentionCandidate(
                reference_id=percept.id,
                kind="percept",
                relevance=0.85,
                salience=prediction.novelty if prediction.comparable else 0.0,
            )
        ]
        memory_scores = self._normalise_scores(
            [(item, item.score) for item in retrieved]
        )
        example_scores = self._normalise_scores(ranked_demonstrations)
        for match, relevance in memory_scores:
            candidates.append(
                AttentionCandidate(
                    reference_id=match.memory.id,
                    kind="memory",
                    relevance=relevance,
                    salience=self._explicit_salience(match.memory.metadata),
                    channels=match.channels,
                )
            )
        for example, relevance in example_scores:
            candidates.append(
                AttentionCandidate(
                    reference_id=example.id,
                    kind="learning_example",
                    relevance=relevance,
                )
            )

        attended = self.attention.select(candidates, limit=attention_limit)
        attended_lookup = {
            (item.reference_id, item.kind): item.weight for item in attended
        }
        selected_memories = tuple(
            item for item in retrieved
            if (item.memory.id, "memory") in attended_lookup
        )[:max_memories]
        selected_demonstrations = tuple(
            item for item in ranked_demonstrations
            if (item[0].id, "learning_example") in attended_lookup
        )

        workspace.remember(
            new_working_memory_entry(
                id=f"percept:{percept.id}",
                content=percept.content,
                kind="percept",
                source=percept.source,
                salience=1.0,
            )
        )
        for match in selected_memories:
            workspace.remember(
                new_working_memory_entry(
                    id=f"memory:{match.memory.id}",
                    content=match.memory.content[:1200],
                    kind=match.memory.kind.value,
                    source=match.memory.source,
                    salience=attended_lookup[(match.memory.id, "memory")],
                )
            )
        for example, _ in selected_demonstrations:
            workspace.remember(
                new_working_memory_entry(
                    id=f"example:{example.id}",
                    content=example.target_text[:1200],
                    kind="approved_demonstration",
                    source=example.source,
                    salience=attended_lookup[(example.id, "learning_example")],
                )
            )

        executive = self.executive.prepare(query)
        attended_trace = tuple(
            (item.reference_id, item.kind, item.weight, item.channels) for item in attended
        )
        return CognitiveCycle(
            percept=percept,
            memories=selected_memories,
            demonstrations=selected_demonstrations,
            executive=executive,
            prediction=prediction,
            attended=attended_trace,
            working_memory_state=workspace.public_state(),
            retrieval=retrieval,
            working_memory_context=tuple(
                {
                    "id": item.id,
                    "kind": item.kind,
                    "source": item.source,
                    "content": item.content,
                }
                for item in workspace.context(limit=5)
            ),
        )
