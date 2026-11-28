from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Sequence

from artificial_human.brain.planner import GoalPlan
from artificial_human.brain.retrieval import MemoryMatch, MemoryRetriever, rank_texts
from artificial_human.cognition.attention import AttentionCandidate, AttentionSystem
from artificial_human.cognition.episodic_replay import (
    EpisodicEvent,
    EpisodicReplayMemory,
)
from artificial_human.cognition.executive import ExecutiveControlSystem, ExecutiveState
from artificial_human.cognition.perception import Percept, PerceptionSystem
from artificial_human.cognition.prediction import PredictionAssessment, PredictionSystem
from artificial_human.cognition.relational_memory import RelationalMemoryNetwork
from artificial_human.cognition.working_memory import (
    WorkingMemory,
    new_working_memory_entry,
)
from artificial_human.core.models import MemoryKind
from artificial_human.learning.system import LearningSystem
from artificial_human.memory.repository import (
    LearningExample,
    MemoryNotFoundError,
    MemoryRecord,
)
from artificial_human.memory.search import HybridMemorySearch
from artificial_human.memory.cognitive import InMemoryCognitiveMemory


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
    associative_memory_state: dict[str, object] = field(default_factory=dict)
    episodic_replay_state: dict[str, object] = field(default_factory=dict)
    specialized_memory_state: dict[str, object] = field(default_factory=dict)

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
            "relational_memory": self.associative_memory_state,
            "episodic_replay": self.episodic_replay_state,
            "specialized_memory": self.specialized_memory_state,
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
        relational_memory: RelationalMemoryNetwork | None = None,
        episodic_memory: EpisodicReplayMemory | None = None,
        cognitive_memory: InMemoryCognitiveMemory | None = None,
        learning: LearningSystem,
    ) -> None:
        self.perception = perception or PerceptionSystem()
        self.attention = attention or AttentionSystem()
        self.prediction = prediction or PredictionSystem()
        self.executive = executive or ExecutiveControlSystem()
        self.retriever = retriever or MemoryRetriever()
        self.memory_search = memory_search
        self.cognitive_memory = cognitive_memory or InMemoryCognitiveMemory()
        self.relational_memory = relational_memory or self.cognitive_memory.relational
        self.episodic_memory = episodic_memory or self.cognitive_memory.episodic_replay
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
            "associative_memory": (
                "session_scoped_relational_links_with_contradiction_suppression"
            ),
            "episodic_memory": "one_trial_context_bound_events_with_partial_cue_recall",
            "semantic_memory": "concept_attribute_facts_with_evidence_and_retrieval_strengthening",
            "procedural_memory": "task_step_sequences_with_outcome_based_strengthening",
            "spatial_memory": "coordinate_indexed_places_with_nearest_neighbour_recall",
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

    @staticmethod
    def _contradiction_pairs(
        matches: Sequence[MemoryMatch],
    ) -> tuple[tuple[str, str], ...]:
        """Extract explicit, owner-visible contradictions among current candidates."""
        available = {item.memory.id for item in matches}
        pairs: set[tuple[str, str]] = set()
        for match in matches:
            values = match.explanation.get("conflicts_with", ())
            if not isinstance(values, (list, tuple, set)):
                continue
            for other in values:
                if isinstance(other, str) and other in available and other != match.memory.id:
                    pairs.add(
                        (match.memory.id, other)
                        if match.memory.id < other else (other, match.memory.id)
                    )
        return tuple(sorted(pairs))

    def _associated_record(
        self,
        owner_id: str,
        memory_id: str,
        supplied_memories: Sequence[MemoryRecord] | None,
    ) -> MemoryRecord | None:
        """Resolve an associated ID only inside the same consent-visible owner scope."""
        if supplied_memories is not None:
            return next(
                (
                    item for item in supplied_memories
                    if item.id == memory_id and item.owner_id == owner_id
                ),
                None,
            )
        try:
            return self.learning.repository.get(owner_id, memory_id)
        except MemoryNotFoundError:
            # Deleted, revoked, or foreign IDs are ignored rather than revived.
            return None

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
        association_context_id: str = "global",
    ) -> CognitiveCycle:
        percept = self.perception.perceive(query)
        specialized_recall = self.cognitive_memory.recall(owner_id, query)
        prior_context = workspace.context(limit=workspace.capacity)
        prediction = self.prediction.assess(
            percept, [entry.content for entry in prior_context]
        )

        retrieved, retrieval = self._retrieve(owner_id, query, memories, max_memories)
        contradictions = self._contradiction_pairs(retrieved)
        direct_memory_ids = [item.memory.id for item in retrieved]
        direct_ids = set(direct_memory_ids)
        visible_contradictions: set[tuple[str, str]] = set()
        for link in self.learning.repository.links(owner_id):
            if link.relation == "contradicts":
                visible_contradictions.add(
                    (link.from_memory_id, link.to_memory_id)
                    if link.from_memory_id < link.to_memory_id
                    else (link.to_memory_id, link.from_memory_id)
                )
        direct_conflicts = {
            pair for pair in visible_contradictions
            if pair[0] in direct_ids and pair[1] in direct_ids
        }
        contradictions = tuple(sorted(set(contradictions) | direct_conflicts))
        associated = self.relational_memory.activate(
            direct_memory_ids,
            context_id=association_context_id,
            contradictions=contradictions,
        )
        direct_episodic_ids = [
            item.memory.id for item in retrieved if item.memory.kind is MemoryKind.EPISODIC
        ]
        temporal_replays = self.episodic_memory.replay(
            direct_episodic_ids,
            context_id=association_context_id,
            blocked_pairs=tuple(sorted(visible_contradictions)),
        )
        blocked_ids = {memory_id for pair in contradictions for memory_id in pair}
        strongest_retrieval_score = max(
            (item.score for item in retrieved),
            default=0.0,
        )
        for activation in associated:
            if activation.memory_id in direct_ids or activation.memory_id in blocked_ids:
                continue
            record = self._associated_record(owner_id, activation.memory_id, memories)
            if record is None:
                continue
            hidden_conflicts = [
                (direct_id, activation.memory_id)
                if direct_id < activation.memory_id
                else (activation.memory_id, direct_id)
                for direct_id in direct_ids
                if (
                    (direct_id, activation.memory_id)
                    if direct_id < activation.memory_id
                    else (activation.memory_id, direct_id)
                ) in visible_contradictions
            ]
            if hidden_conflicts:
                contradictions = tuple(sorted(set(contradictions) | set(hidden_conflicts)))
                blocked_ids.update({memory_id for pair in hidden_conflicts for memory_id in pair})
                self.relational_memory.learn(
                    [],
                    context_id=association_context_id,
                    contradictions=hidden_conflicts,
                )
                continue
            retrieved.append(
                MemoryMatch(
                    memory=record,
                    score=max(activation.strength * strongest_retrieval_score * 0.5, 1e-6),
                    channels=("associative",),
                    explanation={
                        "associative": {
                            "strength": activation.strength,
                            "source": "same_context_co_retrieval",
                        },
                        "conflicts_with": [],
                    },
                )
            )
        for replay in temporal_replays:
            if replay.memory_id in direct_ids or replay.memory_id in blocked_ids:
                continue
            record = self._associated_record(owner_id, replay.memory_id, memories)
            if record is None or record.kind is not MemoryKind.EPISODIC:
                continue
            current_conflicts = self._contradiction_pairs(retrieved)
            if any(replay.memory_id in pair for pair in current_conflicts):
                continue
            replay_explanation = {
                "episodic_replay": {
                    "cue_memory_id": replay.cue_memory_id,
                    "temporal_relation": replay.relation,
                    "distance": replay.distance,
                    "strength": replay.strength,
                },
            }
            existing_index = next(
                (
                    index for index, item in enumerate(retrieved)
                    if item.memory.id == replay.memory_id
                ),
                None,
            )
            if existing_index is not None:
                existing = retrieved[existing_index]
                retrieved[existing_index] = MemoryMatch(
                    memory=existing.memory,
                    score=max(
                        existing.score,
                        replay.strength * strongest_retrieval_score * 0.45,
                    ),
                    channels=tuple(dict.fromkeys((*existing.channels, "episodic_replay"))),
                    explanation={**existing.explanation, **replay_explanation},
                )
                continue
            retrieved.append(
                MemoryMatch(
                    memory=record,
                    score=max(replay.strength * strongest_retrieval_score * 0.45, 1e-6),
                    channels=("episodic_replay",),
                    explanation={
                        **replay_explanation,
                        "conflicts_with": [],
                    },
                )
            )
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
        # Learn only from memory candidates that survived attention. Contradictory
        # pairs remain visible as evidence but never become positive associations.
        self.relational_memory.learn(
            [item.memory.id for item in selected_memories],
            context_id=association_context_id,
            contradictions=contradictions,
        )
        selected_ids = {item.memory.id for item in selected_memories}
        reinforced_replays = [
            item for item in temporal_replays if item.memory_id in selected_ids
        ]
        strengthened_replays = self.episodic_memory.reinforce_in_context(
            reinforced_replays,
            context_id=association_context_id,
        )
        selected_episodes = [
            EpisodicEvent(
                memory_id=item.memory.id,
                occurred_at=item.memory.occurred_at or item.memory.created_at,
            )
            for item in selected_memories
            if item.memory.kind is MemoryKind.EPISODIC
        ]
        episode_formed = self.episodic_memory.record_episode(
            selected_episodes,
            context_id=association_context_id,
        )
        episodic_replay_state = {
            **self.episodic_memory.snapshot(),
            "replayed_candidate_count": len(temporal_replays),
            "replayed_items_selected": len(reinforced_replays),
            "retrieval_strengthenings_this_cycle": strengthened_replays,
            "episode_formed_this_cycle": episode_formed,
        }
        relational_state = {
            "activated_candidates": len(associated),
            "explicit_contradiction_pairs": len(contradictions),
            **self.relational_memory.snapshot(),
        }

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
        for episodic_item in specialized_recall.episodic:
            workspace.remember(
                new_working_memory_entry(
                    id=f"cognitive:episodic:{episodic_item.id}",
                    content=episodic_item.content[:1200],
                    kind="episodic_trace",
                    source="volatile_episodic_memory",
                    salience=episodic_item.salience * episodic_item.strength,
                )
            )
        for semantic_item in specialized_recall.semantic:
            details = "; ".join(f"{key}: {value}" for key, value in semantic_item.attributes)
            workspace.remember(
                new_working_memory_entry(
                    id=f"cognitive:semantic:{semantic_item.id}",
                    content=(semantic_item.concept + (f" ({details})" if details else ""))[:1200],
                    kind="semantic_fact",
                    source="volatile_semantic_memory",
                    salience=semantic_item.confidence * semantic_item.strength,
                )
            )
        for procedure_item in specialized_recall.procedural:
            workspace.remember(
                new_working_memory_entry(
                    id=f"cognitive:procedural:{procedure_item.id}",
                    content=(procedure_item.task + ": " + " -> ".join(procedure_item.steps))[:1200],
                    kind="procedural_sequence",
                    source="volatile_procedural_memory",
                    salience=procedure_item.success_rate * procedure_item.strength,
                )
            )
        for spatial_item in specialized_recall.spatial:
            workspace.remember(
                new_working_memory_entry(
                    id=f"cognitive:spatial:{spatial_item.id}",
                    content=f"{spatial_item.label} at {spatial_item.coordinates}",
                    kind="spatial_place",
                    source="volatile_spatial_memory",
                    salience=0.5,
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
            associative_memory_state=relational_state,
            episodic_replay_state=episodic_replay_state,
            specialized_memory_state={
                "episodic_hits": len(specialized_recall.episodic),
                "semantic_hits": len(specialized_recall.semantic),
                "procedural_hits": len(specialized_recall.procedural),
                "spatial_hits": len(specialized_recall.spatial),
                "stores": self.cognitive_memory.snapshot(),
            },
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
