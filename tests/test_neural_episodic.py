from __future__ import annotations

from artificial_human.core.models import MemoryKind
from artificial_human.cognition.working_memory import WorkingMemory
from artificial_human.memory.cognitive import InMemoryCognitiveMemory
from artificial_human.memory.volatile_repository import InMemoryMemoryRepository
from artificial_human.neural.episodic import EpisodicNeuralCircuit
from artificial_human.organism import ArtificialOrganism


def _store(circuit: EpisodicNeuralCircuit, source_id: str, content: str) -> None:
    circuit.encode(
        owner_id="owner-a",
        consent_id="consent-a",
        source_id=source_id,
        context_id="home",
        content=content,
        occurred_at=10,
    )


def test_partial_cue_activates_engram_and_completes_missing_features() -> None:
    circuit = EpisodicNeuralCircuit()
    _store(circuit, "event-a", "blue mug placed on kitchen table")

    recall = circuit.recall("owner-a", "blue mug")

    assert recall.candidate_count == 1
    assert recall.activated_count == 1
    assert recall.network_spikes >= 3
    assert recall.simulation_steps == 6
    assert recall.matches[0].source_id == "event-a"
    assert recall.matches[0].neural_activation == 1.0
    assert recall.matches[0].expected_completion_count > 0
    assert recall.matches[0].completed_feature_count > 0
    assert recall.matches[0].completion_ratio > 0.0


def test_similar_episodes_show_measurable_interference_and_sparse_separation() -> None:
    circuit = EpisodicNeuralCircuit()
    _store(circuit, "event-a", "blue mug placed on kitchen table")
    _store(circuit, "event-b", "blue mug cleaned on kitchen table")

    recall = circuit.recall("owner-a", "blue mug")

    assert recall.candidate_count == 2
    assert recall.interference_index > 0.9
    assert recall.pattern_separation_index > 0.0
    assert recall.symbolic_top_source_id in {"event-a", "event-b"}
    assert recall.neural_top_source_id in {"event-a", "event-b"}
    assert all(item.neural_activation == 1.0 for item in recall.matches)


def test_successful_recall_strengthens_a_consent_scoped_engram() -> None:
    circuit = EpisodicNeuralCircuit()
    _store(circuit, "event-a", "blue mug placed on kitchen table")

    before = circuit.recall("owner-a", "blue mug").matches[0].score
    assert circuit.strengthen("owner-a", "event-a") is True
    after = circuit.recall("owner-a", "blue mug").matches[0].score

    assert after > before
    assert circuit.snapshot()["successful_strengthenings"] == 1


def test_engram_codes_do_not_retain_text_and_are_owner_scoped() -> None:
    circuit = EpisodicNeuralCircuit()
    engram = circuit.encode(
        owner_id="owner-a",
        consent_id="consent-a",
        source_id="event-a",
        context_id="private-session",
        content="blue mug placed on kitchen table",
        occurred_at=10,
    )

    assert not hasattr(engram, "content")
    assert "blue" not in repr(engram)
    assert all(len(feature) == 20 for feature in engram.feature_ids)
    assert circuit.recall("owner-b", "blue mug").matches == ()
    assert circuit.recall(
        "owner-a", "blue mug", context_id="other-session"
    ).matches == ()


def test_consent_revocation_and_owner_erasure_purge_engram_codes() -> None:
    circuit = EpisodicNeuralCircuit()
    _store(circuit, "event-a", "blue mug placed on kitchen table")
    circuit.encode(
        owner_id="owner-a",
        consent_id="consent-b",
        source_id="event-b",
        context_id="home",
        content="green cup near the sink",
        occurred_at=11,
    )
    circuit.encode(
        owner_id="owner-b",
        consent_id="consent-a",
        source_id="event-c",
        context_id="home",
        content="blue mug in another kitchen",
        occurred_at=12,
    )

    assert circuit.forget_consent("owner-a", "consent-a") == 1
    assert circuit.snapshot()["count"] == 2
    assert circuit.forget_owner("owner-a") == 1
    assert circuit.snapshot()["count"] == 1
    assert circuit.forget_source("event-c") == 1
    assert circuit.snapshot()["count"] == 0


def test_capacity_evicts_the_lowest_salience_engram() -> None:
    circuit = EpisodicNeuralCircuit(capacity=1)
    circuit.encode(
        owner_id="owner-a",
        consent_id="consent-a",
        source_id="weak",
        context_id="home",
        content="low priority event",
        occurred_at=1,
        salience=0.1,
    )
    circuit.encode(
        owner_id="owner-a",
        consent_id="consent-a",
        source_id="strong",
        context_id="home",
        content="important event",
        occurred_at=2,
        salience=1.0,
    )

    assert circuit.snapshot()["count"] == 1
    assert circuit.snapshot()["evictions"] == 1
    assert circuit.recall("owner-a", "low priority").matches == ()
    assert circuit.recall("owner-a", "important event").matches[0].source_id == "strong"


def test_cognitive_memory_keeps_neural_engrams_in_sync_with_consent() -> None:
    repository = InMemoryMemoryRepository()
    consent = repository.grant_consent("owner-a")
    cognitive = InMemoryCognitiveMemory()
    record = repository.add(
        owner_id="owner-a",
        kind=MemoryKind.EPISODIC,
        content="blue mug placed on kitchen table",
        source="owner-provided event",
        consent_id=consent.id,
        metadata={"context_id": "home", "tick": 10},
    )

    assert cognitive.ingest(record) == "episodic"
    recall = cognitive.recall("owner-a", "blue mug")
    assert recall.episodic_neural.candidate_count == 1
    assert recall.episodic_neural.activated_count == 1
    assert cognitive.snapshot()["episodic_neural"]["count"] == 1

    cleared = cognitive.forget_consent("owner-a", consent.id)
    assert cleared["episodic_neural"] == 1
    assert cognitive.snapshot()["episodic_neural"]["count"] == 0


def test_cognitive_cycle_admits_neural_episodic_completions_without_an_llm() -> None:
    organism = ArtificialOrganism(embedding_provider=None, brain_provider=None)
    consent = organism.memory.grant_consent(organism.owner_id)
    record = organism.memory.add(
        owner_id=organism.owner_id,
        kind=MemoryKind.EPISODIC,
        content="blue mug placed on kitchen table",
        source="owner-provided event",
        consent_id=consent.id,
        metadata={"context_id": "home", "tick": 10},
    )
    organism.cognitive_memory.ingest(record)

    cycle = organism.cognitive_brain.architecture.run(
        owner_id=organism.owner_id,
        query="blue mug",
        workspace=WorkingMemory(),
    )

    assert cycle.episodic_neural_state["candidate_count"] == 1
    assert cycle.episodic_neural_state["activated_count"] == 1
    assert cycle.episodic_neural_state["successful_retrieval_strengthenings_this_cycle"] == 1
    assert any(
        "episodic_neural_completion" in item.channels
        for item in cycle.memories
    )
