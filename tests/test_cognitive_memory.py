from __future__ import annotations

from artificial_human.core.models import MemoryKind
from artificial_human.cognition.working_memory import WorkingMemory
from artificial_human.memory.cognitive import (
    EpisodicMemoryStore,
    InMemoryCognitiveMemory,
    ProceduralMemoryStore,
    SemanticMemoryStore,
    SpatialMemoryStore,
)
from artificial_human.memory.volatile_repository import InMemoryMemoryRepository
from artificial_human.memory.repository import MemoryRepository
from artificial_human.organism import ArtificialOrganism


def test_episodic_memory_binds_context_and_time_and_completes_partial_cues() -> None:
    store = EpisodicMemoryStore(capacity=3)
    kitchen = store.store(
        owner_id="owner", consent_id="consent", source_id="event-1",
        context_id="home", content="blue mug placed on kitchen table",
        occurred_at=10, tags=("breakfast",), salience=0.9,
    )
    store.store(
        owner_id="owner", consent_id="consent", source_id="event-2",
        context_id="garage", content="red bicycle repaired near workbench",
        occurred_at=11, tags=("maintenance",), salience=0.8,
    )

    recalled = store.retrieve("owner", "mug table", context_id="home")

    assert len(recalled) == 1
    assert recalled[0].id == kitchen.id
    assert recalled[0].access_count == 1
    assert recalled[0].strength > kitchen.strength
    assert store.retrieve("other-owner", "mug table") == ()


def test_episodic_capacity_evicts_lowest_salience_trace_and_forgetting_is_explicit() -> None:
    store = EpisodicMemoryStore(capacity=1)
    weak = store.store(
        owner_id="owner", consent_id="c", source_id="weak", context_id="ctx",
        content="low priority event", occurred_at=1, salience=0.1,
    )
    strong = store.store(
        owner_id="owner", consent_id="c", source_id="strong", context_id="ctx",
        content="important event", occurred_at=2, salience=1.0,
    )
    assert store.snapshot()["count"] == 1
    assert store.snapshot()["evictions"] == 1
    assert store.forget_source(weak.source_id) is False
    assert store.forget_source(strong.source_id) is True
    assert store.snapshot()["count"] == 0


def test_semantic_memory_merges_evidence_and_strengthens_retrieval() -> None:
    store = SemanticMemoryStore(capacity=3)
    first = store.learn(
        owner_id="owner", consent_id="c", source_id="source-1",
        concept="mitochondria", attributes={"role": "cell energy"}, confidence=0.7,
    )
    merged = store.learn(
        owner_id="owner", consent_id="c", source_id="source-2",
        concept="Mitochondria", attributes={"location": "cell"}, confidence=0.9,
    )
    recalled = store.retrieve("owner", "cell energy mitochondria")

    assert first.id == merged.id
    assert merged.source_ids == ("source-1", "source-2")
    assert dict(merged.attributes) == {"location": "cell", "role": "cell energy"}
    assert recalled[0].id == first.id
    assert recalled[0].access_count == 1
    assert recalled[0].strength > merged.strength


def test_semantic_associations_support_related_concept_retrieval_and_consent_purge() -> None:
    store = SemanticMemoryStore()
    first = store.learn(
        owner_id="owner", consent_id="c", source_id="s1", concept="neuron",
    )
    second = store.learn(
        owner_id="owner", consent_id="c", source_id="s2", concept="synapse",
    )
    store.associate("owner", first.id, second.id, "communicates_through")
    assert store.related("owner", first.id)[0][0].id == second.id
    assert store.forget_consent("owner", "c") == 2
    assert store.snapshot()["count"] == 0
    assert store.snapshot()["associations"] == 0


def test_procedural_memory_learns_steps_and_updates_from_outcomes() -> None:
    store = ProceduralMemoryStore(capacity=2)
    procedure = store.learn(
        owner_id="owner", consent_id="c", source_id="demo-1",
        task="deploy web service",
        steps=("run tests", "build image", "deploy to staging", "verify health"),
    )
    for _ in range(3):
        procedure = store.record_outcome("owner", procedure.id, success=True)
    procedure = store.record_outcome("owner", procedure.id, success=False)
    recalled = store.retrieve("owner", "deploy web service")

    assert recalled[0].id == procedure.id
    assert recalled[0].steps == (
        "run tests", "build image", "deploy to staging", "verify health"
    )
    assert recalled[0].successes == 3
    assert recalled[0].failures == 1
    assert recalled[0].success_rate > 0.5
    assert store.forget_source("demo-1") == 1


def test_spatial_memory_uses_coordinates_and_explicit_relations() -> None:
    store = SpatialMemoryStore(capacity=3)
    kitchen = store.observe(
        owner_id="owner", consent_id="c", source_id="s1",
        label="kitchen", coordinates=(0.0, 0.0, 0.0), context_id="home",
        tags=("food",),
    )
    office = store.observe(
        owner_id="owner", consent_id="c", source_id="s2",
        label="office", coordinates=(10.0, 0.0, 0.0), context_id="home",
    )
    store.connect("owner", kitchen.id, office.id, "east_of")
    nearest = store.nearest("owner", (1.0, 0.0, 0.0), context_id="home")

    assert nearest[0][0].id == kitchen.id
    assert nearest[0][1] == 1.0
    assert store.retrieve("owner", "food kitchen")[0].id == kitchen.id
    assert store.nearest("another-owner", (0.0, 0.0, 0.0)) == ()
    assert store.forget_source("s1") == 1
    assert store.snapshot()["relations"] == 0


def test_cognitive_memory_routes_memory_kinds_to_distinct_stores_and_resets() -> None:
    repository = InMemoryMemoryRepository()
    consent = repository.grant_consent("owner")
    cognitive = InMemoryCognitiveMemory()

    examples = (
        (MemoryKind.EPISODIC, "A delivery arrived at the office", {"context_id": "work", "tick": 5}),
        (MemoryKind.SEMANTIC, "Python is a programming language", {"concept": "Python", "attributes": {"type": "language"}}),
        (MemoryKind.PROCEDURAL, "deploy service", {"task": "deploy service", "steps": ["test", "deploy"]}),
        (MemoryKind.SPATIAL, "office", {"label": "office", "coordinates": [1.0, 2.0, 3.0]}),
    )
    for kind, content, metadata in examples:
        record = repository.add(
            owner_id="owner", kind=kind, content=content, source="explicit input",
            consent_id=consent.id, metadata=metadata,
        )
        assert cognitive.ingest(record) == kind.value

    recalled = cognitive.recall("owner", "Python programming language")
    assert recalled.semantic
    assert cognitive.snapshot()["semantic"]["count"] == 1
    assert cognitive.snapshot()["episodic"]["count"] == 1
    assert cognitive.snapshot()["procedural"]["count"] == 1
    assert cognitive.snapshot()["spatial"]["count"] == 1

    cognitive.forget_consent("owner", consent.id)
    assert cognitive.snapshot()["semantic"]["count"] == 0
    assert cognitive.snapshot()["episodic"]["count"] == 0
    assert cognitive.snapshot()["procedural"]["count"] == 0
    assert cognitive.snapshot()["spatial"]["count"] == 0


def test_active_organism_uses_same_working_memory_owner_and_resets_in_ram() -> None:
    organism = ArtificialOrganism(embedding_provider=None)
    assert not isinstance(organism.memory, MemoryRepository)
    assert organism.cognitive_brain.working_memory_sessions is organism.cognitive_memory.working
    grant = organism.memory.grant_consent(organism.owner_id)
    record = organism.memory.add(
        owner_id=organism.owner_id, kind=MemoryKind.SEMANTIC,
        content="memory architecture", source="test", consent_id=grant.id,
    )
    organism.cognitive_memory.ingest(record)
    assert organism.cognitive_memory.snapshot()["semantic"]["count"] == 1

    organism.reset_volatile_state()
    assert organism.memory.list(organism.owner_id) == []
    assert organism.cognitive_memory.snapshot()["semantic"]["count"] == 0
    assert organism.cognitive_memory.snapshot()["working"]["session_count"] == 0



def test_cognitive_cycle_recalls_specialized_semantic_memory_without_database() -> None:
    organism = ArtificialOrganism(embedding_provider=None)
    grant = organism.memory.grant_consent(organism.owner_id)
    record = organism.memory.add(
        owner_id=organism.owner_id,
        kind=MemoryKind.SEMANTIC,
        content="Python is a programming language",
        source="owner-provided fact",
        consent_id=grant.id,
        metadata={"concept": "Python", "attributes": {"type": "programming language"}},
    )
    organism.cognitive_memory.ingest(record)

    cycle = organism.cognitive_brain.architecture.run(
        owner_id=organism.owner_id,
        query="Python programming language",
        memories=(),
        workspace=WorkingMemory(),
    )

    assert cycle.specialized_memory_state["semantic_hits"] == 1
    assert any(
        item["kind"] == "semantic_fact" and "Python" in item["content"]
        for item in cycle.working_memory_context
    )
