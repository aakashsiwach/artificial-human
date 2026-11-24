from artificial_human.brain.retrieval import MemoryRetriever, rank_texts
from artificial_human.core.models import MemoryKind
from artificial_human.memory.repository import MemoryRecord


def test_rank_texts_prefers_documents_with_more_query_overlap() -> None:
    ranked = rank_texts(
        "how does memory retrieval work",
        [
            ("unrelated", "The heart pumps blood through the body."),
            ("relevant", "Memory retrieval searches stored memory records."),
            ("partial", "Memory records preserve source information."),
        ],
        limit=3,
    )
    assert ranked[0][0] == "relevant"
    assert len(ranked) == 2


def test_memory_retriever_returns_only_matching_records() -> None:
    memories = [
        MemoryRecord(
            id="m1", owner_id="owner", kind=MemoryKind.PREFERENCE,
            content="I prefer concise technical explanations.",
            source="owner statement", consent_id="consent", created_at="2026-01-01",
            metadata={},
        ),
        MemoryRecord(
            id="m2", owner_id="owner", kind=MemoryKind.EPISODIC,
            content="I visited Jaipur during winter.",
            source="journal", consent_id="consent", created_at="2026-01-02",
            metadata={},
        ),
    ]
    results = MemoryRetriever().retrieve("concise technical explanations", memories)
    assert [item.memory.id for item in results] == ["m1"]
