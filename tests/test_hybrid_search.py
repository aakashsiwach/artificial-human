from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest
from conftest import ConceptEmbedding

from artificial_human.core.models import MemoryKind
from artificial_human.memory.repository import MemoryRepository
from artificial_human.memory.search import (
    SearchConfig,
    SearchFilters,
    bm25,
    lexical_terms,
)
from artificial_human.memory.system import LongTermMemory
from artificial_human.memory.temporal import TemporalStatus, TimeRange

NOW = datetime(2026, 6, 15, 12, tzinfo=UTC)


@pytest.fixture
def world() -> tuple[LongTermMemory, dict[str, str]]:
    clock = {"now": datetime(2025, 1, 1, tzinfo=UTC)}
    repository = MemoryRepository(":memory:", clock=lambda: clock["now"])
    memory = LongTermMemory(repository, ConceptEmbedding())
    consent = repository.grant_consent("owner").id
    other_consent = repository.grant_consent("other").id
    ids: dict[str, str] = {}

    def add(key: str, when: str, **fields: object) -> None:
        clock["now"] = datetime.fromisoformat(when)
        record, _ = memory.add(owner_id="owner", consent_id=consent, source=f"src:{key}", **fields)
        ids[key] = record.id

    add("water", "2025-02-01T00:00:00+00:00", kind=MemoryKind.SEMANTIC,
        content="Water boils at approximately 100 degrees Celsius at sea level.")
    add("tea", "2025-02-02T00:00:00+00:00", kind=MemoryKind.PREFERENCE,
        content="I drink green tea every morning.")
    add("job_a", "2025-02-03T00:00:00+00:00", kind=MemoryKind.SEMANTIC,
        content="The user works at Company A.", fact_key="owner.employer",
        valid_from="2024-09-01T00:00:00+00:00")
    clock["now"] = datetime(2026, 3, 5, tzinfo=UTC)
    job_b, _, _ = memory.supersede(
        owner_id="owner", memory_id=ids["job_a"], consent_id=consent,
        content="The user moved to Company B.", source="src:job_b",
        valid_from="2026-03-01T00:00:00+00:00",
    )
    ids["job_b"] = job_b.id
    add("flight", "2026-04-07T00:00:00+00:00", kind=MemoryKind.EPISODIC,
        content="Flew home from Lisbon after the conference.",
        occurred_at="2026-04-07T10:00:00+01:00")
    add("hotel", "2026-04-03T00:00:00+00:00", kind=MemoryKind.EPISODIC,
        content="Checked into a hotel in Lisbon.", occurred_at="2026-04-03T15:00:00+01:00")
    add("flu", "2026-01-21T00:00:00+00:00", kind=MemoryKind.EPISODIC,
        content="Stayed home sick with the flu.", occurred_at="2026-01-20T09:00:00+00:00")
    add("allergy_yes", "2025-03-01T00:00:00+00:00", kind=MemoryKind.SEMANTIC,
        content="The user is allergic to peanuts.", fact_key="owner.allergy")
    add("allergy_no", "2026-02-01T00:00:00+00:00", kind=MemoryKind.SEMANTIC,
        content="The user has no food allergies at all.", fact_key="owner.allergy")
    clock["now"] = datetime(2025, 5, 1, tzinfo=UTC)
    other, _ = memory.add(
        owner_id="other", consent_id=other_consent, kind=MemoryKind.SEMANTIC,
        source="src:other", content="Water boils at lower temperature at high altitude.",
    )
    ids["other_water"] = other.id
    return memory, ids


def _ids(response) -> list[str]:
    return [item.memory.id for item in response.results]


def test_lexical_terms_remove_stopwords_and_share_stems() -> None:
    assert lexical_terms("The water is boiling") == ["water", "boil"]
    assert lexical_terms("boils") == lexical_terms("boiled") == ["boil"]
    ranked = bm25("boiling water", [("a", "Water boils."), ("b", "Tea is hot.")])
    assert [item[0] for item in ranked] == ["a"]
    assert ranked[0][2] == ("boil", "water")


def test_semantic_channel_retrieves_paraphrases_without_shared_keywords(world) -> None:
    memory, ids = world
    lexical_only = memory.search(
        "owner", "At what heat does H2O turn to vapor?", reference_time=NOW,
        config=SearchConfig(use_semantic=False),
    )
    assert ids["water"] not in _ids(lexical_only)
    hybrid = memory.search("owner", "At what heat does H2O turn to vapor?", reference_time=NOW)
    assert _ids(hybrid)[0] == ids["water"]
    assert hybrid.results[0].channels == ("semantic",)
    assert hybrid.strategy == "lexical+semantic"


def test_results_explain_fusion_provenance_and_evidence(world) -> None:
    memory, ids = world
    response = memory.search("owner", "water boiling temperature", reference_time=NOW)
    top = response.results[0]
    assert top.memory.id == ids["water"]
    explanation = top.explanation
    assert explanation["source"] == "src:water"
    assert explanation["fusion"]["method"] == "weighted_reciprocal_rank_fusion"
    expected = 1 / (60 + 1) + 1 / (60 + 1)
    assert explanation["fusion"]["score"] == pytest.approx(expected, abs=1e-6)
    assert explanation["evidence"]["lexical"]["matched_terms"] == ["boil", "water"]
    assert "cosine_similarity" in explanation["evidence"]["semantic"]
    public = response.explain()
    assert "content" not in str(public["results"])  # explanations never carry content


def test_owner_isolation_holds_for_every_channel(world) -> None:
    memory, ids = world
    owner_view = memory.search("owner", "water boils altitude", top_k=10, reference_time=NOW)
    other_view = memory.search("other", "water boils", top_k=10, reference_time=NOW)
    assert ids["other_water"] not in _ids(owner_view)
    assert _ids(other_view) == [ids["other_water"]]
    assert memory.search("intruder", "water", reference_time=NOW).results == ()


def test_lexical_only_mode_when_semantic_unavailable(world) -> None:
    memory, ids = world
    lexical = LongTermMemory(memory.repository, None)
    response = lexical.search("owner", "green tea", reference_time=NOW)
    assert response.strategy == "lexical"
    assert response.semantic["reason"] == "no embedding backend configured"
    assert _ids(response) == [ids["tea"]]


def test_memory_type_and_source_filters(world) -> None:
    memory, ids = world
    episodic = memory.search(
        "owner", "Lisbon", top_k=10, reference_time=NOW,
        filters=SearchFilters(kinds=(MemoryKind.EPISODIC,)),
    )
    assert set(_ids(episodic)) == {ids["flight"], ids["hotel"]}
    assert all(item.memory.kind is MemoryKind.EPISODIC for item in episodic.results)
    preferences = memory.search(
        "owner", "Lisbon", reference_time=NOW,
        filters=SearchFilters(kinds=(MemoryKind.PREFERENCE,)),
    )
    assert preferences.results == ()
    sourced = memory.search(
        "owner", "Lisbon", reference_time=NOW, filters=SearchFilters(source_prefix="src:hotel")
    )
    assert _ids(sourced) == [ids["hotel"]]


def test_current_historical_and_month_queries_distinguish_fact_versions(world) -> None:
    memory, ids = world
    current = memory.search("owner", "Where does the user work now?", reference_time=NOW)
    assert _ids(current)[0] == ids["job_b"]
    assert ids["job_a"] not in _ids(current)
    assert current.temporal_query.intent.value == "current"

    previous = memory.search("owner", "Where did the user work previously?", reference_time=NOW)
    assert _ids(previous)[0] == ids["job_a"]
    assert ids["job_b"] not in _ids(previous)
    assert previous.results[0].temporal.status is TemporalStatus.SUPERSEDED
    assert previous.results[0].temporal.superseded_by == ids["job_b"]

    january = memory.search("owner", "Where did the user work in January?", reference_time=NOW)
    assert _ids(january)[0] == ids["job_a"]
    assert ids["job_b"] not in _ids(january)


def test_explicit_valid_at_and_known_at_filters(world) -> None:
    memory, ids = world
    as_of_2025 = memory.search(
        "owner", "user company", top_k=10, reference_time=NOW,
        filters=SearchFilters(valid_at=datetime(2025, 6, 1, tzinfo=UTC)),
    )
    assert ids["job_a"] in _ids(as_of_2025) and ids["job_b"] not in _ids(as_of_2025)
    # Before the supersession was recorded, the system only knew about Company A.
    known_then = memory.search(
        "owner", "user company", top_k=10, reference_time=NOW,
        filters=SearchFilters(known_at=datetime(2026, 1, 1, tzinfo=UTC)),
    )
    assert _ids(known_then) == [ids["job_a"]]
    assert known_then.results[0].temporal.status is TemporalStatus.VALID


def test_time_range_filter_and_timeline_are_chronological(world) -> None:
    memory, ids = world
    april = TimeRange(datetime(2026, 4, 1, tzinfo=UTC), datetime(2026, 5, 1, tzinfo=UTC))
    timeline = memory.search_engine.timeline(
        "owner", filters=SearchFilters(time_range=april, kinds=(MemoryKind.EPISODIC,)),
        reference_time=NOW,
    )
    assert [record.id for record, _ in timeline] == [ids["hotel"], ids["flight"]]
    first = memory.search("owner", "What did I do first in Lisbon?", reference_time=NOW)
    assert _ids(first)[0] == ids["hotel"]


def test_temporal_only_queries_recall_events_in_a_period(world) -> None:
    memory, ids = world
    january = memory.search("owner", "What happened in January?", reference_time=NOW)
    assert _ids(january)[0] == ids["flu"]
    assert january.results[0].explanation["evidence"]["temporal"]["temporal_only_query"]
    recent = memory.search("owner", "What did I do most recently?", reference_time=NOW)
    assert _ids(recent)[0] == ids["flight"]


def test_temporal_intent_can_be_disabled_for_ablation(world) -> None:
    memory, ids = world
    response = memory.search(
        "owner", "Where does the user work now?", top_k=10, reference_time=NOW,
        config=SearchConfig(use_temporal_intent=False),
    )
    assert response.temporal_query.intent.value == "none"
    assert {ids["job_a"], ids["job_b"]} <= set(_ids(response))


def test_contradictory_memories_are_both_returned_and_flagged(world) -> None:
    memory, ids = world
    response = memory.search("owner", "Is the user allergic to peanuts?", reference_time=NOW)
    returned = _ids(response)
    assert ids["allergy_yes"] in returned and ids["allergy_no"] in returned
    flagged = {item.memory.id: item.explanation["conflicts_with"] for item in response.results}
    assert flagged[ids["allergy_yes"]] == [ids["allergy_no"]]
    without = memory.search(
        "owner", "peanuts", reference_time=NOW,
        config=replace(memory.search_engine.config, use_semantic=False,
                       expand_contradictions=False),
    )
    assert _ids(without) == [ids["allergy_yes"]]
    expanded = memory.search(
        "owner", "peanuts", reference_time=NOW,
        config=replace(memory.search_engine.config, use_semantic=False),
    )
    assert _ids(expanded) == [ids["allergy_yes"], ids["allergy_no"]]
    assert expanded.results[1].channels == ("contradiction",)


def test_irrelevant_queries_return_nothing(world) -> None:
    memory, _ = world
    assert memory.search("owner", "capital of Mars", reference_time=NOW).results == ()


def test_retrieval_is_deterministic_and_respects_top_k(world) -> None:
    memory, _ = world
    first = memory.search("owner", "user Lisbon water", top_k=3, reference_time=NOW)
    second = memory.search("owner", "user Lisbon water", top_k=3, reference_time=NOW)
    assert len(first.results) == 3
    assert first.explain() == second.explain()
    with pytest.raises(ValueError):
        memory.search("owner", "x", top_k=0)
    with pytest.raises(ValueError):
        memory.search("owner", "   ")
