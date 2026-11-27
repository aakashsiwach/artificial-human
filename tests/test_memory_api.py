from __future__ import annotations

import pytest
from conftest import ConceptEmbedding
from fastapi.testclient import TestClient

from artificial_human.api import create_app
from artificial_human.brain.providers.base import ChatMessage, Generation
from artificial_human.organism import ArtificialOrganism


class RecordingProvider:
    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    def generate(
        self, messages: list[ChatMessage], *, temperature: float = 0.2, max_tokens: int = 1200
    ) -> Generation:
        self.messages = messages
        return Generation(content="Answer grounded in the supplied evidence.", model="stub")


@pytest.fixture
def setup(tmp_path):
    provider = RecordingProvider()
    organism = ArtificialOrganism(
        owner_id="owner", database_path=tmp_path / "api.sqlite3",
        identity_path=tmp_path / "identity.json", brain_provider=provider,
        embedding_provider=ConceptEmbedding(),
    )
    client = TestClient(create_app(organism))
    consent = client.post("/consents", json={"purpose": "personal_memory"}).json()["id"]
    return client, consent, provider, organism


def _add(client: TestClient, consent: str, **fields) -> dict:
    payload = {"kind": "semantic", "source": "owner statement", "consent_id": consent, **fields}
    response = client.post("/memories", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def test_create_memory_with_temporal_fields_is_indexed_and_backward_compatible(setup) -> None:
    client, consent, _, _ = setup
    legacy = _add(client, consent, content="Plain memory without temporal fields")
    assert {"id", "kind", "content", "source", "consent_id", "created_at", "metadata"} <= set(
        legacy
    )
    assert legacy["index"] == {"status": "indexed", "reason": None}
    timed = _add(
        client, consent, kind="episodic", content="Flew to Lisbon",
        occurred_at="2026-04-07T10:00:00+01:00",
    )
    assert timed["occurred_at"] == "2026-04-07T09:00:00.000000+00:00"
    assert timed["recorded_at"] == timed["created_at"]
    naive = client.post("/memories", json={
        "kind": "semantic", "content": "x", "source": "s", "consent_id": consent,
        "valid_from": "2026-01-01T00:00:00",
    })
    assert naive.status_code == 400
    listed = client.get("/memories").json()
    assert {item["id"] for item in listed} == {legacy["id"], timed["id"]}
    assert "occurred_at" in listed[0]


def test_search_endpoint_returns_provenance_and_explanations(setup) -> None:
    client, consent, _, _ = setup
    water = _add(client, consent, content="Water boils at 100 degrees Celsius.")
    _add(client, consent, kind="preference", content="I drink green tea.")
    response = client.post("/memories/search", json={"query": "At what heat does H2O steam?"})
    assert response.status_code == 200
    body = response.json()
    assert body["strategy"].startswith("lexical+semantic")
    top = body["results"][0]
    assert top["memory"]["id"] == water["id"]
    assert top["memory"]["source"] == "owner statement"
    assert top["explanation"]["fusion"]["method"] == "weighted_reciprocal_rank_fusion"
    assert "semantic" in top["explanation"]["evidence"]
    hidden = client.post(
        "/memories/search", json={"query": "water boils", "include_content": False}
    ).json()
    assert "content" not in hidden["results"][0]["memory"]
    filtered = client.post(
        "/memories/search", json={"query": "water boils", "kinds": ["preference"]}
    ).json()
    assert filtered["results"] == []
    bad = client.post("/memories/search", json={"query": "x", "valid_at": "yesterday"})
    assert bad.status_code == 400


def test_supersede_inspect_and_temporal_search(setup) -> None:
    client, consent, _, _ = setup
    old = _add(
        client, consent, content="The user works at Company A.", fact_key="owner.employer",
        valid_from="2025-01-01T00:00:00Z",
    )
    superseded = client.post(f"/memories/{old['id']}/supersede", json={
        "content": "The user moved to Company B.", "source": "owner statement 2026",
        "consent_id": consent, "valid_from": "2026-03-01T00:00:00Z",
    })
    assert superseded.status_code == 201
    new_id = superseded.json()["memory"]["id"]
    assert superseded.json()["link"]["relation"] == "supersedes"
    again = client.post(f"/memories/{old['id']}/supersede", json={
        "content": "Branch", "source": "s", "consent_id": consent,
    })
    assert again.status_code == 409

    inspected = client.get(f"/memories/{old['id']}").json()
    assert [item["id"] for item in inspected["versions"]] == [old["id"], new_id]
    assert inspected["memory"]["content"] == "The user works at Company A."
    assert inspected["semantic_index"]["indexed"] is True

    reference = "2026-06-15T00:00:00Z"
    now = client.post("/memories/search", json={
        "query": "Where does the user work now?", "reference_time": reference,
    }).json()
    assert now["results"][0]["memory"]["id"] == new_id
    assert now["temporal_query"]["intent"] == "current"
    past = client.post("/memories/search", json={
        "query": "Where did the user work previously?", "reference_time": reference,
    }).json()
    assert past["results"][0]["memory"]["id"] == old["id"]
    assert past["results"][0]["explanation"]["temporal"]["status"] == "superseded"
    as_of = client.post("/memories/search", json={
        "query": "user company", "valid_at": "2025-06-01T00:00:00Z",
    }).json()
    assert [item["memory"]["id"] for item in as_of["results"]] == [old["id"]]


def test_timeline_correction_contradiction_and_404s(setup) -> None:
    client, consent, _, _ = setup
    later = _add(client, consent, kind="episodic", content="Concert",
                 occurred_at="2026-05-30T20:00:00Z")
    earlier = _add(client, consent, kind="episodic", content="Marathon",
                   occurred_at="2026-02-15T07:00:00Z")
    timeline = client.get("/memories/timeline", params={"kind": "episodic"}).json()
    assert [item["memory"]["id"] for item in timeline] == [earlier["id"], later["id"]]
    window = client.get("/memories/timeline", params={
        "start": "2026-05-01T00:00:00Z", "end": "2026-06-01T00:00:00Z",
    }).json()
    assert [item["memory"]["id"] for item in window] == [later["id"]]

    corrected = client.patch(f"/memories/{earlier['id']}", json={"content": "Half marathon"})
    assert corrected.status_code == 200
    assert corrected.json()["content"] == "Half marathon"
    assert corrected.json()["updated_at"] is not None
    assert corrected.json()["index"]["status"] == "indexed"
    assert client.patch(f"/memories/{earlier['id']}", json={"content": None}).status_code == 400

    contradiction = client.post(
        f"/memories/{earlier['id']}/contradictions", json={"other_memory_id": later["id"]}
    )
    assert contradiction.status_code == 201
    assert contradiction.json()["relation"] == "contradicts"
    exported = client.get("/memories/export").json()
    assert any(item["links"] for item in exported)
    assert all(item["embeddings"] for item in exported)
    assert all("vector" not in str(item["embeddings"]) for item in exported)

    assert client.get("/memories/missing").status_code == 404
    assert client.patch("/memories/missing", json={"content": "x"}).status_code == 404
    assert client.post("/memories/missing/contradictions",
                       json={"other_memory_id": later["id"]}).status_code == 404


def test_index_status_rebuild_and_deletion(setup) -> None:
    client, consent, _, organism = setup
    record = _add(client, consent, content="Water boils.")
    status = client.get("/memories/index/status").json()
    assert status["retrieval_mode"] == "hybrid"
    assert status["indexed_memories"] == 1
    assert status["schema_version"] == 0  # RAM-only adapter has no database schema
    rebuilt = client.post("/memories/index/rebuild", json={"full": True}).json()
    assert rebuilt["indexed"] == 1
    assert client.delete(f"/memories/{record['id']}").status_code == 200
    assert client.get("/memories/index/status").json()["indexed_memories"] == 0
    assert organism.memory.embeddings("owner", "test:test-concepts:10") == {}


def test_consent_revocation_removes_memory_and_vectors_from_search(setup) -> None:
    client, consent, _, organism = setup
    _add(client, consent, content="Water boils.")
    assert client.delete(f"/consents/{consent}").status_code == 200
    assert client.post("/memories/search", json={"query": "water"}).json()["results"] == []
    assert organism.memory.embeddings("owner", "test:test-concepts:10") == {}
    assert client.get("/memories/index/status").json()["eligible_memories"] == 0


def test_brain_cycle_uses_hybrid_temporal_memory_and_traces_retrieval(setup) -> None:
    client, consent, provider, _ = setup
    old = _add(client, consent, content="The user works at Company A.",
               valid_from="2025-01-01T00:00:00Z")
    new = client.post(f"/memories/{old['id']}/supersede", json={
        "content": "The user moved to Company B.", "source": "owner", "consent_id": consent,
        "valid_from": "2026-03-01T00:00:00Z",
    }).json()["memory"]
    _add(client, consent, content="Water boils at 100 degrees Celsius.")

    response = client.post("/brain/respond", json={"message": "Where does the user work now?"})
    assert response.status_code == 200
    body = response.json()
    assert [item["memory_id"] for item in body["recalled_memories"]] == [new["id"]]
    assert body["recalled_memories"][0]["temporal_status"] == "valid"
    retrieval = body["cognitive_trace"]["memory"]["retrieval"]
    assert retrieval["temporal_query"]["intent"] == "current"
    assert retrieval["results"][0]["memory_id"] == new["id"]
    selected = body["cognitive_trace"]["attention"]["selected"]
    memory_items = [item for item in selected if item["kind"] == "memory"]
    assert memory_items and memory_items[0]["retrieval_channels"]
    assert body["conversation_saved"] is False
    # Long-term memory is not written by conversation.
    assert len(client.get("/memories").json()) == 3
    assert "superseded or expired memory describes the past" in provider.messages[0].content

    paraphrase = client.post("/brain/respond", json={"message": "At what heat does H2O steam?"})
    channels = paraphrase.json()["recalled_memories"][0]["retrieval_channels"]
    assert "semantic" in channels


def test_cognitive_status_reports_hybrid_memory(setup) -> None:
    client, _, _, _ = setup
    status = client.get("/brain/cognition/status").json()
    assert status["memory"] == "consent_filtered_hybrid_lexical_semantic_temporal_retrieval"
    assert client.get("/status").json()["systems"]["memory_retrieval"] == (
        "hybrid_lexical_semantic_temporal"
    )
