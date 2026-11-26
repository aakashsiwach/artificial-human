from __future__ import annotations

from fastapi.testclient import TestClient

from artificial_human.api import create_app
from artificial_human.brain.providers.base import ChatMessage, Generation
from artificial_human.cognition.episodic_replay import EpisodicEvent
from artificial_human.organism import ArtificialOrganism


class StubProvider:
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> Generation:
        return Generation(content="Schema review response.", model="stub-model")


def _create_sequence(
    client: TestClient,
    organism: ArtificialOrganism,
    consent_id: str,
    session_id: str,
    *,
    provider: str,
    amount: int,
    start_minute: int,
) -> list[dict[str, object]]:
    events = (
        ("order.created", {"currency": "INR", "region": "IN"}),
        ("payment.authorized", {"provider": provider, "amount": amount}),
        ("order.completed", {"currency": "INR", "provider": provider}),
    )
    records: list[dict[str, object]] = []
    for index, (event_type, attributes) in enumerate(events):
        response = client.post("/memories", json={
            "kind": "episodic",
            "content": f"Owner-recorded event {event_type} in {session_id}",
            "source": "owner event record",
            "consent_id": consent_id,
            "occurred_at": f"2026-11-25T10:{start_minute + index:02d}:00+00:00",
            "metadata": {"event_type": event_type, "attributes": attributes},
        })
        assert response.status_code == 201
        records.append(response.json())

    replay = organism.cognitive_brain.architecture.episodic_memory
    replay.record_episode(
        [
            EpisodicEvent(
                str(record["id"]),
                str(record["occurred_at"]),
            )
            for record in records
        ],
        context_id=organism.cognitive_brain._associative_context_id(
            organism.owner_id, session_id
        ),
    )
    return records


def test_schema_api_preview_and_owner_approval_create_provenance_rich_procedural_memory(
    tmp_path,
) -> None:
    organism = ArtificialOrganism(
        owner_id="schema-api-owner",
        database_path=tmp_path / "schema-api.sqlite3",
        identity_path=tmp_path / "schema-api-identity.json",
        brain_provider=StubProvider(),
        embedding_provider=None,
    )
    client = TestClient(create_app(organism))
    consent = client.post("/consents", json={"purpose": "personal_memory"}).json()
    first = _create_sequence(
        client, organism, consent["id"], "session-a",
        provider="stripe", amount=100, start_minute=1,
    )
    second = _create_sequence(
        client, organism, consent["id"], "session-b",
        provider="razorpay", amount=175, start_minute=10,
    )

    preview = client.post("/knowledge/schemas/preview")
    assert preview.status_code == 200
    result = preview.json()
    assert len(result["proposals"]) == 1
    proposal = result["proposals"][0]
    assert proposal["event_types"] == [
        "order.created", "payment.authorized", "order.completed"
    ]
    assert proposal["support_count"] == 2
    assert proposal["schema"]["steps"][1]["attributes"]["provider"]["kind"] == "slot"
    assert len(client.get("/memories").json()) == len(first) + len(second)

    listed = client.get("/knowledge/schemas/proposals?status=pending_review").json()
    assert len(listed["proposals"]) == 1
    assert listed["proposals"][0]["id"] == proposal["id"]

    approved = client.post(
        f"/knowledge/schemas/proposals/{proposal['id']}/review",
        json={"decision": "approved", "note": "Repeated workflow reviewed."},
    )
    assert approved.status_code == 200
    body = approved.json()
    assert body["proposal"]["status"] == "approved"
    schema_memory = body["memory"]
    assert schema_memory["kind"] == "procedural"
    assert schema_memory["epistemic_status"] == "inferred"
    assert schema_memory["metadata"]["knowledge_type"] == "ordered_event_schema"
    assert schema_memory["metadata"]["schema_proposal_id"] == proposal["id"]
    assert schema_memory["metadata"]["schema_support_count"] == 2
    assert schema_memory["metadata"]["schema_source_sequences"] == proposal["source_sequences"]
    assert len(client.get("/memories").json()) == 7
    assert client.get("/knowledge/schemas/proposals?status=pending_review").json()["proposals"] == []


def test_schema_api_rejection_and_source_correction_prevent_persistence(tmp_path) -> None:
    organism = ArtificialOrganism(
        owner_id="schema-api-reject",
        database_path=tmp_path / "schema-api-reject.sqlite3",
        identity_path=tmp_path / "schema-api-reject-identity.json",
        brain_provider=StubProvider(),
        embedding_provider=None,
    )
    client = TestClient(create_app(organism))
    consent = client.post("/consents", json={"purpose": "personal_memory"}).json()
    _create_sequence(
        client, organism, consent["id"], "reject-a",
        provider="stripe", amount=100, start_minute=1,
    )
    _create_sequence(
        client, organism, consent["id"], "reject-b",
        provider="razorpay", amount=175, start_minute=10,
    )
    proposal = client.post("/knowledge/schemas/preview").json()["proposals"][0]
    before = len(client.get("/memories").json())

    rejected = client.post(
        f"/knowledge/schemas/proposals/{proposal['id']}/review",
        json={"decision": "rejected", "note": "Do not generalize this pattern."},
    )
    assert rejected.status_code == 200
    assert rejected.json()["proposal"]["status"] == "rejected"
    assert len(client.get("/memories").json()) == before

    # A correction to any source makes a pending proposal unavailable for review.
    corrected_first = _create_sequence(
        client, organism, consent["id"], "correct-a",
        provider="stripe", amount=100, start_minute=30,
    )
    _create_sequence(
        client, organism, consent["id"], "correct-b",
        provider="razorpay", amount=175, start_minute=40,
    )
    pending = client.post("/knowledge/schemas/preview").json()["proposals"][0]
    corrected = client.patch(
        f"/memories/{corrected_first[0]['id']}",
        json={"metadata": {"event_type": "order.cancelled", "attributes": {}}},
    )
    assert corrected.status_code == 200
    stale = client.post(
        f"/knowledge/schemas/proposals/{pending['id']}/review",
        json={"decision": "approved", "note": "Try stale proposal."},
    )
    assert stale.status_code == 404


def test_schema_proposals_are_cleared_when_consent_is_revoked(tmp_path) -> None:
    organism = ArtificialOrganism(
        owner_id="schema-api-consent",
        database_path=tmp_path / "schema-api-consent.sqlite3",
        identity_path=tmp_path / "schema-api-consent-identity.json",
        brain_provider=StubProvider(),
        embedding_provider=None,
    )
    client = TestClient(create_app(organism))
    consent = client.post("/consents", json={"purpose": "personal_memory"}).json()
    _create_sequence(
        client, organism, consent["id"], "consent-a",
        provider="stripe", amount=100, start_minute=1,
    )
    _create_sequence(
        client, organism, consent["id"], "consent-b",
        provider="razorpay", amount=175, start_minute=10,
    )
    proposal = client.post("/knowledge/schemas/preview").json()["proposals"][0]
    assert client.delete(f"/consents/{consent['id']}").status_code == 200
    assert client.get("/knowledge/schemas/proposals").json()["proposals"] == []
    assert client.post(
        f"/knowledge/schemas/proposals/{proposal['id']}/review",
        json={"decision": "approved", "note": "Revoked data cannot be used."},
    ).status_code == 404
