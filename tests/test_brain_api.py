from fastapi.testclient import TestClient

from artificial_human.brain.providers.base import ChatMessage, Generation
from artificial_human.api import create_app
from artificial_human.organism import ArtificialOrganism


class StubProvider:
    def generate(
        self, messages: list[ChatMessage], *, temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> Generation:
        return Generation(
            content="I can support that with the available evidence.",
            model="stub-model",
        )


def test_brain_status_reports_missing_model_without_exposing_secrets(tmp_path) -> None:
    organism = ArtificialOrganism(
        owner_id="owner",
        database_path=tmp_path / "status.sqlite3",
        identity_path=tmp_path / "identity.json",
    )
    client = TestClient(create_app(organism))
    response = client.get("/brain/status")
    assert response.status_code == 200
    assert response.json()["configured"] is False
    assert "api_key" not in response.json()


def test_brain_endpoint_uses_consent_memory_and_returns_evidence(tmp_path) -> None:
    organism = ArtificialOrganism(
        owner_id="owner",
        database_path=tmp_path / "brain.sqlite3",
        identity_path=tmp_path / "identity.json",
        brain_provider=StubProvider(),
    )
    client = TestClient(create_app(organism))

    profile = client.put("/identity", json={
        "display_name": "Aakash",
        "description": "Software engineer",
        "principles": ["Be honest about uncertainty."],
        "boundaries": ["Do not send anything without approval."],
        "communication_preferences": {"tone": "natural"},
    })
    assert profile.status_code == 200
    assert client.get("/identity").json()["display_name"] == "Aakash"

    consent = client.post("/consents", json={"purpose": "personal_memory"}).json()
    memory = client.post("/memories", json={
        "kind": "preference",
        "content": "I prefer concise technical explanations.",
        "source": "owner statement",
        "consent_id": consent["id"],
    })
    assert memory.status_code == 201

    response = client.post("/brain/respond", json={
        "message": "How do I prefer technical explanations?",
        "mode": "assistant",
    })
    assert response.status_code == 200
    assert response.json()["model"] == "stub-model"
    assert response.json()["conversation_saved"] is False
    assert response.json()["recalled_memories"][0]["memory_id"] == memory.json()["id"]
    assert len(response.json()["plan_steps"]) == 4
    assert response.json()["cognitive_trace"]["architecture"] == "neuroscience_informed_cognitive_cycle_v1"
    assert response.json()["cognitive_trace"]["perception"]["modality"] == "language"
    assert response.json()["working_memory_persisted"] is False


def test_brain_endpoint_returns_service_unavailable_without_provider(tmp_path) -> None:
    organism = ArtificialOrganism(
        owner_id="owner",
        database_path=tmp_path / "not-configured.sqlite3",
        identity_path=tmp_path / "identity.json",
    )
    response = TestClient(create_app(organism)).post(
        "/brain/respond", json={"message": "Hello"}
    )
    assert response.status_code == 503
    assert "ARTIFICIAL_HUMAN_LLM_MODEL" in response.json()["detail"]


def test_revoking_memory_consent_removes_memory_from_brain_context(tmp_path) -> None:
    organism = ArtificialOrganism(
        owner_id="owner",
        database_path=tmp_path / "revoked.sqlite3",
        identity_path=tmp_path / "identity.json",
        brain_provider=StubProvider(),
    )
    client = TestClient(create_app(organism))
    consent = client.post("/consents", json={"purpose": "personal_memory"}).json()
    client.post("/memories", json={
        "kind": "semantic",
        "content": "The owner's preferred editor is VS Code.",
        "source": "owner statement",
        "consent_id": consent["id"],
    })
    assert client.delete(f"/consents/{consent['id']}").status_code == 200

    response = client.post("/brain/respond", json={
        "message": "Which editor do I prefer?"
    })
    assert response.status_code == 200
    assert response.json()["recalled_memories"] == []


def test_style_evaluation_endpoint_returns_reference_metrics_without_saving_chat(tmp_path) -> None:
    organism = ArtificialOrganism(
        owner_id="owner",
        database_path=tmp_path / "evaluation.sqlite3",
        identity_path=tmp_path / "identity.json",
        brain_provider=StubProvider(),
    )
    client = TestClient(create_app(organism))
    response = client.post("/brain/evaluate-style", json={
        "message": "Explain this concept in my preferred style.",
        "reference_response": "I can support that with the available evidence.",
    })

    assert response.status_code == 200
    assert response.json()["mode"] == "style_clone"
    assert response.json()["reference_metrics"]["exact_match"] is True
    assert response.json()["reference_metrics"]["token_f1"] == 1.0
    assert response.json()["conversation_saved"] is False
    assert len(response.json()["plan_steps"]) == 4



def test_brain_session_retains_only_volatile_working_memory_and_can_be_cleared(tmp_path) -> None:
    organism = ArtificialOrganism(
        owner_id="owner",
        database_path=tmp_path / "session.sqlite3",
        identity_path=tmp_path / "identity.json",
        brain_provider=StubProvider(),
    )
    client = TestClient(create_app(organism))
    first = client.post("/brain/respond", json={
        "message": "The hippocampus is involved in episodic memory.",
        "session_id": "research-session",
    })
    assert first.status_code == 200
    assert first.json()["working_memory_session"] == "research-session"
    assert first.json()["working_memory_persisted"] is False

    second = client.post("/brain/respond", json={
        "message": "What was the brain structure I just mentioned?",
        "session_id": "research-session",
    })
    assert second.status_code == 200
    assert second.json()["cognitive_trace"]["working_memory"]["item_count"] >= 1

    cleared = client.delete("/brain/sessions/research-session")
    assert cleared.status_code == 200
    assert cleared.json()["cleared"] is True
    assert client.delete("/brain/sessions/research-session").json()["cleared"] is False



def test_cognitive_status_reports_component_boundaries(tmp_path) -> None:
    organism = ArtificialOrganism(
        owner_id="owner",
        database_path=tmp_path / "cognitive-status.sqlite3",
        identity_path=tmp_path / "identity.json",
        brain_provider=StubProvider(),
    )
    response = TestClient(create_app(organism)).get("/brain/cognition/status")
    assert response.status_code == 200
    assert response.json()["perception"] == "explicit_language_input"
    assert response.json()["working_memory"] == "capacity_limited_volatile_workspace"


def test_revoking_memory_consent_clears_volatile_working_memory(tmp_path) -> None:
    organism = ArtificialOrganism(
        owner_id="owner",
        database_path=tmp_path / "session-consent.sqlite3",
        identity_path=tmp_path / "identity.json",
        brain_provider=StubProvider(),
    )
    client = TestClient(create_app(organism))
    consent = client.post("/consents", json={"purpose": "personal_memory"}).json()
    memory = client.post("/memories", json={
        "kind": "preference",
        "content": "My preferred code editor is VS Code.",
        "source": "owner statement",
        "consent_id": consent["id"],
    })
    assert memory.status_code == 201

    first = client.post("/brain/respond", json={
        "message": "Which code editor do I prefer?",
        "session_id": "consent-session",
    })
    assert first.status_code == 200
    assert any(
        item["kind"] == "preference"
        for item in first.json()["cognitive_trace"]["working_memory"]["items"]
    )

    revoked = client.delete(f"/consents/{consent['id']}")
    assert revoked.status_code == 200

    second = client.post("/brain/respond", json={
        "message": "Do you remember my preferred editor?",
        "session_id": "consent-session",
    })
    assert second.status_code == 200
    assert all(
        item["kind"] != "preference"
        for item in second.json()["cognitive_trace"]["working_memory"]["items"]
    )
