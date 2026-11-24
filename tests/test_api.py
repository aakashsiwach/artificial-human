from fastapi.testclient import TestClient

from artificial_human.api import create_app
from artificial_human.organism import ArtificialOrganism


def test_local_api_consent_memory_and_immune_flow(tmp_path) -> None:
    organism = ArtificialOrganism(owner_id="test-owner", database_path=tmp_path / "api.sqlite3")
    client = TestClient(create_app(organism))

    assert client.get("/health").json()["status"] == "ok"
    denied = client.post("/memories", json={
        "kind": "episodic", "content": "event", "source": "test", "consent_id": "nope"
    })
    assert denied.status_code == 403

    grant = client.post("/consents", json={}).json()
    created = client.post("/memories", json={
        "kind": "episodic", "content": "First day building the organism",
        "source": "owner note", "consent_id": grant["id"]
    })
    assert created.status_code == 201
    assert len(client.get("/memories").json()) == 1

    signal = client.post("/signals/evaluate", json={
        "name": "api_error_rate", "value": 0.2, "source": "test"
    }).json()
    assert signal["incident"] is not None
    assert len(client.get("/repairs").json()) == 1

    assert client.delete(f"/consents/{grant['id']}").status_code == 200
    assert client.get("/memories").json() == []

def test_root_endpoint_explains_where_to_find_api_docs(tmp_path) -> None:
    organism = ArtificialOrganism(owner_id="root-test", database_path=tmp_path / "root.sqlite3")
    response = TestClient(create_app(organism)).get("/")

    assert response.status_code == 200
    assert response.json()["service"] == "Artificial Human"
    assert response.json()["docs"] == "/docs"
    assert response.json()["health"] == "/health"
