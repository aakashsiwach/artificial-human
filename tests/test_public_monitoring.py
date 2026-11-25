from fastapi.testclient import TestClient

from artificial_human.api import create_app
from artificial_human.organism import ArtificialOrganism


def _client(tmp_path):
    organism = ArtificialOrganism(
        owner_id="monitor-test",
        database_path=tmp_path / "monitor.sqlite3",
    )
    return TestClient(create_app(organism))


def test_monitor_overview_returns_only_aggregate_development_state(tmp_path) -> None:
    client = _client(tmp_path)

    response = client.get("/monitor/overview?days=280&seed=7")

    assert response.status_code == 200
    payload = response.json()
    prenatal = payload["development"]["prenatal"]
    assert prenatal["age_days"] == 280
    assert prenatal["gestational_week"] == 40
    assert prenatal["population_count"] == 7
    assert prenatal["projection_count"] == 10
    assert len(prenatal["populations"]) == 7
    assert len(prenatal["projections"]) == 10
    assert "owner_id" not in payload["organism"]
    assert "private prompt" not in response.text
    assert "memory content" not in response.text
    assert "cognitive_systems" in payload


def test_public_deployment_requires_bearer_token_and_hides_docs(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("ARTIFICIAL_HUMAN_PUBLIC_DEPLOYMENT", "true")
    monkeypatch.setenv("ARTIFICIAL_HUMAN_API_TOKEN", "monitor-test-secret")
    monkeypatch.setenv(
        "ARTIFICIAL_HUMAN_CORS_ORIGINS",
        "https://monitor.example.test",
    )
    client = _client(tmp_path)

    assert client.get("/health").status_code == 200
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/monitor/overview").status_code == 401
    assert client.get("/status").status_code == 401

    headers = {"Authorization": "Bearer monitor-test-secret"}
    assert client.get("/monitor/overview", headers=headers).status_code == 200
    assert client.get("/status", headers=headers).status_code == 200


def test_public_deployment_fails_closed_when_token_is_missing(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("ARTIFICIAL_HUMAN_PUBLIC_DEPLOYMENT", "true")
    monkeypatch.delenv("ARTIFICIAL_HUMAN_API_TOKEN", raising=False)
    client = _client(tmp_path)

    assert client.get("/health").status_code == 200
    assert client.get("/monitor/overview").status_code == 503
