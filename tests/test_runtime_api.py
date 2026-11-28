from __future__ import annotations

import time

from fastapi.testclient import TestClient

from artificial_human.api import create_app
from artificial_human.organism import ArtificialOrganism
from artificial_human.runtime import RuntimeState


def test_api_lifespan_owns_and_stops_the_shared_runtime() -> None:
    organism = ArtificialOrganism(owner_id="runtime-api-owner", embedding_provider=None)
    app = create_app(organism)

    with TestClient(app) as client:
        status = client.get("/runtime/status")
        assert status.status_code == 200
        assert status.json()["state"] == RuntimeState.RUNNING.value

        event = client.post("/runtime/events", json={
            "modality": "vision",
            "values": [0.1, 0.4, 0.8],
            "timestamp": 1,
            "confidence": 0.9,
        })
        assert event.status_code == 202
        assert event.json()["accepted"] is True

        time.sleep(0.15)
        after_event = client.get("/runtime/status").json()
        assert after_event["processed_events"] >= 1
        assert after_event["organism"]["neural"]["simulation_step"] >= 2

        paused = client.post("/runtime/pause").json()
        assert paused["state"] == RuntimeState.PAUSED.value
        paused_at = paused["tick_count"]
        time.sleep(0.12)
        assert client.get("/runtime/status").json()["tick_count"] == paused_at

        resumed = client.post("/runtime/resume").json()
        assert resumed["state"] == RuntimeState.RUNNING.value
        assert organism.neural_dynamics.snapshot().idle_ticks == 0

    assert organism.runtime.state is RuntimeState.STOPPED


def test_api_development_clock_endpoint_does_not_depend_on_wall_clock() -> None:
    organism = ArtificialOrganism(owner_id="development-api-owner", embedding_provider=None)
    with TestClient(create_app(organism)) as client:
        before = client.get("/runtime/status").json()["organism"]["neural"]["simulation_step"]
        response = client.post("/runtime/development", json={"postnatal_units": 7})
        assert response.status_code == 200
        assert response.json()["postnatal_units"] == 507
        after = client.get("/runtime/status").json()["organism"]["neural"]["simulation_step"]
        assert after >= before
