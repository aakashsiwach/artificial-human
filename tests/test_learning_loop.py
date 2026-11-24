from fastapi.testclient import TestClient
import pytest

from artificial_human.api import create_app
from artificial_human.core.models import MemoryKind
from artificial_human.memory.repository import ConsentRequiredError, MemoryRepository
from artificial_human.organism import ArtificialOrganism


def test_learning_examples_require_separate_consent_and_owner_review(tmp_path) -> None:
    repository = MemoryRepository(tmp_path / "learning.sqlite3")
    memory_consent = repository.grant_consent("owner", purpose="personal_memory")
    with pytest.raises(ConsentRequiredError):
        repository.add_training_example(
            owner_id="owner",
            task_category="communication",
            input_text="Write a status update",
            target_text="A concise status update",
            source="owner correction",
            consent_id=memory_consent.id,
        )

    learning_consent = repository.grant_consent("owner", purpose="personal_learning")
    example = repository.add_training_example(
        owner_id="owner",
        task_category="communication",
        input_text="Write a status update",
        target_text="A concise status update",
        source="owner correction",
        consent_id=learning_consent.id,
    )
    assert example.status == "pending_review"
    assert repository.export_training_corpus("owner") == []

    assert repository.review_training_example(
        "owner", example.id, decision="approved", reviewer="owner"
    )
    corpus = repository.export_training_corpus("owner")
    assert len(corpus) == 1
    assert corpus[0]["target"] == "A concise status update"

    assert repository.revoke_consent("owner", learning_consent.id)
    assert repository.export_training_corpus("owner") == []


def test_api_exposes_reviewed_training_corpus_only(tmp_path) -> None:
    organism = ArtificialOrganism(
        owner_id="test-owner", database_path=tmp_path / "learning-api.sqlite3"
    )
    client = TestClient(create_app(organism))
    consent = client.post("/consents", json={"purpose": "personal_learning"}).json()

    created = client.post("/training/examples", json={
        "task_category": "decision",
        "input_text": "Choose between options A and B",
        "target_text": "Choose A because it best meets the stated constraints",
        "source": "owner-provided example",
        "consent_id": consent["id"],
    })
    assert created.status_code == 201
    example_id = created.json()["id"]
    assert client.get("/training/corpus").json()["count"] == 0

    reviewed = client.post(
        f"/training/examples/{example_id}/review",
        json={"decision": "approved", "note": "Matches my intended reasoning"},
    )
    assert reviewed.status_code == 200
    corpus = client.get("/training/corpus").json()
    assert corpus["count"] == 1
    assert corpus["examples"][0]["input"] == "Choose between options A and B"

    assert client.delete(f"/training/examples/{example_id}").json()["deleted"] is True
    assert client.get("/training/corpus").json()["count"] == 0
