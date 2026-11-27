from __future__ import annotations

import sqlite3
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from artificial_human.api import create_app
from artificial_human.core.models import MemoryKind
from artificial_human.memory.volatile_repository import InMemoryMemoryRepository
from artificial_human.organism import ArtificialOrganism


def test_active_organism_never_connects_to_sqlite_in_clean_process(tmp_path) -> None:
    database_path = tmp_path / "must-not-open.sqlite3"
    identity_path = tmp_path / "must-not-write.json"
    code = """
import sqlite3
import sys
from pathlib import Path

def forbidden_connect(*args, **kwargs):
    raise AssertionError("the active organism must never connect to SQLite")

sqlite3.connect = forbidden_connect
from artificial_human.organism import ArtificialOrganism
from artificial_human.core.models import MemoryKind

organism = ArtificialOrganism(
    owner_id="lazy-sqlite-owner",
    database_path=Path(sys.argv[1]),
    identity_path=Path(sys.argv[2]),
    embedding_provider=None,
)
grant = organism.memory.grant_consent("lazy-sqlite-owner")
organism.memory.add(
    owner_id="lazy-sqlite-owner",
    kind=MemoryKind.EPISODIC,
    content="RAM only",
    source="test",
    consent_id=grant.id,
)
assert organism.memory.list("lazy-sqlite-owner")[0].content == "RAM only"
assert Path(sys.argv[1]).exists() is False
assert Path(sys.argv[2]).exists() is False
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(database_path), str(identity_path)],
        capture_output=True,
        check=False,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert database_path.exists() is False
    assert identity_path.exists() is False


def test_default_organism_never_initialises_sqlite_or_disk_files(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "must-not-exist.sqlite3"
    identity_path = tmp_path / "must-not-exist.json"
    monkeypatch.setenv("ARTIFICIAL_HUMAN_DB_PATH", str(database_path))
    monkeypatch.setenv("ARTIFICIAL_HUMAN_IDENTITY_PATH", str(identity_path))
    monkeypatch.delenv("ARTIFICIAL_HUMAN_PUBLIC_DEPLOYMENT", raising=False)
    monkeypatch.delenv("ARTIFICIAL_HUMAN_STORAGE_MODE", raising=False)

    def forbidden_connect(*args, **kwargs):
        raise AssertionError("SQLite must not be opened by the active organism runtime")

    monkeypatch.setattr(sqlite3, "connect", forbidden_connect)
    organism = ArtificialOrganism(
        owner_id="ram-owner",
        database_path=database_path,
        identity_path=identity_path,
        embedding_provider=None,
    )

    assert isinstance(organism.memory, InMemoryMemoryRepository)
    assert organism.memory.path.as_posix() == "<in-memory>"
    assert organism.identity_repository is None
    assert organism.identity.display_name == "Artificial Human"
    assert database_path.exists() is False
    assert identity_path.exists() is False
    assert organism.status()["runtime"] == {
        "storage_mode": "in_memory_no_database",
        "external_database_connected": False,
        "personal_experience_persists_restart": False,
        "identity_source": "version_controlled_source_code",
    }


def test_api_memory_consent_search_and_revocation_use_only_ram(tmp_path, monkeypatch) -> None:
    def forbidden_connect(*args, **kwargs):
        raise AssertionError("SQLite must not be opened by the active organism runtime")

    monkeypatch.setattr(sqlite3, "connect", forbidden_connect)
    database_path = tmp_path / "api-must-not-exist.sqlite3"
    organism = ArtificialOrganism(
        owner_id="api-owner", database_path=database_path, embedding_provider=None
    )
    client = TestClient(create_app(organism))

    consent = client.post("/consents", json={"purpose": "personal_memory"})
    assert consent.status_code == 201
    memory = client.post("/memories", json={
        "kind": "episodic", "content": "A live in-memory event",
        "source": "explicit owner input", "consent_id": consent.json()["id"],
    })
    assert memory.status_code == 201
    assert len(client.get("/memories").json()) == 1
    result = client.post("/memories/search", json={"query": "live in-memory event"})
    assert result.status_code == 200
    assert result.json()["results"][0]["memory"]["id"] == memory.json()["id"]

    assert client.delete(f"/consents/{consent.json()['id']}").status_code == 200
    assert client.get("/memories").json() == []
    assert database_path.exists() is False


def test_memory_capacity_is_bounded_without_silent_eviction() -> None:
    repository = InMemoryMemoryRepository(max_records=2)
    consent = repository.grant_consent("owner")
    for content in ("one", "two"):
        repository.add(
            owner_id="owner", kind=MemoryKind.EPISODIC, content=content,
            source="test", consent_id=consent.id,
        )
    with pytest.raises(ValueError, match="capacity"):
        repository.add(
            owner_id="owner", kind=MemoryKind.EPISODIC, content="three",
            source="test", consent_id=consent.id,
        )
    assert [item.content for item in repository.list("owner")] == ["two", "one"]


def test_memory_reset_clears_all_runtime_state_and_reopens_repository() -> None:
    repository = InMemoryMemoryRepository()
    grant = repository.grant_consent("owner")
    repository.add(
        owner_id="owner", kind=MemoryKind.SEMANTIC, content="temporary",
        source="test", consent_id=grant.id,
    )
    repository.close()
    with pytest.raises(RuntimeError, match="closed"):
        repository.list("owner")
    repository.reset()
    assert repository.list("owner") == []
    assert repository.has_active_consent("owner", "personal_memory") is False
