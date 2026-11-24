from __future__ import annotations

import sqlite3

import pytest

from artificial_human.memory import repository as repository_module
from artificial_human.memory.repository import SCHEMA_VERSION, MemoryRepository

# Exact schema shipped before versioned migrations (milestone 1).
_LEGACY_SCHEMA = """
CREATE TABLE consents (
    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, purpose TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL, revoked_at TEXT
);
CREATE INDEX idx_consents_owner ON consents(owner_id, purpose, active);
CREATE TABLE memories (
    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, kind TEXT NOT NULL, content TEXT NOT NULL,
    source TEXT NOT NULL, consent_id TEXT NOT NULL REFERENCES consents(id),
    created_at TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX idx_memories_owner ON memories(owner_id, created_at DESC);
CREATE TABLE training_examples (
    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, task_category TEXT NOT NULL,
    input_text TEXT NOT NULL, target_text TEXT NOT NULL, source TEXT NOT NULL,
    consent_id TEXT NOT NULL REFERENCES consents(id),
    status TEXT NOT NULL DEFAULT 'pending_review', created_at TEXT NOT NULL,
    reviewer TEXT, review_note TEXT
);
INSERT INTO consents VALUES ('c1', 'owner', 'personal_memory', 1, '2026-01-01T00:00:00+00:00', NULL);
INSERT INTO memories VALUES ('m1', 'owner', 'preference', 'Prefers concise answers',
    'owner statement', 'c1', '2026-01-02T00:00:00+00:00', '{"salience": 0.5}');
INSERT INTO consents VALUES ('c2', 'owner', 'personal_learning', 1, '2026-01-01T00:00:00+00:00', NULL);
INSERT INTO training_examples VALUES ('e1', 'owner', 'email', 'in', 'out', 'manual', 'c2',
    'approved', '2026-01-03T00:00:00+00:00', 'owner', '');
"""


def _legacy_database(path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(_LEGACY_SCHEMA)
    connection.commit()
    connection.close()


def test_legacy_database_migrates_without_losing_records(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    _legacy_database(path)

    repository = MemoryRepository(path)

    assert repository.schema_version() == SCHEMA_VERSION
    [record] = repository.list("owner")
    assert (record.id, record.content, record.source) == (
        "m1", "Prefers concise answers", "owner statement"
    )
    assert record.metadata == {"salience": 0.5}
    assert record.created_at == "2026-01-02T00:00:00+00:00"
    assert record.occurred_at is None and record.valid_from is None
    assert record.epistemic_status == "asserted"
    assert repository.list_training_examples("owner")[0].id == "e1"
    connection = sqlite3.connect(path)
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
    connection.close()
    assert {"memory_links", "memory_embeddings", "memory_index_failures"} <= tables


def test_migration_is_idempotent_when_reopened(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    _legacy_database(path)
    MemoryRepository(path)
    reopened = MemoryRepository(path)
    assert reopened.schema_version() == SCHEMA_VERSION
    assert [item.id for item in reopened.list("owner")] == ["m1"]


def test_new_database_starts_at_current_schema(tmp_path) -> None:
    assert MemoryRepository(tmp_path / "new.sqlite3").schema_version() == SCHEMA_VERSION


def test_failed_migration_rolls_back_and_preserves_previous_version(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "legacy.sqlite3"
    _legacy_database(path)
    MemoryRepository(path)

    def broken(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE half_applied (id TEXT)")
        raise RuntimeError("simulated migration failure")

    monkeypatch.setattr(
        repository_module, "MIGRATIONS", (*repository_module.MIGRATIONS, (99, broken))
    )
    monkeypatch.setattr(repository_module, "SCHEMA_VERSION", 99)
    with pytest.raises(RuntimeError, match="simulated"):
        MemoryRepository(path)

    connection = sqlite3.connect(path)
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
    connection.close()
    assert version == SCHEMA_VERSION
    assert "half_applied" not in tables


def test_refuses_to_open_a_newer_schema(tmp_path) -> None:
    path = tmp_path / "future.sqlite3"
    MemoryRepository(path)
    connection = sqlite3.connect(path)
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    connection.commit()
    connection.close()
    with pytest.raises(RuntimeError, match="newer"):
        MemoryRepository(path)
