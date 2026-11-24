from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from artificial_human.core.models import MemoryKind


class ConsentRequiredError(PermissionError):
    pass


@dataclass(frozen=True, slots=True)
class ConsentGrant:
    id: str
    owner_id: str
    purpose: str
    active: bool
    created_at: str
    revoked_at: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str
    owner_id: str
    kind: MemoryKind
    content: str
    source: str
    consent_id: str
    created_at: str
    metadata: dict[str, Any]


class MemoryRepository:
    """SQLite memory store; personal-memory writes require active owner consent."""

    def __init__(self, path: str | Path | None = None) -> None:
        configured = os.getenv("ARTIFICIAL_HUMAN_DB_PATH")
        raw_path = str(path or configured or ".artificial-human/organism.sqlite3")
        self._memory_uri: str | None = None
        self._keeper: sqlite3.Connection | None = None
        if raw_path == ":memory:":
            self._memory_uri = f"file:ah-{uuid4().hex}?mode=memory&cache=shared"
            self._keeper = sqlite3.connect(self._memory_uri, uri=True, check_same_thread=False)
            self.path = Path(":memory:")
        else:
            self.path = Path(raw_path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self._memory_uri:
            connection = sqlite3.connect(self._memory_uri, uri=True, timeout=10)
        else:
            connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS consents (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    revoked_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_consents_owner
                    ON consents(owner_id, purpose, active);
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    consent_id TEXT NOT NULL REFERENCES consents(id),
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_memories_owner
                    ON memories(owner_id, created_at DESC);
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def grant_consent(self, owner_id: str, purpose: str = "personal_memory") -> ConsentGrant:
        if not owner_id.strip() or not purpose.strip():
            raise ValueError("owner_id and purpose are required")
        grant = ConsentGrant(
            id=str(uuid4()), owner_id=owner_id, purpose=purpose,
            active=True, created_at=self._now(),
        )
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO consents(id, owner_id, purpose, active, created_at)
                   VALUES (?, ?, ?, 1, ?)""",
                (grant.id, grant.owner_id, grant.purpose, grant.created_at),
            )
        return grant

    def revoke_consent(self, owner_id: str, consent_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE consents SET active = 0, revoked_at = ?
                   WHERE id = ? AND owner_id = ? AND active = 1""",
                (self._now(), consent_id, owner_id),
            )
        return cursor.rowcount == 1

    def add(
        self, *, owner_id: str, kind: MemoryKind, content: str, source: str,
        consent_id: str, metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        if not content.strip():
            raise ValueError("Memory content cannot be empty")
        if not source.strip():
            raise ValueError("Memory source is required for provenance")
        with self._connect() as connection:
            consent = connection.execute(
                """SELECT id FROM consents
                   WHERE id = ? AND owner_id = ? AND purpose = 'personal_memory' AND active = 1""",
                (consent_id, owner_id),
            ).fetchone()
            if consent is None:
                raise ConsentRequiredError("Active personal_memory consent is required")
            record = MemoryRecord(
                id=str(uuid4()), owner_id=owner_id, kind=kind, content=content,
                source=source, consent_id=consent_id, created_at=self._now(),
                metadata=metadata or {},
            )
            connection.execute(
                """INSERT INTO memories
                   (id, owner_id, kind, content, source, consent_id, created_at, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (record.id, record.owner_id, record.kind.value, record.content,
                 record.source, record.consent_id, record.created_at,
                 json.dumps(record.metadata, ensure_ascii=False)),
            )
        return record

    def list(self, owner_id: str, *, limit: int = 100) -> list[MemoryRecord]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT m.* FROM memories m
                   JOIN consents c ON c.id = m.consent_id
                   WHERE m.owner_id = ? AND c.active = 1
                   ORDER BY m.created_at DESC LIMIT ?""",
                (owner_id, limit),
            ).fetchall()
        return [
            MemoryRecord(
                id=row["id"], owner_id=row["owner_id"], kind=MemoryKind(row["kind"]),
                content=row["content"], source=row["source"], consent_id=row["consent_id"],
                created_at=row["created_at"], metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]

    def delete(self, owner_id: str, memory_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM memories WHERE id = ? AND owner_id = ?", (memory_id, owner_id)
            )
        return cursor.rowcount == 1

    def erase_owner(self, owner_id: str) -> int:
        """Delete all memories and revoke all consent grants for this owner."""
        now = self._now()
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM memories WHERE owner_id = ?", (owner_id,))
            connection.execute(
                "UPDATE consents SET active = 0, revoked_at = ? WHERE owner_id = ?",
                (now, owner_id),
            )
        return cursor.rowcount

    def export(self, owner_id: str) -> list[dict[str, Any]]:
        return [
            {
                "id": item.id, "owner_id": item.owner_id, "kind": item.kind.value,
                "content": item.content, "source": item.source, "consent_id": item.consent_id,
                "created_at": item.created_at, "metadata": item.metadata,
            }
            for item in self.list(owner_id, limit=500)
        ]
