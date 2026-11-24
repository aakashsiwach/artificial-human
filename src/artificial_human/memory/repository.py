from __future__ import annotations

import builtins
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


@dataclass(frozen=True, slots=True)
class LearningExample:
    id: str
    owner_id: str
    task_category: str
    input_text: str
    target_text: str
    source: str
    consent_id: str
    status: str
    created_at: str
    reviewer: str | None = None
    review_note: str | None = None


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
                CREATE TABLE IF NOT EXISTS training_examples (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    task_category TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    target_text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    consent_id TEXT NOT NULL REFERENCES consents(id),
                    status TEXT NOT NULL DEFAULT 'pending_review',
                    created_at TEXT NOT NULL,
                    reviewer TEXT,
                    review_note TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_training_owner
                    ON training_examples(owner_id, status, created_at DESC);
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
            example_cursor = connection.execute(
                "DELETE FROM training_examples WHERE owner_id = ?", (owner_id,)
            )
            memory_cursor = connection.execute(
                "DELETE FROM memories WHERE owner_id = ?", (owner_id,)
            )
            connection.execute(
                "UPDATE consents SET active = 0, revoked_at = ? WHERE owner_id = ?",
                (now, owner_id),
            )
        return example_cursor.rowcount + memory_cursor.rowcount

    def export(self, owner_id: str) -> builtins.list[dict[str, Any]]:
        return [
            {
                "id": item.id, "owner_id": item.owner_id, "kind": item.kind.value,
                "content": item.content, "source": item.source, "consent_id": item.consent_id,
                "created_at": item.created_at, "metadata": item.metadata,
            }
            for item in self.list(owner_id, limit=500)
        ]


    def add_training_example(
        self, *, owner_id: str, task_category: str, input_text: str,
        target_text: str, source: str, consent_id: str,
    ) -> LearningExample:
        if not all(value.strip() for value in (task_category, input_text, target_text, source)):
            raise ValueError("task_category, input_text, target_text, and source are required")
        with self._connect() as connection:
            consent = connection.execute(
                """SELECT id FROM consents
                   WHERE id = ? AND owner_id = ? AND purpose = 'personal_learning' AND active = 1""",
                (consent_id, owner_id),
            ).fetchone()
            if consent is None:
                raise ConsentRequiredError("Active personal_learning consent is required")
            example = LearningExample(
                id=str(uuid4()), owner_id=owner_id, task_category=task_category,
                input_text=input_text, target_text=target_text, source=source,
                consent_id=consent_id, status="pending_review", created_at=self._now(),
            )
            connection.execute(
                """INSERT INTO training_examples
                   (id, owner_id, task_category, input_text, target_text, source,
                    consent_id, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (example.id, example.owner_id, example.task_category, example.input_text,
                 example.target_text, example.source, example.consent_id,
                 example.status, example.created_at),
            )
        return example

    def list_training_examples(
        self, owner_id: str, *, status: str | None = None, limit: int = 100
    ) -> builtins.list[LearningExample]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        if status not in (None, "pending_review", "approved", "rejected"):
            raise ValueError("status must be pending_review, approved, rejected, or omitted")
        query = """SELECT e.* FROM training_examples e
                   JOIN consents c ON c.id = e.consent_id
                   WHERE e.owner_id = ? AND c.active = 1"""
        values: builtins.list[Any] = [owner_id]
        if status is not None:
            query += " AND e.status = ?"
            values.append(status)
        query += " ORDER BY e.created_at DESC LIMIT ?"
        values.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [
            LearningExample(
                id=row["id"], owner_id=row["owner_id"], task_category=row["task_category"],
                input_text=row["input_text"], target_text=row["target_text"], source=row["source"],
                consent_id=row["consent_id"], status=row["status"], created_at=row["created_at"],
                reviewer=row["reviewer"], review_note=row["review_note"],
            )
            for row in rows
        ]

    def review_training_example(
        self, owner_id: str, example_id: str, *, decision: str, reviewer: str, note: str = ""
    ) -> bool:
        if decision not in ("approved", "rejected"):
            raise ValueError("decision must be approved or rejected")
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE training_examples
                   SET status = ?, reviewer = ?, review_note = ?
                   WHERE id = ? AND owner_id = ? AND status = 'pending_review'
                     AND consent_id IN (
                       SELECT id FROM consents WHERE active = 1 AND purpose = 'personal_learning'
                     )""",
                (decision, reviewer, note, example_id, owner_id),
            )
        return cursor.rowcount == 1

    def delete_training_example(self, owner_id: str, example_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM training_examples WHERE id = ? AND owner_id = ?",
                (example_id, owner_id),
            )
        return cursor.rowcount == 1

    def export_training_corpus(self, owner_id: str) -> builtins.list[dict[str, str]]:
        approved = self.list_training_examples(owner_id, status="approved", limit=500)
        return [
            {
                "id": item.id,
                "task_category": item.task_category,
                "input": item.input_text,
                "target": item.target_text,
                "source": item.source,
                "created_at": item.created_at,
            }
            for item in approved
        ]
