from __future__ import annotations

import builtins
import hashlib
import json
import os
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal
from uuid import uuid4

from artificial_human.core.models import MemoryKind
from artificial_human.memory.temporal import optional_utc_iso


class ConsentRequiredError(PermissionError):
    pass


class MemoryNotFoundError(KeyError):
    """The memory does not exist for this owner or its consent is inactive."""


EpistemicStatus = Literal["asserted", "inferred"]
EPISTEMIC_STATUSES: Final = ("asserted", "inferred")
LinkRelation = Literal["supersedes", "contradicts"]

SCHEMA_VERSION: Final = 2


class _Unset:
    """Sentinel distinguishing 'not supplied' from an explicit ``None``."""

    def __repr__(self) -> str:
        return "UNSET"


UNSET: Final = _Unset()


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
    occurred_at: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    fact_key: str | None = None
    epistemic_status: str = "asserted"
    updated_at: str | None = None

    @property
    def recorded_at(self) -> str:
        """When the system learned this information (transaction time)."""
        return self.created_at


@dataclass(frozen=True, slots=True)
class MemoryLink:
    id: str
    owner_id: str
    from_memory_id: str
    to_memory_id: str
    relation: str
    created_at: str
    effective_at: str | None = None
    note: str = ""


@dataclass(frozen=True, slots=True)
class StoredEmbedding:
    memory_id: str
    space: str
    model_id: str
    model_version: str
    dimensions: int
    content_hash: str
    vector: bytes
    created_at: str


@dataclass(frozen=True, slots=True)
class IndexCounts:
    eligible: int
    indexed: int
    stale: int
    failed: int
    other_spaces: int
    failures: tuple[dict[str, str], ...] = field(default_factory=tuple)


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


def content_fingerprint(content: str) -> str:
    """Hash of indexed text so stale vectors can never be attributed to new content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


_BASELINE_SCHEMA: Final = """
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

_TEMPORAL_COLUMNS: Final = (
    ("occurred_at", "TEXT"),
    ("valid_from", "TEXT"),
    ("valid_to", "TEXT"),
    ("fact_key", "TEXT"),
    ("epistemic_status", "TEXT NOT NULL DEFAULT 'asserted'"),
    ("updated_at", "TEXT"),
)

_MEMORY_V2_TABLES: Final = (
    """CREATE TABLE IF NOT EXISTS memory_links (
        id TEXT PRIMARY KEY,
        owner_id TEXT NOT NULL,
        from_memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
        to_memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
        relation TEXT NOT NULL CHECK (relation IN ('supersedes', 'contradicts')),
        created_at TEXT NOT NULL,
        effective_at TEXT,
        note TEXT NOT NULL DEFAULT '',
        UNIQUE (from_memory_id, to_memory_id, relation)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_memory_links_owner ON memory_links(owner_id, relation)",
    "CREATE INDEX IF NOT EXISTS idx_memory_links_to ON memory_links(to_memory_id)",
    """CREATE TABLE IF NOT EXISTS memory_embeddings (
        memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
        owner_id TEXT NOT NULL,
        space TEXT NOT NULL,
        model_id TEXT NOT NULL,
        model_version TEXT NOT NULL,
        dimensions INTEGER NOT NULL CHECK (dimensions > 0),
        content_hash TEXT NOT NULL,
        vector BLOB NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (memory_id, space)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_memory_embeddings_owner ON memory_embeddings(owner_id, space)",
    """CREATE TABLE IF NOT EXISTS memory_index_failures (
        memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
        owner_id TEXT NOT NULL,
        space TEXT NOT NULL,
        error_type TEXT NOT NULL,
        attempted_at TEXT NOT NULL,
        PRIMARY KEY (memory_id, space)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_memories_fact_key ON memories(owner_id, fact_key)",
)


def _migrate_to_v1(connection: sqlite3.Connection) -> None:
    for statement in _BASELINE_SCHEMA.split(";"):
        if statement.strip():
            connection.execute(statement)


def _migrate_to_v2(connection: sqlite3.Connection) -> None:
    existing = {row[1] for row in connection.execute("PRAGMA table_info(memories)")}
    for column, declaration in _TEMPORAL_COLUMNS:
        if column not in existing:
            connection.execute(f"ALTER TABLE memories ADD COLUMN {column} {declaration}")
    for statement in _MEMORY_V2_TABLES:
        connection.execute(statement)


MIGRATIONS: Final[tuple[tuple[int, Callable[[sqlite3.Connection], None]], ...]] = (
    (1, _migrate_to_v1),
    (2, _migrate_to_v2),
)


class MemoryRepository:
    """SQLite memory store; personal-memory writes require active owner consent."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        # The clock is injectable so evaluations can replay a deterministic
        # history; production code always uses the real UTC time.
        self._clock = clock or (lambda: datetime.now(UTC))
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

    def _open(self) -> sqlite3.Connection:
        if self._memory_uri:
            connection = sqlite3.connect(self._memory_uri, uri=True, timeout=10)
        else:
            connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = self._open()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialise(self) -> None:
        """Apply pending schema migrations, each inside its own transaction.

        ``PRAGMA user_version`` records the applied schema version. Databases
        created before versioning report 0 and already contain the baseline
        tables; the baseline migration is idempotent so they upgrade safely.
        A failed migration rolls back and leaves the previous version intact.
        """
        connection = self._open()
        connection.isolation_level = None
        try:
            current = connection.execute("PRAGMA user_version").fetchone()[0]
            if current > SCHEMA_VERSION:
                raise RuntimeError(
                    f"Database schema version {current} is newer than this software "
                    f"supports ({SCHEMA_VERSION}); refusing to modify it."
                )
            for version, migrate in MIGRATIONS:
                if version <= current:
                    continue
                connection.execute("BEGIN IMMEDIATE")
                try:
                    migrate(connection)
                    connection.execute(f"PRAGMA user_version = {version}")
                    connection.execute("COMMIT")
                except BaseException:
                    connection.execute("ROLLBACK")
                    raise
        finally:
            connection.close()

    def schema_version(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("PRAGMA user_version").fetchone()[0])

    def _now(self) -> str:
        moment = self._clock()
        if moment.tzinfo is None:
            raise ValueError("The repository clock must return timezone-aware datetimes")
        return moment.astimezone(UTC).isoformat()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"], owner_id=row["owner_id"], kind=MemoryKind(row["kind"]),
            content=row["content"], source=row["source"], consent_id=row["consent_id"],
            created_at=row["created_at"], metadata=json.loads(row["metadata_json"]),
            occurred_at=row["occurred_at"], valid_from=row["valid_from"],
            valid_to=row["valid_to"], fact_key=row["fact_key"],
            epistemic_status=row["epistemic_status"], updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_link(row: sqlite3.Row) -> MemoryLink:
        return MemoryLink(
            id=row["id"], owner_id=row["owner_id"], from_memory_id=row["from_memory_id"],
            to_memory_id=row["to_memory_id"], relation=row["relation"],
            created_at=row["created_at"], effective_at=row["effective_at"], note=row["note"],
        )

    # ------------------------------------------------------------------ consent

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
        """Revoke a grant and remove derived embeddings created under it.

        Records remain stored (hidden) as before; their vectors are deleted in
        the same transaction so no further semantic processing can occur.
        """
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE consents SET active = 0, revoked_at = ?
                   WHERE id = ? AND owner_id = ? AND active = 1""",
                (self._now(), consent_id, owner_id),
            )
            if cursor.rowcount == 1:
                for table in ("memory_embeddings", "memory_index_failures"):
                    connection.execute(
                        f"""DELETE FROM {table} WHERE owner_id = ? AND memory_id IN (
                               SELECT id FROM memories WHERE consent_id = ? AND owner_id = ?
                           )""",
                        (owner_id, consent_id, owner_id),
                    )
        return cursor.rowcount == 1

    def has_active_consent(self, owner_id: str, purpose: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM consents WHERE owner_id = ? AND purpose = ? AND active = 1",
                (owner_id, purpose),
            ).fetchone()
        return row is not None

    # ------------------------------------------------------------- memory CRUD

    @staticmethod
    def _validate_validity(valid_from: str | None, valid_to: str | None) -> None:
        # Both values are normalised UTC ISO strings, so string order is time order.
        if valid_from and valid_to and valid_from >= valid_to:
            raise ValueError("valid_from must be earlier than valid_to")

    @staticmethod
    def _validate_epistemic(status: str) -> None:
        if status not in EPISTEMIC_STATUSES:
            raise ValueError(f"epistemic_status must be one of {EPISTEMIC_STATUSES}")

    @staticmethod
    def _require_active_memory_consent(
        connection: sqlite3.Connection, owner_id: str, consent_id: str
    ) -> None:
        consent = connection.execute(
            """SELECT id FROM consents
               WHERE id = ? AND owner_id = ? AND purpose = 'personal_memory' AND active = 1""",
            (consent_id, owner_id),
        ).fetchone()
        if consent is None:
            raise ConsentRequiredError("Active personal_memory consent is required")

    def _insert_memory(
        self, connection: sqlite3.Connection, *, owner_id: str, kind: MemoryKind,
        content: str, source: str, consent_id: str, metadata: dict[str, Any] | None,
        occurred_at: str | datetime | None, valid_from: str | datetime | None,
        valid_to: str | datetime | None, fact_key: str | None, epistemic_status: str,
    ) -> MemoryRecord:
        if not content.strip():
            raise ValueError("Memory content cannot be empty")
        if not source.strip():
            raise ValueError("Memory source is required for provenance")
        self._validate_epistemic(epistemic_status)
        normalised = (
            optional_utc_iso(occurred_at), optional_utc_iso(valid_from), optional_utc_iso(valid_to)
        )
        self._validate_validity(normalised[1], normalised[2])
        self._require_active_memory_consent(connection, owner_id, consent_id)
        key = fact_key.strip() if fact_key and fact_key.strip() else None
        record = MemoryRecord(
            id=str(uuid4()), owner_id=owner_id, kind=kind, content=content,
            source=source, consent_id=consent_id, created_at=self._now(),
            metadata=metadata or {}, occurred_at=normalised[0], valid_from=normalised[1],
            valid_to=normalised[2], fact_key=key, epistemic_status=epistemic_status,
        )
        connection.execute(
            """INSERT INTO memories
               (id, owner_id, kind, content, source, consent_id, created_at, metadata_json,
                occurred_at, valid_from, valid_to, fact_key, epistemic_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (record.id, record.owner_id, record.kind.value, record.content,
             record.source, record.consent_id, record.created_at,
             json.dumps(record.metadata, ensure_ascii=False), record.occurred_at,
             record.valid_from, record.valid_to, record.fact_key, record.epistemic_status),
        )
        return record

    def add(
        self, *, owner_id: str, kind: MemoryKind, content: str, source: str,
        consent_id: str, metadata: dict[str, Any] | None = None,
        occurred_at: str | datetime | None = None, valid_from: str | datetime | None = None,
        valid_to: str | datetime | None = None, fact_key: str | None = None,
        epistemic_status: str = "asserted",
    ) -> MemoryRecord:
        with self._connect() as connection:
            return self._insert_memory(
                connection, owner_id=owner_id, kind=kind, content=content, source=source,
                consent_id=consent_id, metadata=metadata, occurred_at=occurred_at,
                valid_from=valid_from, valid_to=valid_to, fact_key=fact_key,
                epistemic_status=epistemic_status,
            )

    _VISIBLE: Final = """SELECT m.* FROM memories m
                         JOIN consents c ON c.id = m.consent_id
                         WHERE m.owner_id = ? AND c.active = 1
                           AND c.purpose = 'personal_memory'"""

    def list(self, owner_id: str, *, limit: int = 100) -> list[MemoryRecord]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        with self._connect() as connection:
            rows = connection.execute(
                self._VISIBLE + " ORDER BY m.created_at DESC, m.id LIMIT ?",
                (owner_id, limit),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def iter_visible(
        self, owner_id: str, *, kinds: Sequence[MemoryKind] | None = None
    ) -> builtins.list[MemoryRecord]:
        """Every consent-visible record for one owner; the search candidate set."""
        query = self._VISIBLE
        values: builtins.list[Any] = [owner_id]
        if kinds:
            query += f" AND m.kind IN ({', '.join('?' for _ in kinds)})"
            values.extend(kind.value for kind in kinds)
        query += " ORDER BY m.created_at, m.id"
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get(self, owner_id: str, memory_id: str) -> MemoryRecord:
        with self._connect() as connection:
            row = connection.execute(
                self._VISIBLE + " AND m.id = ?", (owner_id, memory_id)
            ).fetchone()
        if row is None:
            raise MemoryNotFoundError(memory_id)
        return self._row_to_record(row)

    def correct(
        self, owner_id: str, memory_id: str, *,
        content: str | _Unset = UNSET, metadata: dict[str, Any] | _Unset = UNSET,
        occurred_at: str | datetime | None | _Unset = UNSET,
        valid_from: str | datetime | None | _Unset = UNSET,
        valid_to: str | datetime | None | _Unset = UNSET,
        fact_key: str | None | _Unset = UNSET,
        epistemic_status: str | _Unset = UNSET,
    ) -> MemoryRecord:
        """Correct a record in place (rectification of an error in the record).

        A correction is different from a change in the world: use
        :meth:`supersede` when a fact stopped being true so history is kept.
        Changing content deletes the record's embeddings in the same
        transaction, so a stale vector can never describe the new text.
        """
        with self._connect() as connection:
            row = connection.execute(
                self._VISIBLE + " AND m.id = ?", (owner_id, memory_id)
            ).fetchone()
            if row is None:
                raise MemoryNotFoundError(memory_id)
            current = self._row_to_record(row)
            new_content = current.content if isinstance(content, _Unset) else content
            if not new_content.strip():
                raise ValueError("Memory content cannot be empty")
            new_status = (
                current.epistemic_status
                if isinstance(epistemic_status, _Unset) else epistemic_status
            )
            self._validate_epistemic(new_status)
            values = {
                "occurred_at": current.occurred_at if isinstance(occurred_at, _Unset)
                else optional_utc_iso(occurred_at),
                "valid_from": current.valid_from if isinstance(valid_from, _Unset)
                else optional_utc_iso(valid_from),
                "valid_to": current.valid_to if isinstance(valid_to, _Unset)
                else optional_utc_iso(valid_to),
            }
            self._validate_validity(values["valid_from"], values["valid_to"])
            new_key = current.fact_key if isinstance(fact_key, _Unset) else (
                fact_key.strip() if fact_key and fact_key.strip() else None
            )
            new_metadata = current.metadata if isinstance(metadata, _Unset) else metadata
            updated_at = self._now()
            connection.execute(
                """UPDATE memories SET content = ?, metadata_json = ?, occurred_at = ?,
                       valid_from = ?, valid_to = ?, fact_key = ?, epistemic_status = ?,
                       updated_at = ?
                   WHERE id = ? AND owner_id = ?""",
                (new_content, json.dumps(new_metadata, ensure_ascii=False),
                 values["occurred_at"], values["valid_from"], values["valid_to"], new_key,
                 new_status, updated_at, memory_id, owner_id),
            )
            if new_content != current.content:
                connection.execute(
                    "DELETE FROM memory_embeddings WHERE memory_id = ?", (memory_id,)
                )
                connection.execute(
                    "DELETE FROM memory_index_failures WHERE memory_id = ?", (memory_id,)
                )
            row = connection.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return self._row_to_record(row)

    def supersede(
        self, *, owner_id: str, memory_id: str, kind: MemoryKind | None = None,
        content: str, source: str, consent_id: str, metadata: dict[str, Any] | None = None,
        occurred_at: str | datetime | None = None, valid_from: str | datetime | None = None,
        valid_to: str | datetime | None = None, epistemic_status: str = "asserted",
        note: str = "",
    ) -> tuple[MemoryRecord, MemoryLink]:
        """Record that a new memory replaces an older one without erasing it.

        The old record is not modified. A ``supersedes`` link stores when the
        replacement took effect (the new record's ``valid_from``, else its
        ``occurred_at``, else the time it was recorded). An inferred record
        cannot supersede an asserted one: uncertain inferences must not displace
        established information.
        """
        with self._connect() as connection:
            row = connection.execute(
                self._VISIBLE + " AND m.id = ?", (owner_id, memory_id)
            ).fetchone()
            if row is None:
                raise MemoryNotFoundError(memory_id)
            previous = self._row_to_record(row)
            if epistemic_status == "inferred" and previous.epistemic_status == "asserted":
                raise ValueError("An inferred memory cannot supersede an asserted memory")
            existing = connection.execute(
                """SELECT from_memory_id FROM memory_links
                   WHERE to_memory_id = ? AND relation = 'supersedes'""",
                (memory_id,),
            ).fetchone()
            if existing is not None:
                raise ValueError(
                    f"Memory is already superseded by {existing['from_memory_id']}; "
                    "supersede the latest version instead"
                )
            record = self._insert_memory(
                connection, owner_id=owner_id, kind=kind or previous.kind, content=content,
                source=source, consent_id=consent_id, metadata=metadata,
                occurred_at=occurred_at, valid_from=valid_from, valid_to=valid_to,
                fact_key=previous.fact_key, epistemic_status=epistemic_status,
            )
            link = MemoryLink(
                id=str(uuid4()), owner_id=owner_id, from_memory_id=record.id,
                to_memory_id=memory_id, relation="supersedes", created_at=record.created_at,
                effective_at=record.valid_from or record.occurred_at or record.created_at,
                note=note,
            )
            self._insert_link(connection, link)
        return record, link

    @staticmethod
    def _insert_link(connection: sqlite3.Connection, link: MemoryLink) -> None:
        connection.execute(
            """INSERT INTO memory_links
               (id, owner_id, from_memory_id, to_memory_id, relation, created_at,
                effective_at, note)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (link.id, link.owner_id, link.from_memory_id, link.to_memory_id, link.relation,
             link.created_at, link.effective_at, link.note),
        )

    def link_contradiction(
        self, owner_id: str, memory_id: str, other_memory_id: str, *, note: str = ""
    ) -> MemoryLink:
        """Explicitly record that two memories conflict; neither is discarded."""
        if memory_id == other_memory_id:
            raise ValueError("A memory cannot contradict itself")
        with self._connect() as connection:
            for item in (memory_id, other_memory_id):
                if connection.execute(
                    self._VISIBLE + " AND m.id = ?", (owner_id, item)
                ).fetchone() is None:
                    raise MemoryNotFoundError(item)
            duplicate = connection.execute(
                """SELECT * FROM memory_links WHERE relation = 'contradicts' AND (
                       (from_memory_id = ? AND to_memory_id = ?)
                    OR (from_memory_id = ? AND to_memory_id = ?))""",
                (memory_id, other_memory_id, other_memory_id, memory_id),
            ).fetchone()
            if duplicate is not None:
                return self._row_to_link(duplicate)
            link = MemoryLink(
                id=str(uuid4()), owner_id=owner_id, from_memory_id=memory_id,
                to_memory_id=other_memory_id, relation="contradicts",
                created_at=self._now(), note=note,
            )
            self._insert_link(connection, link)
        return link

    def links(self, owner_id: str) -> builtins.list[MemoryLink]:
        """Links whose both endpoints are currently consent-visible."""
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT l.* FROM memory_links l
                   JOIN memories a ON a.id = l.from_memory_id
                   JOIN consents ca ON ca.id = a.consent_id AND ca.active = 1
                   JOIN memories b ON b.id = l.to_memory_id
                   JOIN consents cb ON cb.id = b.consent_id AND cb.active = 1
                   WHERE l.owner_id = ? ORDER BY l.created_at, l.id""",
                (owner_id,),
            ).fetchall()
        return [self._row_to_link(row) for row in rows]

    def version_chain(self, owner_id: str, memory_id: str) -> builtins.list[MemoryRecord]:
        """All visible versions connected by supersession, oldest first."""
        self.get(owner_id, memory_id)
        links = [link for link in self.links(owner_id) if link.relation == "supersedes"]
        newer = {link.to_memory_id: link.from_memory_id for link in links}
        older = {link.from_memory_id: link.to_memory_id for link in links}
        head = memory_id
        while head in older:
            head = older[head]
        chain = [head]
        while chain[-1] in newer:
            chain.append(newer[chain[-1]])
        return [self.get(owner_id, item) for item in chain]

    def delete(self, owner_id: str, memory_id: str) -> bool:
        """Delete a memory together with its embeddings, index state, and links."""
        with self._connect() as connection:
            owned = connection.execute(
                "SELECT 1 FROM memories WHERE id = ? AND owner_id = ?", (memory_id, owner_id)
            ).fetchone()
            if owned is None:
                return False
            connection.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (memory_id,))
            connection.execute(
                "DELETE FROM memory_index_failures WHERE memory_id = ?", (memory_id,)
            )
            connection.execute(
                "DELETE FROM memory_links WHERE from_memory_id = ? OR to_memory_id = ?",
                (memory_id, memory_id),
            )
            cursor = connection.execute(
                "DELETE FROM memories WHERE id = ? AND owner_id = ?", (memory_id, owner_id)
            )
        return cursor.rowcount == 1

    def erase_owner(self, owner_id: str) -> int:
        """Delete all memories and revoke all consent grants for this owner."""
        now = self._now()
        with self._connect() as connection:
            for table in ("memory_embeddings", "memory_index_failures", "memory_links"):
                connection.execute(f"DELETE FROM {table} WHERE owner_id = ?", (owner_id,))
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

    @staticmethod
    def record_to_dict(item: MemoryRecord) -> dict[str, Any]:
        return {
            "id": item.id, "owner_id": item.owner_id, "kind": item.kind.value,
            "content": item.content, "source": item.source, "consent_id": item.consent_id,
            "created_at": item.created_at, "recorded_at": item.recorded_at,
            "occurred_at": item.occurred_at, "valid_from": item.valid_from,
            "valid_to": item.valid_to, "fact_key": item.fact_key,
            "epistemic_status": item.epistemic_status, "updated_at": item.updated_at,
            "metadata": item.metadata,
        }

    def export(self, owner_id: str) -> builtins.list[dict[str, Any]]:
        """Export visible records with their links and embedding metadata.

        Raw vectors are derived data and are not exported; they can be rebuilt
        from content with the recorded model identity.
        """
        records = self.iter_visible(owner_id)
        records.sort(key=lambda item: (item.created_at, item.id), reverse=True)
        links = self.links(owner_id)
        with self._connect() as connection:
            embedding_rows = connection.execute(
                """SELECT memory_id, space, model_id, model_version, dimensions, created_at
                   FROM memory_embeddings WHERE owner_id = ?""",
                (owner_id,),
            ).fetchall()
        embeddings: dict[str, builtins.list[dict[str, Any]]] = {}
        for row in embedding_rows:
            embeddings.setdefault(row["memory_id"], []).append({
                "space": row["space"], "model_id": row["model_id"],
                "model_version": row["model_version"], "dimensions": row["dimensions"],
                "indexed_at": row["created_at"],
            })
        exported = []
        for item in records[:500]:
            payload = self.record_to_dict(item)
            payload["links"] = [
                {
                    "id": link.id, "relation": link.relation, "from_memory_id": link.from_memory_id,
                    "to_memory_id": link.to_memory_id, "effective_at": link.effective_at,
                    "created_at": link.created_at, "note": link.note,
                }
                for link in links if item.id in (link.from_memory_id, link.to_memory_id)
            ]
            payload["embeddings"] = embeddings.get(item.id, [])
            exported.append(payload)
        return exported

    # --------------------------------------------------------------- embeddings

    def store_embedding(
        self, *, owner_id: str, memory_id: str, space: str, model_id: str,
        model_version: str, dimensions: int, content_hash: str, vector: bytes,
    ) -> bool:
        """Atomically store a vector only if the memory is still eligible.

        The insert is conditional on the record still existing for this owner,
        its consent still being active, and its content still matching the
        hashed text. If any condition fails (deleted, revoked, or corrected in
        the meantime) nothing is written and ``False`` is returned.
        """
        with self._connect() as connection:
            row = connection.execute(
                self._VISIBLE + " AND m.id = ?", (owner_id, memory_id)
            ).fetchone()
            if row is None or content_fingerprint(row["content"]) != content_hash:
                return False
            connection.execute(
                """INSERT INTO memory_embeddings
                   (memory_id, owner_id, space, model_id, model_version, dimensions,
                    content_hash, vector, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(memory_id, space) DO UPDATE SET
                     model_id = excluded.model_id, model_version = excluded.model_version,
                     dimensions = excluded.dimensions, content_hash = excluded.content_hash,
                     vector = excluded.vector, created_at = excluded.created_at""",
                (memory_id, owner_id, space, model_id, model_version, dimensions,
                 content_hash, vector, self._now()),
            )
            connection.execute(
                "DELETE FROM memory_index_failures WHERE memory_id = ? AND space = ?",
                (memory_id, space),
            )
        return True

    def record_index_failure(
        self, *, owner_id: str, memory_id: str, space: str, error_type: str
    ) -> None:
        """Record only the failure class; never the content or provider message."""
        with self._connect() as connection:
            if connection.execute(
                "SELECT 1 FROM memories WHERE id = ? AND owner_id = ?", (memory_id, owner_id)
            ).fetchone() is None:
                return
            connection.execute(
                """INSERT INTO memory_index_failures
                   (memory_id, owner_id, space, error_type, attempted_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(memory_id, space) DO UPDATE SET
                     error_type = excluded.error_type, attempted_at = excluded.attempted_at""",
                (memory_id, owner_id, space, error_type[:120], self._now()),
            )

    def embeddings(
        self, owner_id: str, space: str, memory_ids: Iterable[str] | None = None
    ) -> dict[str, StoredEmbedding]:
        """Current-space vectors for consent-visible records whose content is unchanged."""
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT e.*, m.content AS current_content FROM memory_embeddings e
                   JOIN memories m ON m.id = e.memory_id
                   JOIN consents c ON c.id = m.consent_id
                   WHERE e.owner_id = ? AND m.owner_id = ? AND e.space = ?
                     AND c.active = 1 AND c.purpose = 'personal_memory'""",
                (owner_id, owner_id, space),
            ).fetchall()
        wanted = set(memory_ids) if memory_ids is not None else None
        result: dict[str, StoredEmbedding] = {}
        for row in rows:
            if wanted is not None and row["memory_id"] not in wanted:
                continue
            if content_fingerprint(row["current_content"]) != row["content_hash"]:
                continue
            result[row["memory_id"]] = StoredEmbedding(
                memory_id=row["memory_id"], space=row["space"], model_id=row["model_id"],
                model_version=row["model_version"], dimensions=row["dimensions"],
                content_hash=row["content_hash"], vector=row["vector"],
                created_at=row["created_at"],
            )
        return result

    def delete_embeddings_outside_space(self, owner_id: str, space: str) -> int:
        """Remove vectors produced by a different (previous) embedding model."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM memory_embeddings WHERE owner_id = ? AND space != ?",
                (owner_id, space),
            )
            connection.execute(
                "DELETE FROM memory_index_failures WHERE owner_id = ? AND space != ?",
                (owner_id, space),
            )
        return cursor.rowcount

    def index_counts(self, owner_id: str, space: str | None) -> IndexCounts:
        visible = self.iter_visible(owner_id)
        current = self.embeddings(owner_id, space) if space else {}
        with self._connect() as connection:
            stored = connection.execute(
                "SELECT memory_id, space FROM memory_embeddings WHERE owner_id = ?", (owner_id,)
            ).fetchall()
            failures = connection.execute(
                """SELECT memory_id, error_type, attempted_at FROM memory_index_failures
                   WHERE owner_id = ? AND space = ? ORDER BY attempted_at, memory_id""",
                (owner_id, space or ""),
            ).fetchall()
        in_space = {row["memory_id"] for row in stored if row["space"] == space}
        visible_ids = {item.id for item in visible}
        return IndexCounts(
            eligible=len(visible),
            indexed=len(current),
            stale=len((in_space & visible_ids) - set(current)),
            failed=len(failures),
            other_spaces=sum(1 for row in stored if row["space"] != space),
            failures=tuple(
                {
                    "memory_id": row["memory_id"], "error_type": row["error_type"],
                    "attempted_at": row["attempted_at"],
                }
                for row in failures
            ),
        )

    # ----------------------------------------------------------------- learning

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
