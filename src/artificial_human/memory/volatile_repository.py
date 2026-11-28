"""Bounded process-local memory repository for the active organism runtime.

This adapter implements the existing consent/provenance memory contract with Python
data structures only. It opens no SQLite connections and creates no files. State is
cleared on close/process restart. The SQLite MemoryRepository remains available only
for explicit legacy tooling and offline compatibility tests.
"""
from __future__ import annotations

import builtins
from dataclasses import replace
import json
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from collections.abc import Callable, Iterable, Sequence
from typing import Any
from uuid import uuid4

from artificial_human.core.models import MemoryKind
from artificial_human.memory.repository import (
    ConsentGrant,
    ConsentRequiredError,
    IndexCounts,
    LearningExample,
    MemoryLink,
    MemoryNotFoundError,
    MemoryRecord,
    StoredEmbedding,
    UNSET,
    content_fingerprint,
)
from artificial_human.memory.temporal import optional_utc_iso
from artificial_human.memory.contracts import MemoryStore

DEFAULT_MAX_RECORDS = 10_000
DEFAULT_MAX_TRAINING_EXAMPLES = 1_000
DEFAULT_MAX_EMBEDDINGS = 10_000


class InMemoryMemoryRepository(MemoryStore):
    """Consent-aware repository backed only by bounded in-process dictionaries."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
        max_records: int = DEFAULT_MAX_RECORDS,
        max_training_examples: int = DEFAULT_MAX_TRAINING_EXAMPLES,
        max_embeddings: int = DEFAULT_MAX_EMBEDDINGS,
        max_consents: int = 2_000,
        max_links: int = 20_000,
    ) -> None:
        # 'path' is accepted for source compatibility and deliberately ignored.
        if min(max_records, max_training_examples, max_embeddings, max_consents, max_links) < 1:
            raise ValueError("in-memory repository capacities must be positive")
        self.path = Path("<in-memory>")
        self._clock = clock or (lambda: datetime.now(UTC))
        self.max_records = max_records
        self.max_training_examples = max_training_examples
        self.max_embeddings = max_embeddings
        self.max_consents = max_consents
        self.max_links = max_links
        self._consents: dict[str, ConsentGrant] = {}
        self._records: dict[str, MemoryRecord] = {}
        self._links: dict[str, MemoryLink] = {}
        self._embeddings: dict[tuple[str, str], StoredEmbedding] = {}
        self._index_failures: dict[tuple[str, str], dict[str, str]] = {}
        self._training_examples: dict[str, LearningExample] = {}
        self._lock = RLock()
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("in-memory repository is closed")

    def _now(self) -> str:
        return self._clock().astimezone(UTC).isoformat()

    @staticmethod
    def schema_version() -> int:
        return 0

    def close(self) -> None:
        with self._lock:
            self._consents.clear()
            self._records.clear()
            self._links.clear()
            self._embeddings.clear()
            self._index_failures.clear()
            self._training_examples.clear()
            self._closed = True

    def reset(self) -> None:
        """Erase all volatile state and reopen the repository for a fresh organism."""
        with self._lock:
            self._consents.clear()
            self._records.clear()
            self._links.clear()
            self._embeddings.clear()
            self._index_failures.clear()
            self._training_examples.clear()
            self._closed = False

    def grant_consent(
        self, owner_id: str, purpose: str = "personal_memory"
    ) -> ConsentGrant:
        self._ensure_open()
        if not owner_id.strip() or not purpose.strip():
            raise ValueError("owner_id and purpose are required")
        if purpose not in {"personal_memory", "personal_learning", "external_embedding"}:
            raise ValueError("unsupported consent purpose")
        with self._lock:
            if len(self._consents) >= self.max_consents:
                raise ValueError("in-memory consent capacity reached")
            grant = ConsentGrant(
                id=str(uuid4()), owner_id=owner_id, purpose=purpose,
                active=True, created_at=self._now(),
            )
            self._consents[grant.id] = grant
            return grant

    def revoke_consent(self, owner_id: str, consent_id: str) -> bool:
        self._ensure_open()
        with self._lock:
            grant = self._consents.get(consent_id)
            if grant is None or grant.owner_id != owner_id or not grant.active:
                return False
            self._consents[consent_id] = replace(
                grant, active=False, revoked_at=self._now()
            )
            revoked_ids = {
                record.id for record in self._records.values()
                if record.owner_id == owner_id and record.consent_id == consent_id
            }
            for memory_id in revoked_ids:
                self._records.pop(memory_id, None)
                for key in [key for key in self._embeddings if key[0] == memory_id]:
                    self._embeddings.pop(key, None)
                for key in [key for key in self._index_failures if key[0] == memory_id]:
                    self._index_failures.pop(key, None)
            for link_id, link in list(self._links.items()):
                if link.owner_id == owner_id and (
                    link.from_memory_id in revoked_ids or link.to_memory_id in revoked_ids
                ):
                    self._links.pop(link_id, None)
            for example_id, example in list(self._training_examples.items()):
                if example.owner_id == owner_id and example.consent_id == consent_id:
                    self._training_examples.pop(example_id, None)
            return True

    def has_active_consent(self, owner_id: str, purpose: str) -> bool:
        self._ensure_open()
        return any(
            item.owner_id == owner_id and item.purpose == purpose and item.active
            for item in self._consents.values()
        )

    def _consent(self, owner_id: str, consent_id: str, purpose: str) -> ConsentGrant:
        grant = self._consents.get(consent_id)
        if (
            grant is None or grant.owner_id != owner_id or grant.purpose != purpose
            or not grant.active
        ):
            raise ConsentRequiredError(f"Active {purpose} consent is required")
        return grant

    @staticmethod
    def _validate_validity(valid_from: str | datetime | None, valid_to: str | datetime | None) -> None:
        start = optional_utc_iso(valid_from)
        end = optional_utc_iso(valid_to)
        if start and end and datetime.fromisoformat(start) >= datetime.fromisoformat(end):
            raise ValueError("valid_from must be earlier than valid_to")

    @staticmethod
    def _validate_epistemic(status: str) -> None:
        if status not in {"asserted", "inferred"}:
            raise ValueError("epistemic_status must be 'asserted' or 'inferred'")

    def _visible_record(self, owner_id: str, memory_id: str) -> MemoryRecord:
        record = self._records.get(memory_id)
        if record is None or record.owner_id != owner_id:
            raise MemoryNotFoundError(memory_id)
        grant = self._consents.get(record.consent_id)
        if grant is None or not grant.active:
            raise MemoryNotFoundError(memory_id)
        return record

    def add(
        self, *, owner_id: str, kind: MemoryKind, content: str, source: str,
        consent_id: str, metadata: dict[str, Any] | None = None,
        occurred_at: str | datetime | None = None, valid_from: str | datetime | None = None,
        valid_to: str | datetime | None = None, fact_key: str | None = None,
        epistemic_status: str = "asserted",
    ) -> MemoryRecord:
        self._ensure_open()
        if not owner_id.strip() or not content.strip() or not source.strip():
            raise ValueError("owner_id, content, and source are required")
        if len(content) > 20_000:
            raise ValueError("memory content exceeds the 20000-character runtime limit")
        if len(json.dumps(metadata or {}, ensure_ascii=False, allow_nan=False)) > 32_000:
            raise ValueError("memory metadata exceeds the 32000-byte runtime limit")
        self._consent(owner_id, consent_id, "personal_memory")
        self._validate_validity(valid_from, valid_to)
        self._validate_epistemic(epistemic_status)
        occurred = optional_utc_iso(occurred_at)
        start = optional_utc_iso(valid_from)
        end = optional_utc_iso(valid_to)
        with self._lock:
            if len(self._records) >= self.max_records:
                raise ValueError("in-memory memory capacity reached; no record was evicted")
            now = self._now()
            record = MemoryRecord(
                id=str(uuid4()), owner_id=owner_id, kind=kind, content=content,
                source=source, consent_id=consent_id, created_at=now,
                metadata=dict(metadata or {}), occurred_at=occurred,
                valid_from=start, valid_to=end, fact_key=fact_key,
                epistemic_status=epistemic_status, updated_at=None,
            )
            self._records[record.id] = record
            return record

    def add_consolidated_memory(
        self, *, owner_id: str, source_memory_ids: Sequence[str],
        content: str, source: str, consent_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        self._ensure_open()
        self._consent(owner_id, consent_id, "personal_memory")
        if len(set(source_memory_ids)) < 2:
            raise ValueError("at least two distinct episodic source memories are required")
        with self._lock:
            records = [self._visible_record(owner_id, item) for item in source_memory_ids]
            if any(item.kind is not MemoryKind.EPISODIC for item in records):
                raise ValueError("all consolidation sources must be episodic")
            if any(item.consent_id != consent_id for item in records):
                raise ValueError("consolidation sources must share the active consent grant")
            source_ids = {item.id for item in records}
            if any(
                link.relation == "contradicts"
                and link.from_memory_id in source_ids and link.to_memory_id in source_ids
                for link in self._links.values()
            ):
                raise ValueError("Contradictory sources cannot be consolidated")
            details = dict(metadata or {})
            details.update({
                "consolidation_source_memory_ids": list(source_memory_ids),
                "consolidation_source_count": len(source_memory_ids),
                "consolidation_review_required": False,
            })
        return self.add(
            owner_id=owner_id, kind=MemoryKind.SEMANTIC, content=content,
            source=source, consent_id=consent_id, metadata=details,
            epistemic_status="inferred",
        )

    def add_schema_memory(
        self, *, owner_id: str, source_sequences: Sequence[Sequence[str]],
        event_types: Sequence[str], schema: dict[str, Any],
        content: str, source: str, consent_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        self._ensure_open()
        self._consent(owner_id, consent_id, "personal_memory")
        sequences = tuple(tuple(sequence) for sequence in source_sequences)
        types = tuple(event_types)
        if len(sequences) < 2 or len(set(sequences)) != len(sequences):
            raise ValueError("At least two distinct source sequences are required")
        if len(types) < 2 or not all(isinstance(item, str) and item for item in types):
            raise ValueError("At least two event types are required")
        if any(len(sequence) != len(types) for sequence in sequences):
            raise ValueError("Every source sequence must match the schema event-type length")
        if any(
            not sequence or any(not isinstance(memory_id, str) or not memory_id for memory_id in sequence)
            for sequence in sequences
        ):
            raise ValueError("Source sequences must contain non-empty memory identifiers")
        if not isinstance(schema, dict) or not schema:
            raise ValueError("schema must be a non-empty object")
        try:
            schema_json = json.dumps(schema, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("schema must contain only finite JSON-compatible values") from error
        source_ids = tuple(dict.fromkeys(
            memory_id for sequence in sequences for memory_id in sequence
        ))
        with self._lock:
            source_records = {
                item: self._visible_record(owner_id, item) for item in source_ids
            }
            if any(record.kind is not MemoryKind.EPISODIC for record in source_records.values()):
                raise ValueError("schema sources must be episodic")
            if any(record.consent_id != consent_id for record in source_records.values()):
                raise ValueError("schema sources must share the active consent grant")
            for sequence in sequences:
                actual_types = tuple(
                    str(source_records[memory_id].metadata.get("event_type", "")).strip().lower()
                    for memory_id in sequence
                )
                if actual_types != types:
                    raise ValueError("A source event type changed after schema extraction")
            if any(
                link.relation == "contradicts"
                and link.from_memory_id in source_ids and link.to_memory_id in source_ids
                for link in self._links.values()
            ):
                raise ValueError("Contradictory sources cannot be used for schema extraction")
        details = dict(metadata or {})
        details.update({
            "knowledge_type": "ordered_event_schema",
            "schema": json.loads(schema_json),
            "schema_event_types": list(types),
            "schema_source_sequences": [list(item) for item in sequences],
            "schema_source_memory_ids": list(source_ids),
            "schema_support_count": len(sequences),
        })
        return self.add(
            owner_id=owner_id, kind=MemoryKind.PROCEDURAL, content=content,
            source=source, consent_id=consent_id, metadata=details,
            epistemic_status="inferred",
        )

    def list(self, owner_id: str, *, limit: int = 100) -> builtins.list[MemoryRecord]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        return sorted(self.iter_visible(owner_id), key=lambda item: (item.created_at, item.id), reverse=True)[:limit]

    def iter_visible(
        self, owner_id: str, *, kinds: Sequence[MemoryKind] | None = None,
    ) -> builtins.list[MemoryRecord]:
        self._ensure_open()
        with self._lock:
            items = [
                record for record in self._records.values()
                if record.owner_id == owner_id
                and (grant := self._consents.get(record.consent_id)) is not None
                and grant.active
            ]
        if kinds:
            selected = set(kinds)
            items = [item for item in items if item.kind in selected]
        return sorted(items, key=lambda item: (item.created_at, item.id))

    def get(self, owner_id: str, memory_id: str) -> MemoryRecord:
        self._ensure_open()
        with self._lock:
            return self._visible_record(owner_id, memory_id)

    def correct(
        self, owner_id: str, memory_id: str, *, content=UNSET, metadata=UNSET,
        occurred_at=UNSET, valid_from=UNSET, valid_to=UNSET, fact_key=UNSET,
        epistemic_status=UNSET,
    ) -> MemoryRecord:
        self._ensure_open()
        with self._lock:
            current = self._visible_record(owner_id, memory_id)
            values: dict[str, Any] = {}
            for name, value in (
                ("content", content), ("metadata", metadata),
                ("occurred_at", occurred_at), ("valid_from", valid_from),
                ("valid_to", valid_to), ("fact_key", fact_key),
                ("epistemic_status", epistemic_status),
            ):
                if value is not UNSET:
                    values[name] = value
            if "content" in values and (not isinstance(values["content"], str) or not values["content"].strip()):
                raise ValueError("content cannot be empty")
            if "metadata" in values:
                if not isinstance(values["metadata"], dict):
                    raise ValueError("metadata must be an object")
                values["metadata"] = dict(values["metadata"])
            for name in ("occurred_at", "valid_from", "valid_to"):
                if name in values:
                    values[name] = optional_utc_iso(values[name])
            next_start = values.get("valid_from", current.valid_from)
            next_end = values.get("valid_to", current.valid_to)
            self._validate_validity(next_start, next_end)
            if "epistemic_status" in values:
                self._validate_epistemic(values["epistemic_status"])
            updated = replace(current, **values, updated_at=self._now())
            self._records[memory_id] = updated
            if updated.content != current.content:
                for key in [key for key in self._embeddings if key[0] == memory_id]:
                    self._embeddings.pop(key, None)
            return updated

    def supersede(
        self, *, owner_id: str, memory_id: str, content: str, source: str,
        consent_id: str, kind: MemoryKind | None = None,
        metadata: dict[str, Any] | None = None, occurred_at: str | datetime | None = None,
        valid_from: str | datetime | None = None, valid_to: str | datetime | None = None,
        epistemic_status: str = "asserted", note: str = "",
    ) -> tuple[MemoryRecord, MemoryLink]:
        self._ensure_open()
        self._consent(owner_id, consent_id, "personal_memory")
        with self._lock:
            previous = self._visible_record(owner_id, memory_id)
            if previous.consent_id != consent_id:
                raise ValueError("superseding memory must use the same consent grant")
            if any(link.to_memory_id == memory_id and link.relation == "supersedes"
                   for link in self._links.values()):
                raise ValueError("memory is already superseded")
            if kind is not None and kind is not previous.kind:
                raise ValueError("superseding memory must keep the same kind")
        record = self.add(
            owner_id=owner_id, kind=kind or previous.kind, content=content, source=source,
            consent_id=consent_id, metadata=metadata, occurred_at=occurred_at,
            valid_from=valid_from, valid_to=valid_to, fact_key=previous.fact_key,
            epistemic_status=epistemic_status,
        )
        link = MemoryLink(
            id=str(uuid4()), owner_id=owner_id, from_memory_id=record.id,
            to_memory_id=previous.id, relation="supersedes", created_at=self._now(),
            effective_at=optional_utc_iso(valid_from), note=note,
        )
        with self._lock:
            if len(self._links) >= self.max_links:
                self.delete(owner_id, record.id)
                raise ValueError("in-memory link capacity reached")
            self._links[link.id] = link
        return record, link

    def link_contradiction(
        self, owner_id: str, memory_id: str, other_memory_id: str, *, note: str = ""
    ) -> MemoryLink:
        self._ensure_open()
        with self._lock:
            first = self._visible_record(owner_id, memory_id)
            second = self._visible_record(owner_id, other_memory_id)
            if first.id == second.id:
                raise ValueError("a memory cannot contradict itself")
            if first.consent_id != second.consent_id:
                raise ValueError("contradiction links require the same consent grant")
            if any(
                link.relation == "contradicts"
                and {link.from_memory_id, link.to_memory_id} == {first.id, second.id}
                for link in self._links.values()
            ):
                raise ValueError("contradiction link already exists")
            link = MemoryLink(
                id=str(uuid4()), owner_id=owner_id, from_memory_id=first.id,
                to_memory_id=second.id, relation="contradicts", created_at=self._now(),
                note=note,
            )
            if len(self._links) >= self.max_links:
                raise ValueError("in-memory link capacity reached")
            self._links[link.id] = link
            return link

    def links(self, owner_id: str) -> builtins.list[MemoryLink]:
        self._ensure_open()
        with self._lock:
            return sorted(
                (item for item in self._links.values() if item.owner_id == owner_id),
                key=lambda item: (item.created_at, item.id),
            )

    def version_chain(self, owner_id: str, memory_id: str) -> builtins.list[MemoryRecord]:
        self._ensure_open()
        with self._lock:
            self._visible_record(owner_id, memory_id)
            related = {memory_id}
            changed = True
            while changed:
                changed = False
                for link in self._links.values():
                    if link.owner_id == owner_id and link.relation == "supersedes":
                        pair = {link.from_memory_id, link.to_memory_id}
                        if pair & related and not pair <= related:
                            related |= pair
                            changed = True
            records = [
                self._visible_record(owner_id, item)
                for item in related
            ]
        return sorted(records, key=lambda item: (item.created_at, item.id))

    def delete(self, owner_id: str, memory_id: str) -> bool:
        self._ensure_open()
        with self._lock:
            record = self._records.get(memory_id)
            if record is None or record.owner_id != owner_id:
                return False
            self._records.pop(memory_id, None)
            for link_id, link in list(self._links.items()):
                if link.from_memory_id == memory_id or link.to_memory_id == memory_id:
                    self._links.pop(link_id, None)
            for key in [key for key in self._embeddings if key[0] == memory_id]:
                self._embeddings.pop(key, None)
            for key in [key for key in self._index_failures if key[0] == memory_id]:
                self._index_failures.pop(key, None)
            return True

    def erase_owner(self, owner_id: str) -> int:
        self._ensure_open()
        with self._lock:
            ids = [item.id for item in self._records.values() if item.owner_id == owner_id]
            for memory_id in ids:
                self.delete(owner_id, memory_id)
            for consent_id, grant in list(self._consents.items()):
                if grant.owner_id == owner_id:
                    self._consents.pop(consent_id, None)
            for example_id, example in list(self._training_examples.items()):
                if example.owner_id == owner_id:
                    self._training_examples.pop(example_id, None)
            return len(ids)

    @staticmethod
    def record_to_dict(item: MemoryRecord) -> dict[str, Any]:
        from artificial_human.memory.records import memory_record_to_dict

        return memory_record_to_dict(item)

    def export(self, owner_id: str) -> builtins.list[dict[str, Any]]:
        self._ensure_open()
        records = self.iter_visible(owner_id)
        link_map: dict[str, builtins.list[dict[str, Any]]] = {}
        for link in self.links(owner_id):
            for memory_id in (link.from_memory_id, link.to_memory_id):
                link_map.setdefault(memory_id, []).append({
                    "id": link.id, "relation": link.relation,
                    "from_memory_id": link.from_memory_id,
                    "to_memory_id": link.to_memory_id, "created_at": link.created_at,
                    "effective_at": link.effective_at, "note": link.note,
                })
        result: builtins.list[dict[str, Any]] = []
        for record in records:
            embeddings = self.embeddings(owner_id, None, [record.id])
            result.append({
                **self.record_to_dict(record),
                "links": link_map.get(record.id, []),
                "embeddings": [
                    {
                        "space": item.space, "model_id": item.model_id,
                        "model_version": item.model_version, "dimensions": item.dimensions,
                        "content_hash": item.content_hash, "created_at": item.created_at,
                    }
                    for item in embeddings.values()
                ],
            })
        return result

    def store_embedding(
        self, *, owner_id: str, memory_id: str, space: str, model_id: str,
        model_version: str, dimensions: int, content_hash: str, vector: bytes,
    ) -> bool:
        self._ensure_open()
        with self._lock:
            try:
                record = self._visible_record(owner_id, memory_id)
            except MemoryNotFoundError:
                return False
            if content_fingerprint(record.content) != content_hash:
                return False
            key = (memory_id, space)
            if key not in self._embeddings and len(self._embeddings) >= self.max_embeddings:
                return False
            self._embeddings[key] = StoredEmbedding(
                memory_id=memory_id, space=space, model_id=model_id,
                model_version=model_version, dimensions=dimensions,
                content_hash=content_hash, vector=vector, created_at=self._now(),
            )
            return True

    def record_index_failure(
        self, *, owner_id: str, memory_id: str, space: str, error_type: str,
    ) -> None:
        self._ensure_open()
        with self._lock:
            self._visible_record(owner_id, memory_id)
            key = (memory_id, space)
            if key not in self._index_failures and len(self._index_failures) >= self.max_embeddings:
                return
            self._index_failures[key] = {
                "memory_id": memory_id, "error_type": error_type, "attempted_at": self._now(),
            }

    def embeddings(
        self, owner_id: str, space: str | None, memory_ids: Iterable[str] | None = None,
    ) -> dict[str, StoredEmbedding]:
        self._ensure_open()
        wanted = set(memory_ids) if memory_ids is not None else None
        with self._lock:
            result={}
            for (memory_id, stored_space), item in self._embeddings.items():
                if wanted is not None and memory_id not in wanted:
                    continue
                if space is not None and stored_space != space:
                    continue
                try:
                    record = self._visible_record(owner_id, memory_id)
                except MemoryNotFoundError:
                    continue
                if content_fingerprint(record.content) == item.content_hash:
                    result[memory_id] = item
            return result

    def delete_embeddings_outside_space(self, owner_id: str, space: str) -> int:
        self._ensure_open()
        with self._lock:
            keys=[
                key for key in self._embeddings
                if key[1] != space and self._records.get(key[0], None) is not None
                and self._records[key[0]].owner_id == owner_id
            ]
            for key in keys:
                self._embeddings.pop(key, None)
            for key in list(self._index_failures):
                if key[1] != space and self._records.get(key[0], None) is not None                         and self._records[key[0]].owner_id == owner_id:
                    self._index_failures.pop(key, None)
            return len(keys)

    def index_counts(self, owner_id: str, space: str | None) -> IndexCounts:
        visible=self.iter_visible(owner_id)
        current=self.embeddings(owner_id, space) if space is not None else {}
        with self._lock:
            stored=[
                key for key in self._embeddings
                if (record := self._records.get(key[0])) is not None and record.owner_id == owner_id
            ]
            failures=[
                item for (memory_id, item_space), item in self._index_failures.items()
                if item_space == (space or "")
                and (record := self._records.get(memory_id)) is not None
                and record.owner_id == owner_id
            ]
        visible_ids={item.id for item in visible}
        in_space={memory_id for memory_id, stored_space in stored if stored_space == space}
        return IndexCounts(
            eligible=len(visible), indexed=len(current),
            stale=len((in_space & visible_ids)-set(current)), failed=len(failures),
            other_spaces=sum(1 for _, stored_space in stored if stored_space != space),
            failures=tuple(sorted(failures, key=lambda item: (item["attempted_at"], item["memory_id"]))),
        )

    def add_training_example(
        self, *, owner_id: str, task_category: str, input_text: str,
        target_text: str, source: str, consent_id: str,
    ) -> LearningExample:
        self._ensure_open()
        if not all(value.strip() for value in (task_category, input_text, target_text, source)):
            raise ValueError("task_category, input_text, target_text, and source are required")
        self._consent(owner_id, consent_id, "personal_learning")
        with self._lock:
            if len(self._training_examples) >= self.max_training_examples:
                raise ValueError("in-memory training-example capacity reached")
            item=LearningExample(
                id=str(uuid4()), owner_id=owner_id, task_category=task_category,
                input_text=input_text, target_text=target_text, source=source,
                consent_id=consent_id, status="pending_review", created_at=self._now(),
            )
            self._training_examples[item.id]=item
            return item

    def list_training_examples(
        self, owner_id: str, *, status: str | None = None, limit: int = 100,
    ) -> builtins.list[LearningExample]:
        self._ensure_open()
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        if status not in (None, "pending_review", "approved", "rejected"):
            raise ValueError("invalid training-example status")
        with self._lock:
            items=[
                item for item in self._training_examples.values()
                if item.owner_id == owner_id
                and (status is None or item.status == status)
                and (grant := self._consents.get(item.consent_id)) is not None
                and grant.active
            ]
        return sorted(items, key=lambda item: (item.created_at, item.id), reverse=True)[:limit]

    def review_training_example(
        self, owner_id: str, example_id: str, *, decision: str, reviewer: str, note: str = "",
    ) -> bool:
        self._ensure_open()
        if decision not in {"approved", "rejected"}:
            raise ValueError("decision must be approved or rejected")
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        with self._lock:
            item=self._training_examples.get(example_id)
            if item is None or item.owner_id != owner_id or item.status != "pending_review":
                return False
            grant=self._consents.get(item.consent_id)
            if grant is None or not grant.active or grant.purpose != "personal_learning":
                return False
            self._training_examples[example_id]=replace(
                item,status=decision,reviewer=reviewer,review_note=note,
            )
            return True

    def delete_training_example(self, owner_id: str, example_id: str) -> bool:
        self._ensure_open()
        with self._lock:
            item=self._training_examples.get(example_id)
            if item is None or item.owner_id != owner_id:
                return False
            self._training_examples.pop(example_id,None)
            return True

    def export_training_corpus(self, owner_id: str) -> builtins.list[dict[str, str]]:
        return [
            {
                "id": item.id, "task_category": item.task_category,
                "input": item.input_text, "target": item.target_text,
                "source": item.source, "created_at": item.created_at,
            }
            for item in self.list_training_examples(owner_id,status="approved",limit=500)
        ]
