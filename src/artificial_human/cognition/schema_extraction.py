from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from threading import RLock
from typing import Any
from uuid import uuid4

from artificial_human.cognition.episodic_replay import EpisodicReplayMemory
from artificial_human.core.models import MemoryKind
from artificial_human.memory.repository import (
    MemoryNotFoundError,
    MemoryRecord,
    MemoryRepository,
)


_EVENT_TYPE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_ATTRIBUTE_KEY = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_SENSITIVE_ATTRIBUTE_KEYS = frozenset(
    {
        "api_key", "access_token", "auth_token", "authorization", "cookie",
        "email", "email_address", "password", "phone", "phone_number", "secret",
        "session_id", "token", "ssn", "social_security_number", "card_number",
        "credit_card", "cvv", "account_number", "bank_account", "ip_address",
    }
)
Primitive = str | int | float | bool
PatternKey = tuple[tuple[str, ...], str]


@dataclass(frozen=True, slots=True)
class SchemaProposal:
    """An owner-reviewable generalisation over multiple distinct event sequences."""

    event_types: tuple[str, ...]
    source_sequences: tuple[tuple[str, ...], ...]
    schema: dict[str, Any]
    support_count: int
    summary: str
    status: str = "pending_review"
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    reviewed_at: str | None = None
    review_note: str = ""
    schema_memory_id: str | None = None

    @property
    def source_memory_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(memory_id for sequence in self.source_sequences for memory_id in sequence)
        )


@dataclass(frozen=True, slots=True)
class _StoredProposal:
    proposal: SchemaProposal
    owner_scope: str
    consent_id: str
    fingerprint: str


class SchemaExtractionSystem:
    """Induce conservative event schemas from explicitly annotated owner memories.

    This system does not parse free text into event types. It requires owner-authored
    `metadata.event_type` and optional primitive `metadata.attributes`. Event-type
    sequences must recur in at least two distinct hashed owner/session contexts and
    use different source-event records. Attributes shared by all examples remain
    constants; differing values become typed slots. No varying attribute values or
    source text are copied into a proposal.

    Proposals are bounded and process-local. Preview never changes persistent memory.
    Explicit owner approval is required before a procedural/inferred schema memory
    is created, with provenance to every source example.
    """

    def __init__(
        self,
        *,
        min_distinct_sessions: int = 2,
        min_distinct_examples: int = 2,
        max_sequence_length: int = 4,
        max_proposals: int = 100,
    ) -> None:
        for name, value, minimum in (
            ("min_distinct_sessions", min_distinct_sessions, 2),
            ("min_distinct_examples", min_distinct_examples, 2),
            ("max_sequence_length", max_sequence_length, 2),
            ("max_proposals", max_proposals, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        self.min_distinct_sessions = min_distinct_sessions
        self.min_distinct_examples = min_distinct_examples
        self.max_sequence_length = max_sequence_length
        self.max_proposals = max_proposals
        self._proposals: OrderedDict[str, _StoredProposal] = OrderedDict()
        self._fingerprints: dict[tuple[str, str], str] = {}
        self._lock = RLock()

    @staticmethod
    def _owner_scope(owner_id: str) -> str:
        return hashlib.sha256(owner_id.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _event_descriptor(
        record: MemoryRecord,
    ) -> tuple[str, dict[str, Primitive]] | None:
        if record.kind is not MemoryKind.EPISODIC:
            return None
        raw_type = record.metadata.get("event_type")
        if not isinstance(raw_type, str):
            return None
        event_type = raw_type.strip().lower()
        if not _EVENT_TYPE.fullmatch(event_type):
            return None
        raw_attributes = record.metadata.get("attributes", {})
        if not isinstance(raw_attributes, dict):
            return None
        attributes: dict[str, Primitive] = {}
        for raw_key, value in raw_attributes.items():
            if not isinstance(raw_key, str):
                continue
            key = raw_key.strip().lower()
            if (
                not _ATTRIBUTE_KEY.fullmatch(key)
                or key in _SENSITIVE_ATTRIBUTE_KEYS
                or key.endswith(("_token", "_secret", "_password"))
            ):
                continue
            if isinstance(value, bool):
                attributes[key] = value
            elif isinstance(value, int):
                attributes[key] = value
            elif isinstance(value, float):
                if value == value and abs(value) != float("inf"):
                    attributes[key] = value
            elif isinstance(value, str) and 0 < len(value) <= 120:
                attributes[key] = value
        return event_type, attributes

    @staticmethod
    def _value_type(values: tuple[Primitive, ...]) -> str:
        kinds = {
            "boolean" if isinstance(value, bool)
            else "integer" if isinstance(value, int)
            else "number" if isinstance(value, float)
            else "string"
            for value in values
        }
        return next(iter(kinds)) if len(kinds) == 1 else "mixed"

    @classmethod
    def _induce_schema(
        cls,
        event_types: tuple[str, ...],
        examples: tuple[tuple[MemoryRecord, ...], ...],
    ) -> dict[str, Any]:
        steps: list[dict[str, Any]] = []
        for position, event_type in enumerate(event_types):
            attribute_maps: list[dict[str, Primitive]] = []
            for example in examples:
                descriptor = cls._event_descriptor(example[position])
                if descriptor is None:
                    raise ValueError("Every schema source event must remain annotated")
                attribute_maps.append(descriptor[1])
            common_keys = set(attribute_maps[0])
            for attributes in attribute_maps[1:]:
                common_keys.intersection_update(attributes)
            induced: dict[str, dict[str, Any]] = {}
            for key in sorted(common_keys):
                values = tuple(attributes[key] for attributes in attribute_maps)
                unique = {json.dumps(value, sort_keys=True, separators=(",", ":")) for value in values}
                if len(unique) == 1:
                    induced[key] = {"kind": "constant", "value": values[0]}
                else:
                    induced[key] = {
                        "kind": "slot",
                        "name": f"step_{position}_{key}",
                        "value_type": cls._value_type(values),
                    }
            steps.append({"index": position, "event_type": event_type, "attributes": induced})
        return {
            "version": 1,
            "pattern_type": "ordered_event_sequence",
            "steps": steps,
            "support": {
                "distinct_sessions": len(examples),
                "distinct_examples": len(examples),
            },
        }

    @staticmethod
    def _is_conflicted(
        records: tuple[MemoryRecord, ...],
        contradictions: set[frozenset[str]],
    ) -> bool:
        ids = [record.id for record in records]
        return any(
            frozenset((left, right)) in contradictions
            for index, left in enumerate(ids)
            for right in ids[index + 1 :]
        )

    @staticmethod
    def _contains_contiguous(
        longer: tuple[str, ...], shorter: tuple[str, ...]
    ) -> bool:
        return any(
            longer[index : index + len(shorter)] == shorter
            for index in range(len(longer) - len(shorter) + 1)
        )

    def prepare_from_episodes(
        self,
        *,
        owner_id: str,
        context_prefix: str,
        episodic_memory: EpisodicReplayMemory,
        repository: MemoryRepository,
    ) -> tuple[SchemaProposal, ...]:
        """Find repeated annotated sequences and propose fixed/variable attributes."""
        episodes = episodic_memory.episodes_for_context_prefix(context_prefix)
        record_cache: dict[str, MemoryRecord | None] = {}

        def resolve(memory_id: str) -> MemoryRecord | None:
            if memory_id not in record_cache:
                try:
                    record_cache[memory_id] = repository.get(owner_id, memory_id)
                except MemoryNotFoundError:
                    record_cache[memory_id] = None
            return record_cache[memory_id]

        links = repository.links(owner_id)
        contradictions = {
            frozenset((link.from_memory_id, link.to_memory_id))
            for link in links if link.relation == "contradicts"
        }
        grouped: dict[PatternKey, dict[str, tuple[MemoryRecord, ...]]] = {}
        for context_id, episode in episodes:
            records = tuple(resolve(event.memory_id) for event in episode)
            for length in range(2, min(self.max_sequence_length, len(records)) + 1):
                for start in range(len(records) - length + 1):
                    window = records[start : start + length]
                    if any(record is None for record in window):
                        continue
                    selected = tuple(record for record in window if record is not None)
                    consent_ids = {record.consent_id for record in selected}
                    if (
                        len(consent_ids) != 1
                        or self._is_conflicted(selected, contradictions)
                    ):
                        continue
                    descriptors = tuple(self._event_descriptor(record) for record in selected)
                    if any(item is None for item in descriptors):
                        continue
                    event_types = tuple(
                        item[0] for item in descriptors if item is not None
                    )
                    consent_id = next(iter(consent_ids))
                    grouped.setdefault((event_types, consent_id), {}).setdefault(
                        context_id, selected
                    )

        candidates: list[
            tuple[PatternKey, dict[str, tuple[MemoryRecord, ...]]]
        ] = []
        for key, by_context in grouped.items():
            distinct: list[tuple[MemoryRecord, ...]] = []
            seen_examples: set[tuple[str, ...]] = set()
            for context_id in sorted(by_context):
                example = by_context[context_id]
                source_ids = tuple(record.id for record in example)
                # Replaying the exact same source records in another context is
                # not independent evidence of a reusable schema.
                if source_ids in seen_examples:
                    continue
                seen_examples.add(source_ids)
                distinct.append(example)
            if (
                len(by_context) >= self.min_distinct_sessions
                and len(distinct) >= self.min_distinct_examples
            ):
                candidates.append((key, {str(index): example for index, example in enumerate(distinct)}))

        # Keep maximal recurring sequences. Shorter sub-patterns are omitted when a
        # longer recurring sequence covers at least the same session support.
        selected_candidates: list[
            tuple[PatternKey, dict[str, tuple[MemoryRecord, ...]]]
        ] = []
        for key, examples in sorted(
            candidates, key=lambda item: (-len(item[0][0]), item[0][0], item[0][1])
        ):
            event_types, consent_id = key
            covered = any(
                other_key[1] == consent_id
                and len(other_key[0]) > len(event_types)
                and self._contains_contiguous(other_key[0], event_types)
                and len(other_examples) >= len(examples)
                for other_key, other_examples in candidates
            )
            if not covered:
                selected_candidates.append((key, examples))

        created: list[SchemaProposal] = []
        for (event_types, consent_id), indexed_examples in selected_candidates:
            examples = tuple(indexed_examples[key] for key in sorted(indexed_examples, key=int))
            schema = self._induce_schema(event_types, examples)
            source_sequences = tuple(
                tuple(record.id for record in example) for example in examples
            )
            source_ids = tuple(
                dict.fromkeys(memory_id for sequence in source_sequences for memory_id in sequence)
            )
            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "consent_id": consent_id,
                        "event_types": event_types,
                        "source_sequences": source_sequences,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            owner_scope = self._owner_scope(owner_id)
            key = (owner_scope, fingerprint)
            with self._lock:
                if key in self._fingerprints:
                    continue
                if len(self._proposals) >= self.max_proposals:
                    evictable = next(
                        (
                            proposal_id for proposal_id, stored in self._proposals.items()
                            if stored.proposal.status != "pending_review"
                        ),
                        None,
                    )
                    if evictable is None:
                        continue
                    evicted = self._proposals.pop(evictable)
                    self._fingerprints.pop(
                        (evicted.owner_scope, evicted.fingerprint), None
                    )
                proposal = SchemaProposal(
                    event_types=event_types,
                    source_sequences=source_sequences,
                    schema=deepcopy(schema),
                    support_count=len(examples),
                    summary=(
                        "Recurring ordered event schema observed in "
                        f"{len(examples)} distinct owner/session examples: "
                        + " → ".join(event_types)
                        + ". Varying annotated attributes are represented as typed slots; "
                        "owner approval is required before the schema is saved."
                    ),
                )
                self._proposals[proposal.id] = _StoredProposal(
                    proposal=proposal,
                    owner_scope=owner_scope,
                    consent_id=consent_id,
                    fingerprint=fingerprint,
                )
                self._fingerprints[key] = proposal.id
                created.append(proposal)
        return tuple(created)

    def proposal_context(
        self, owner_id: str, proposal_id: str
    ) -> tuple[SchemaProposal, str]:
        with self._lock:
            stored = self._proposals.get(proposal_id)
            if stored is None or stored.owner_scope != self._owner_scope(owner_id):
                raise KeyError(proposal_id)
            if stored.proposal.status != "pending_review":
                raise ValueError("Only pending schema proposals can be reviewed")
            return replace(
                stored.proposal, schema=deepcopy(stored.proposal.schema)
            ), stored.consent_id

    def list_proposals(
        self,
        owner_id: str,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> tuple[SchemaProposal, ...]:
        if status not in (None, "pending_review", "approved", "rejected"):
            raise ValueError("status must be pending_review, approved, rejected, or omitted")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        owner_scope = self._owner_scope(owner_id)
        with self._lock:
            selected = [
                replace(stored.proposal, schema=deepcopy(stored.proposal.schema))
                for stored in reversed(tuple(self._proposals.values()))
                if stored.owner_scope == owner_scope
                and (status is None or stored.proposal.status == status)
            ]
        return tuple(selected[:limit])

    def approve_with(
        self,
        owner_id: str,
        proposal_id: str,
        *,
        reviewer: str,
        note: str = "",
        persist: Callable[[SchemaProposal, str], str],
    ) -> tuple[SchemaProposal, str]:
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        with self._lock:
            stored = self._proposals.get(proposal_id)
            if stored is None or stored.owner_scope != self._owner_scope(owner_id):
                raise KeyError(proposal_id)
            if stored.proposal.status != "pending_review":
                raise ValueError("Only pending schema proposals can be reviewed")
            memory_id = persist(
                replace(stored.proposal, schema=deepcopy(stored.proposal.schema)),
                stored.consent_id,
            )
            if not isinstance(memory_id, str) or not memory_id:
                raise ValueError("Schema persistence must return a memory identifier")
            reviewed = replace(
                stored.proposal,
                status="approved",
                reviewed_at=datetime.now(UTC).isoformat(),
                review_note=note[:2_000],
                schema_memory_id=memory_id,
            )
            self._proposals[proposal_id] = replace(stored, proposal=reviewed)
            return replace(reviewed, schema=deepcopy(reviewed.schema)), memory_id

    def review(
        self,
        owner_id: str,
        proposal_id: str,
        *,
        decision: str,
        reviewer: str,
        note: str = "",
    ) -> SchemaProposal:
        if decision not in ("approved", "rejected"):
            raise ValueError("decision must be approved or rejected")
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        with self._lock:
            stored = self._proposals.get(proposal_id)
            if stored is None or stored.owner_scope != self._owner_scope(owner_id):
                raise KeyError(proposal_id)
            if stored.proposal.status != "pending_review":
                raise ValueError("Only pending schema proposals can be reviewed")
            if decision == "approved":
                raise ValueError("Approved schemas must be saved through approve_with")
            reviewed = replace(
                stored.proposal,
                status="rejected",
                reviewed_at=datetime.now(UTC).isoformat(),
                review_note=note[:2_000],
            )
            self._proposals[proposal_id] = replace(stored, proposal=reviewed)
            return replace(reviewed, schema=deepcopy(reviewed.schema))

    def clear_consent(self, owner_id: str, consent_id: str) -> int:
        return self._remove_matching(
            lambda item: item.owner_scope == self._owner_scope(owner_id)
            and item.consent_id == consent_id
        )

    def forget_memory(self, owner_id: str, memory_id: str) -> int:
        return self._remove_matching(
            lambda item: item.owner_scope == self._owner_scope(owner_id)
            and memory_id in item.proposal.source_memory_ids
        )

    def clear_owner(self, owner_id: str) -> int:
        owner_scope = self._owner_scope(owner_id)
        return self._remove_matching(lambda item: item.owner_scope == owner_scope)

    def _remove_matching(
        self, predicate: Callable[[_StoredProposal], bool]
    ) -> int:
        with self._lock:
            selected = [
                proposal_id for proposal_id, stored in self._proposals.items()
                if predicate(stored)
            ]
            for proposal_id in selected:
                stored = self._proposals.pop(proposal_id)
                self._fingerprints.pop((stored.owner_scope, stored.fingerprint), None)
            return len(selected)

    def snapshot(self, owner_id: str) -> dict[str, int | str]:
        owner_scope = self._owner_scope(owner_id)
        with self._lock:
            proposals = [
                item.proposal for item in self._proposals.values()
                if item.owner_scope == owner_scope
            ]
        return {
            "pending_review": sum(item.status == "pending_review" for item in proposals),
            "approved": sum(item.status == "approved" for item in proposals),
            "rejected": sum(item.status == "rejected" for item in proposals),
            "max_proposals": self.max_proposals,
            "storage": "volatile_process_memory",
        }
