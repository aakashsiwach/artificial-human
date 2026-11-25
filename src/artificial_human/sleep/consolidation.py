from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from threading import RLock
from uuid import uuid4

from artificial_human.cognition.episodic_replay import EpisodicReplayMemory
from artificial_human.core.models import MemoryKind
from artificial_human.memory.repository import (
    MemoryNotFoundError,
    MemoryRecord,
    MemoryRepository,
)


@dataclass(frozen=True, slots=True)
class ConsolidationProposal:
    """An owner-reviewable synthesis candidate backed by repeated episodes."""

    source_memory_ids: tuple[str, ...]
    summary: str
    recurrence_count: int = 1
    status: str = "pending_review"
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    reviewed_at: str | None = None
    review_note: str = ""
    consolidated_memory_id: str | None = None


@dataclass(frozen=True, slots=True)
class _StoredProposal:
    proposal: ConsolidationProposal
    owner_scope: str
    consent_id: str
    fingerprint: str


class SleepSystem:
    """Finds repeated event patterns and queues proposals; never writes silently.

    Consolidation is intentionally owner-reviewed. The detector looks for the
    same directed adjacent episodic transition in at least two distinct hashed
    owner/session contexts. It proposes a deterministic summary, but it does not
    infer that the pattern is true, merge source records, or alter replay state.

    Pending proposals and their summaries are bounded and process-local. Approval
    is performed by the API using the same active consent grant as all source
    records, then the resulting semantic memory is marked as inferred and retains
    source IDs in metadata. A process restart discards unreviewed proposals.
    """

    def __init__(
        self,
        *,
        min_distinct_sessions: int = 2,
        max_proposals: int = 100,
    ) -> None:
        if (
            isinstance(min_distinct_sessions, bool)
            or not isinstance(min_distinct_sessions, int)
            or min_distinct_sessions < 2
        ):
            raise ValueError("min_distinct_sessions must be an integer of at least 2")
        if (
            isinstance(max_proposals, bool)
            or not isinstance(max_proposals, int)
            or max_proposals < 1
        ):
            raise ValueError("max_proposals must be a positive integer")
        self.min_distinct_sessions = min_distinct_sessions
        self.max_proposals = max_proposals
        self._proposals: OrderedDict[str, _StoredProposal] = OrderedDict()
        self._fingerprints: dict[tuple[str, str], str] = {}
        self._lock = RLock()

    @staticmethod
    def _owner_scope(owner_id: str) -> str:
        return hashlib.sha256(owner_id.encode("utf-8")).hexdigest()[:24]

    def prepare_from_episodes(
        self,
        *,
        owner_id: str,
        context_prefix: str,
        episodic_memory: EpisodicReplayMemory,
        repository: MemoryRepository,
    ) -> tuple[ConsolidationProposal, ...]:
        """Create proposals for adjacent event transitions repeated across sessions."""
        episodes = episodic_memory.episodes_for_context_prefix(context_prefix)
        contexts_by_transition: dict[tuple[str, str], set[str]] = {}
        for context_id, episode in episodes:
            for left_event, right_event in zip(episode, episode[1:], strict=False):
                if left_event.memory_id == right_event.memory_id:
                    continue
                contexts_by_transition.setdefault(
                    (left_event.memory_id, right_event.memory_id), set()
                ).add(context_id)

        visible_links = repository.links(owner_id)
        blocked_pairs = {
            frozenset((link.from_memory_id, link.to_memory_id))
            for link in visible_links if link.relation == "contradicts"
        }
        created: list[ConsolidationProposal] = []
        for (left_id, right_id), contexts in sorted(contexts_by_transition.items()):
            if len(contexts) < self.min_distinct_sessions:
                continue
            if frozenset((left_id, right_id)) in blocked_pairs:
                continue
            try:
                left_record = repository.get(owner_id, left_id)
                right_record = repository.get(owner_id, right_id)
            except MemoryNotFoundError:
                # Deleted, revoked, or foreign source records are never revived.
                continue
            if (
                left_record.kind is not MemoryKind.EPISODIC
                or right_record.kind is not MemoryKind.EPISODIC
                or left_record.consent_id != right_record.consent_id
            ):
                continue

            fingerprint = hashlib.sha256(
                "|".join((owner_id, left_record.consent_id, left_id, right_id)).encode("utf-8")
            ).hexdigest()
            owner_scope = self._owner_scope(owner_id)
            key = (owner_scope, fingerprint)
            with self._lock:
                existing_id = self._fingerprints.get(key)
                if existing_id is not None:
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
                    self._fingerprints.pop((evicted.owner_scope, evicted.fingerprint), None)

                summary = (
                    f"Recurring event pattern observed in {len(contexts)} distinct sessions:\n"
                    f"1. {left_record.content[:240]}\n"
                    f"2. {right_record.content[:240]}\n\n"
                    "This is an inferred candidate based on repeated temporal co-occurrence, "
                    "not an established fact. Owner approval is required before it is saved."
                )
                proposal = ConsolidationProposal(
                    source_memory_ids=(left_record.id, right_record.id),
                    summary=summary,
                    recurrence_count=len(contexts),
                )
                self._proposals[proposal.id] = _StoredProposal(
                    proposal=proposal,
                    owner_scope=owner_scope,
                    consent_id=left_record.consent_id,
                    fingerprint=fingerprint,
                )
                self._fingerprints[key] = proposal.id
                created.append(proposal)
        return tuple(created)

    def prepare(self, memories: list[MemoryRecord]) -> ConsolidationProposal | None:
        """Compatibility preview for callers without episodic replay history."""
        if len(memories) < 2:
            return None
        kinds = sorted({item.kind.value for item in memories})
        return ConsolidationProposal(
            source_memory_ids=tuple(item.id for item in memories),
            summary=(
                f"Review {len(memories)} memories across {', '.join(kinds)}. "
                "No synthesis is persisted until the owner approves it."
            ),
        )

    def proposal_context(
        self, owner_id: str, proposal_id: str
    ) -> tuple[ConsolidationProposal, str]:
        """Return a pending owner's proposal and its consent ID for atomic approval."""
        with self._lock:
            stored = self._proposals.get(proposal_id)
            if stored is None or stored.owner_scope != self._owner_scope(owner_id):
                raise KeyError(proposal_id)
            if stored.proposal.status != "pending_review":
                raise ValueError("Only pending proposals can be reviewed")
            return stored.proposal, stored.consent_id

    def get_proposal(self, owner_id: str, proposal_id: str) -> ConsolidationProposal:
        with self._lock:
            stored = self._proposals.get(proposal_id)
            if stored is None or stored.owner_scope != self._owner_scope(owner_id):
                raise KeyError(proposal_id)
            return stored.proposal

    def list_proposals(
        self,
        owner_id: str,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> tuple[ConsolidationProposal, ...]:
        if status not in (None, "pending_review", "approved", "rejected"):
            raise ValueError("status must be pending_review, approved, rejected, or omitted")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        with self._lock:
            selected = [
                stored.proposal
                for stored in reversed(tuple(self._proposals.values()))
                if stored.owner_scope == self._owner_scope(owner_id)
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
        persist: Callable[[ConsolidationProposal, str], str],
    ) -> tuple[ConsolidationProposal, str]:
        """Persist and mark approval under one queue lock to prevent duplicate writes.

        The callback must atomically revalidate consent and source records in the
        repository. The queue lock serializes concurrent approvals for this
        process; after the callback succeeds, the proposal is marked approved
        before another request can observe it as pending.
        """
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        with self._lock:
            stored = self._proposals.get(proposal_id)
            if stored is None or stored.owner_scope != self._owner_scope(owner_id):
                raise KeyError(proposal_id)
            if stored.proposal.status != "pending_review":
                raise ValueError("Only pending proposals can be reviewed")
            memory_id = persist(stored.proposal, stored.consent_id)
            if not isinstance(memory_id, str) or not memory_id:
                raise ValueError("Approval persistence must return a memory identifier")
            reviewed = replace(
                stored.proposal,
                status="approved",
                reviewed_at=datetime.now(UTC).isoformat(),
                review_note=note[:2_000],
                consolidated_memory_id=memory_id,
            )
            self._proposals[proposal_id] = replace(stored, proposal=reviewed)
            return reviewed, memory_id

    def review(
        self,
        owner_id: str,
        proposal_id: str,
        *,
        decision: str,
        reviewer: str,
        note: str = "",
        consolidated_memory_id: str | None = None,
    ) -> ConsolidationProposal:
        if decision not in ("approved", "rejected"):
            raise ValueError("decision must be approved or rejected")
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        with self._lock:
            stored = self._proposals.get(proposal_id)
            if stored is None or stored.owner_scope != self._owner_scope(owner_id):
                raise KeyError(proposal_id)
            if stored.proposal.status != "pending_review":
                raise ValueError("Only pending proposals can be reviewed")
            if decision == "approved" and not consolidated_memory_id:
                raise ValueError("An approved proposal must reference its saved memory")
            reviewed = replace(
                stored.proposal,
                status=decision,
                reviewed_at=datetime.now(UTC).isoformat(),
                review_note=note[:2_000],
                consolidated_memory_id=consolidated_memory_id,
            )
            self._proposals[proposal_id] = replace(stored, proposal=reviewed)
            return reviewed

    def clear_consent(self, owner_id: str, consent_id: str) -> int:
        """Forget pending and reviewed proposals derived under a revoked grant."""
        return self._remove_matching(
            lambda item: item.owner_scope == self._owner_scope(owner_id)
            and item.consent_id == consent_id
        )

    def forget_memory(self, owner_id: str, memory_id: str) -> int:
        """Invalidate proposals whose evidence was corrected, superseded, or deleted."""
        return self._remove_matching(
            lambda item: item.owner_scope == self._owner_scope(owner_id)
            and memory_id in item.proposal.source_memory_ids
        )

    def clear_owner(self, owner_id: str) -> int:
        return self._remove_matching(
            lambda item: item.owner_scope == self._owner_scope(owner_id)
        )

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
        with self._lock:
            proposals = [
                item.proposal for item in self._proposals.values()
                if item.owner_scope == self._owner_scope(owner_id)
            ]
        return {
            "pending_review": sum(item.status == "pending_review" for item in proposals),
            "approved": sum(item.status == "approved" for item in proposals),
            "rejected": sum(item.status == "rejected" for item in proposals),
            "max_proposals": self.max_proposals,
            "storage": "volatile_process_memory",
        }
