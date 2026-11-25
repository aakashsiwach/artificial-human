from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from threading import RLock
from typing import Final


_RELATIONAL_MEMORY_VERSION: Final[str] = "contextual-relational-memory-v1"
_CONTEXT_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_MEMORY_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@dataclass(frozen=True, slots=True)
class AssociatedMemory:
    """One memory identifier activated through a previously learned relation."""

    memory_id: str
    strength: float


class RelationalMemoryNetwork:
    """Bounded, volatile co-retrieval links over opaque memory IDs.

    The graph never stores memory text, source labels, owner IDs, or raw session IDs.
    A caller supplies owner/session-scoped context identifiers and explicit
    contradiction pairs from the consent-visible memory repository.

    Positive links are learned only among memories selected by attention. When an
    explicit contradiction pair is observed, any positive link between those two
    records is removed and is not relearned during that cycle. Contradiction
    relations are supplied afresh from the repository; they are not duplicated
    as long-lived graph state.

    Retrieval-dependent strengthening is an engineering hypothesis: a relation
    used to complete a cue receives a small bounded increment. A global edge budget
    creates measurable capacity interference. This is not a biological hippocampal
    model or evidence of human-like episodic memory.
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.08,
        retrieval_rate: float = 0.025,
        decay: float = 0.995,
        max_links: int = 256,
    ) -> None:
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be between 0 and 1")
        if not 0.0 <= retrieval_rate <= 1.0:
            raise ValueError("retrieval_rate must be between 0 and 1")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be between 0 and 1")
        if isinstance(max_links, bool) or not isinstance(max_links, int) or max_links < 1:
            raise ValueError("max_links must be a positive integer")
        self.learning_rate = learning_rate
        self.retrieval_rate = retrieval_rate
        self.decay = decay
        self.max_links = max_links
        self._links: dict[tuple[str, str, str], float] = {}
        self._inputs_processed = 0
        self._coactivations = 0
        self._retrieval_strengthenings = 0
        self._contradiction_suppressions = 0
        self._evicted_links = 0
        self._lock = RLock()

    @staticmethod
    def _validate_context(context_id: str) -> None:
        if not isinstance(context_id, str) or not _CONTEXT_PATTERN.fullmatch(context_id):
            raise ValueError("context_id must be a short lowercase non-content identifier")

    @staticmethod
    def _validate_memory_id(memory_id: str) -> None:
        if not isinstance(memory_id, str) or not _MEMORY_ID_PATTERN.fullmatch(memory_id):
            raise ValueError("memory_id must be a 1-64 character opaque identifier")

    @classmethod
    def _normalise_ids(cls, memory_ids: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        for memory_id in memory_ids:
            cls._validate_memory_id(memory_id)
        return tuple(sorted(set(memory_ids)))

    @classmethod
    def _normalise_conflicts(
        cls, contradictions: tuple[tuple[str, str], ...] | list[tuple[str, str]]
    ) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for first, second in contradictions:
            cls._validate_memory_id(first)
            cls._validate_memory_id(second)
            if first == second:
                raise ValueError("A memory cannot contradict itself")
            pairs.add((first, second) if first < second else (second, first))
        return pairs

    @staticmethod
    def _key(context_id: str, first: str, second: str) -> tuple[str, str, str]:
        left, right = (first, second) if first < second else (second, first)
        return context_id, left, right

    def activate(
        self,
        memory_ids: tuple[str, ...] | list[str],
        *,
        context_id: str,
        contradictions: tuple[tuple[str, str], ...] | list[tuple[str, str]] = (),
        limit: int = 3,
    ) -> tuple[AssociatedMemory, ...]:
        """Spread one hop from retrieved IDs, suppress conflicts, and strengthen used links."""
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")
        self._validate_context(context_id)
        direct = set(self._normalise_ids(memory_ids))
        conflicts = self._normalise_conflicts(contradictions)

        with self._lock:
            self._inputs_processed += 1

            # Decay advances only when this context is active.
            decayed: dict[tuple[str, str, str], float] = {}
            for key, strength in self._links.items():
                if key[0] != context_id:
                    decayed[key] = strength
                    continue
                next_strength = strength * self.decay
                if next_strength >= 0.001:
                    decayed[key] = next_strength
            self._links = decayed

            # Explicitly contradictory evidence removes any previously positive link.
            for first, second in conflicts:
                key = self._key(context_id, first, second)
                if key in self._links:
                    del self._links[key]
                    self._contradiction_suppressions += 1

            spread: dict[str, tuple[float, str]] = {}
            for (link_context, first, second), strength in self._links.items():
                if link_context != context_id or (first, second) in conflicts:
                    continue
                if first in direct and second not in direct:
                    candidate = (strength, first)
                    if second not in spread or candidate > spread[second]:
                        spread[second] = candidate
                elif second in direct and first not in direct:
                    candidate = (strength, second)
                    if first not in spread or candidate > spread[first]:
                        spread[first] = candidate

            ranked = sorted(
                spread.items(),
                key=lambda item: (-item[1][0], item[0]),
            )[:limit]

            # Successful cue completion strengthens the edge that supplied it.
            results: list[AssociatedMemory] = []
            for memory_id, (strength, cue_id) in ranked:
                key = self._key(context_id, cue_id, memory_id)
                current = self._links.get(key, strength)
                updated = current + self.retrieval_rate * (1.0 - current)
                self._links[key] = min(1.0, updated)
                self._retrieval_strengthenings += 1
                results.append(AssociatedMemory(memory_id=memory_id, strength=round(updated, 6)))
            return tuple(results)

    def learn(
        self,
        memory_ids: tuple[str, ...] | list[str],
        *,
        context_id: str,
        contradictions: tuple[tuple[str, str], ...] | list[tuple[str, str]] = (),
    ) -> None:
        """Strengthen co-attended records, excluding explicit contradiction pairs."""
        self._validate_context(context_id)
        direct = self._normalise_ids(memory_ids)
        conflicts = self._normalise_conflicts(contradictions)
        blocked = {
            self._key(context_id, first, second)
            for first, second in conflicts
        }

        with self._lock:
            for key in blocked:
                if key in self._links:
                    del self._links[key]
                    self._contradiction_suppressions += 1

            for first, second in combinations(direct, 2):
                key = self._key(context_id, first, second)
                if key in blocked:
                    continue
                current = self._links.get(key, 0.0)
                updated = current + self.learning_rate * (1.0 - current)
                if key not in self._links and len(self._links) >= self.max_links:
                    weakest = min(self._links, key=lambda item: (self._links[item], item))
                    if self._links[weakest] >= updated:
                        continue
                    del self._links[weakest]
                    self._evicted_links += 1
                self._links[key] = min(1.0, updated)
                self._coactivations += 1

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            strengths = tuple(self._links.values())
            return {
                "version": _RELATIONAL_MEMORY_VERSION,
                "association_count": len(self._links),
                "association_context_count": len({key[0] for key in self._links}),
                "inputs_processed": self._inputs_processed,
                "coactivations": self._coactivations,
                "retrieval_strengthenings": self._retrieval_strengthenings,
                "contradiction_suppressions": self._contradiction_suppressions,
                "evicted_links": self._evicted_links,
                "mean_association_strength": (
                    round(sum(strengths) / len(strengths), 6) if strengths else 0.0
                ),
                "max_association_strength": round(max(strengths, default=0.0), 6),
                "max_links": self.max_links,
                "state_storage": "volatile_process_memory",
                "retains_memory_content": False,
                "retains_owner_or_session_ids": False,
            }

    def forget_context(self, context_id: str) -> int:
        self._validate_context(context_id)
        with self._lock:
            selected = [key for key in self._links if key[0] == context_id]
            for key in selected:
                del self._links[key]
            return len(selected)

    def forget_contexts(self, context_prefix: str) -> int:
        if (
            not isinstance(context_prefix, str)
            or not context_prefix
            or len(context_prefix) > 64
            or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", context_prefix)
        ):
            raise ValueError("context_prefix must be a short lowercase non-content prefix")
        with self._lock:
            selected = [key for key in self._links if key[0].startswith(context_prefix)]
            for key in selected:
                del self._links[key]
            return len(selected)

    def reset(self) -> None:
        with self._lock:
            self._links.clear()
            self._inputs_processed = 0
            self._coactivations = 0
            self._retrieval_strengthenings = 0
            self._contradiction_suppressions = 0
            self._evicted_links = 0
