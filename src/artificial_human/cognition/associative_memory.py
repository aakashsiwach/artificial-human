from __future__ import annotations

import re
from dataclasses import dataclass
from threading import RLock
from typing import Final

from artificial_human.cognition.core_knowledge import (
    CORE_KNOWLEDGE,
    CoreKnowledgeFact,
    retrieve_core_knowledge,
)


ASSOCIATIVE_MEMORY_VERSION: Final[str] = "source-knowledge-association-v2"
DEFAULT_ASSOCIATION_CONTEXT_ID: Final[str] = "global"
_CONTEXT_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


@dataclass(frozen=True, slots=True)
class KnowledgeActivation:
    fact: CoreKnowledgeFact
    activation: float
    directly_activated: bool
    spread_activation: float
    association_context_id: str | None = None


class AssociativeKnowledgeNetwork:
    """A bounded in-process associative graph over source-defined facts.

    Links are partitioned by a caller-supplied, non-content context identifier.
    Spreading activation only follows links learned in the matching context.
    This prevents an association learned in one conversation from silently
    influencing a different conversation.

    Only source fact identifiers, context identifiers, and association weights
    are retained. Context identifiers must be short non-content labels; callers
    should hash opaque session IDs rather than pass raw user-provided values.
    User text, personal details, and raw prompts are never stored by this network.

    The total link budget remains global, so capacity pressure can evict a weak
    link from another context. This is an engineering abstraction, not a model
    of hippocampal tissue or proof of biological memory.
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.08,
        decay: float = 0.995,
        max_links: int = 128,
    ) -> None:
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be between 0 and 1")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be between 0 and 1")
        if isinstance(max_links, bool) or not isinstance(max_links, int) or max_links < 1:
            raise ValueError("max_links must be a positive integer")
        self.learning_rate = learning_rate
        self.decay = decay
        self.max_links = max_links
        self._facts = {item.fact_id: item for item in CORE_KNOWLEDGE}
        # Key: (context_id, lexically_sorted_fact_id_a, lexically_sorted_fact_id_b).
        self._links: dict[tuple[str, str, str], float] = {}
        self._coactivations = 0
        self._inputs_processed = 0
        self._lock = RLock()

    @staticmethod
    def _validate_context_id(context_id: str) -> None:
        if not isinstance(context_id, str) or not _CONTEXT_ID_PATTERN.fullmatch(context_id):
            raise ValueError(
                "context_id must be a 1-64 character lowercase identifier "
                "containing only letters, digits, dots, underscores, or hyphens"
            )

    @staticmethod
    def _link_key(context_id: str, first: str, second: str) -> tuple[str, str, str]:
        left, right = (first, second) if first <= second else (second, first)
        return context_id, left, right

    def process(
        self,
        query: str,
        *,
        limit: int = 5,
        context_id: str = DEFAULT_ASSOCIATION_CONTEXT_ID,
    ) -> tuple[KnowledgeActivation, ...]:
        """Retrieve facts, learn scoped co-activation links, then spread within scope."""
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")
        self._validate_context_id(context_id)
        if not query.strip():
            return ()

        direct_facts = retrieve_core_knowledge(query, limit=limit)
        direct = {
            fact.fact_id: max(0.25, 1.0 - (position * 0.12))
            for position, fact in enumerate(direct_facts)
        }
        with self._lock:
            self._inputs_processed += 1

            # Interference/forgetting advances only within the active context.
            # Unrelated conversations do not age one another's associations.
            decayed_links: dict[tuple[str, str, str], float] = {}
            for key, strength in self._links.items():
                if key[0] != context_id:
                    decayed_links[key] = strength
                    continue
                decayed = strength * self.decay
                if decayed >= 0.001:
                    decayed_links[key] = decayed
            self._links = decayed_links

            if not direct:
                return ()

            identifiers = list(direct)
            for index, first in enumerate(identifiers):
                for second in identifiers[index + 1:]:
                    key = self._link_key(context_id, first, second)
                    current = self._links.get(key, 0.0)
                    updated = current + (
                        self.learning_rate * direct[first] * direct[second] * (1.0 - current)
                    )
                    if key not in self._links and len(self._links) >= self.max_links:
                        weakest = min(self._links, key=lambda item: (self._links[item], item))
                        if self._links[weakest] >= updated:
                            continue
                        del self._links[weakest]
                    self._links[key] = min(1.0, updated)
                    self._coactivations += 1

            spread: dict[str, float] = {}
            for (link_context, first, second), strength in self._links.items():
                if link_context != context_id:
                    continue
                if first in direct and second not in direct:
                    spread[second] = max(
                        spread.get(second, 0.0), direct[first] * strength
                    )
                elif second in direct and first not in direct:
                    spread[first] = max(
                        spread.get(first, 0.0), direct[second] * strength
                    )

            activations = {
                fact_id: (activation + 0.55 * spread.get(fact_id, 0.0))
                for fact_id, activation in direct.items()
            }
            for fact_id, activation in spread.items():
                activations.setdefault(fact_id, 0.55 * activation)

            ranked = sorted(
                activations,
                key=lambda fact_id: (-activations[fact_id], fact_id),
            )[:limit]
            return tuple(
                KnowledgeActivation(
                    fact=self._facts[fact_id],
                    activation=round(activations[fact_id], 6),
                    directly_activated=fact_id in direct,
                    spread_activation=round(spread.get(fact_id, 0.0), 6),
                    association_context_id=(
                        context_id if spread.get(fact_id, 0.0) > 0.0 else None
                    ),
                )
                for fact_id in ranked
            )

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            strengths = tuple(self._links.values())
            contexts = {key[0] for key in self._links}
            return {
                "version": ASSOCIATIVE_MEMORY_VERSION,
                "source_fact_count": len(self._facts),
                "association_count": len(self._links),
                "association_context_count": len(contexts),
                "context_isolation": "spread_activation_within_matching_context",
                "coactivations": self._coactivations,
                "inputs_processed": self._inputs_processed,
                "mean_association_strength": (
                    round(sum(strengths) / len(strengths), 6) if strengths else 0.0
                ),
                "max_association_strength": round(max(strengths, default=0.0), 6),
                "state_storage": "volatile_process_memory",
                "retains_user_text": False,
            }

    def forget_context(self, context_id: str) -> bool:
        """Delete every learned association belonging to one context."""
        self._validate_context_id(context_id)
        with self._lock:
            selected = [key for key in self._links if key[0] == context_id]
            for key in selected:
                del self._links[key]
            return bool(selected)

    def forget_contexts(self, context_prefix: str) -> int:
        """Delete links for all contexts sharing a validated non-content prefix."""
        if (
            not isinstance(context_prefix, str)
            or not context_prefix
            or len(context_prefix) > 64
            or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", context_prefix)
        ):
            raise ValueError("context_prefix must be a short lowercase non-content prefix")
        with self._lock:
            selected = [
                key for key in self._links if key[0].startswith(context_prefix)
            ]
            for key in selected:
                del self._links[key]
            return len(selected)

    def reset(self) -> None:
        """Clear all learned links and counters without changing source facts."""
        with self._lock:
            self._links.clear()
            self._coactivations = 0
            self._inputs_processed = 0
