from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Final

from artificial_human.cognition.core_knowledge import (
    CORE_KNOWLEDGE,
    CoreKnowledgeFact,
    retrieve_core_knowledge,
)


ASSOCIATIVE_MEMORY_VERSION: Final[str] = "source-knowledge-association-v1"


@dataclass(frozen=True, slots=True)
class KnowledgeActivation:
    fact: CoreKnowledgeFact
    activation: float
    directly_activated: bool
    spread_activation: float


class AssociativeKnowledgeNetwork:
    """A bounded in-process associative graph over source-defined facts.

    Only source fact identifiers and association weights are retained. User text,
    personal details, and raw prompts are never stored by this network. Co-activation
    strengthens an undirected link with a bounded Hebbian-style rule; links decay
    slowly on subsequent inputs. This is an engineering abstraction, not a model
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
        self._links: dict[tuple[str, str], float] = {}
        self._coactivations = 0
        self._inputs_processed = 0
        self._lock = RLock()

    @staticmethod
    def _link_key(first: str, second: str) -> tuple[str, str]:
        return (first, second) if first <= second else (second, first)

    def process(self, query: str, *, limit: int = 5) -> tuple[KnowledgeActivation, ...]:
        """Retrieve facts, learn co-activation links, then spread activation."""
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")
        if not query.strip():
            return ()
        direct_facts = retrieve_core_knowledge(query, limit=limit)
        direct = {
            fact.fact_id: max(0.25, 1.0 - (position * 0.12))
            for position, fact in enumerate(direct_facts)
        }
        with self._lock:
            self._inputs_processed += 1

            # Forgetting/interference is bounded and applies to every new input,
            # including inputs that do not match any source-defined fact.
            self._links = {
                key: strength * self.decay
                for key, strength in self._links.items()
                if strength * self.decay >= 0.001
            }
            if not direct:
                return ()

            identifiers = list(direct)
            for index, first in enumerate(identifiers):
                for second in identifiers[index + 1:]:
                    key = self._link_key(first, second)
                    current = self._links.get(key, 0.0)
                    updated = current + (
                        self.learning_rate * direct[first] * direct[second] * (1.0 - current)
                    )
                    if key not in self._links and len(self._links) >= self.max_links:
                        weakest = min(self._links, key=lambda item: self._links[item])
                        if self._links[weakest] >= updated:
                            continue
                        del self._links[weakest]
                    self._links[key] = min(1.0, updated)
                    self._coactivations += 1

            spread: dict[str, float] = {}
            for (first, second), strength in self._links.items():
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
                )
                for fact_id in ranked
            )

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            strengths = tuple(self._links.values())
            return {
                "version": ASSOCIATIVE_MEMORY_VERSION,
                "source_fact_count": len(self._facts),
                "association_count": len(self._links),
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
