"""Distinct bounded in-process cognitive memory mechanisms.

These are deterministic software structures with separate encoding, retrieval,
strengthening and forgetting rules. They are not literal replicas of biological
memory systems and never persist runtime experience to disk.
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from threading import RLock
from uuid import uuid4

from artificial_human.cognition.episodic_replay import EpisodicReplayMemory
from artificial_human.cognition.relational_memory import RelationalMemoryNetwork
from artificial_human.cognition.working_memory import WorkingMemorySessions
from artificial_human.core.models import MemoryKind
from artificial_human.memory.repository import MemoryRecord


def _tokens(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", value.casefold()))


def _overlap(query: frozenset[str], candidate: frozenset[str]) -> float:
    if not query or not candidate:
        return 0.0
    return len(query & candidate) / len(query | candidate)


def _validate_capacity(capacity: int, name: str) -> None:
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class CognitiveMemoryRecall:
    episodic: tuple["EpisodicTrace", ...]
    semantic: tuple["SemanticFact", ...]
    procedural: tuple["Procedure", ...]
    spatial: tuple["SpatialPlace", ...]


@dataclass(frozen=True, slots=True)
class EpisodicTrace:
    id: str
    owner_id: str
    consent_id: str
    source_id: str
    context_id: str
    content: str
    occurred_at: int
    tags: tuple[str, ...]
    salience: float
    strength: float = 0.5
    access_count: int = 0


@dataclass(frozen=True, slots=True)
class SemanticFact:
    id: str
    owner_id: str
    consent_id: str
    concept: str
    attributes: tuple[tuple[str, str], ...]
    source_ids: tuple[str, ...]
    confidence: float
    strength: float
    access_count: int = 0


@dataclass(frozen=True, slots=True)
class SemanticAssociation:
    owner_id: str
    source_id: str
    target_id: str
    relation: str
    strength: float


@dataclass(frozen=True, slots=True)
class Procedure:
    id: str
    owner_id: str
    consent_id: str
    task: str
    steps: tuple[str, ...]
    source_ids: tuple[str, ...]
    successes: int = 0
    failures: int = 0
    strength: float = 0.5
    access_count: int = 0

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return (self.successes + 1.0) / (total + 2.0)


@dataclass(frozen=True, slots=True)
class SpatialPlace:
    id: str
    owner_id: str
    consent_id: str
    source_id: str
    label: str
    coordinates: tuple[float, float, float]
    context_id: str
    tags: tuple[str, ...]
    access_count: int = 0


@dataclass(frozen=True, slots=True)
class SpatialRelation:
    owner_id: str
    source_id: str
    target_id: str
    relation: str
    strength: float = 1.0


class EpisodicMemoryStore:
    """One-trial event memory with context/time binding and partial-cue recall."""

    def __init__(self, *, capacity: int = 2_000, retrieval_rate: float = 0.12) -> None:
        _validate_capacity(capacity, "episodic capacity")
        if not math.isfinite(retrieval_rate) or not 0.0 < retrieval_rate <= 1.0:
            raise ValueError("retrieval_rate must be in (0, 1]")
        self.capacity = capacity
        self.retrieval_rate = retrieval_rate
        self._items: dict[str, EpisodicTrace] = {}
        self._lock = RLock()
        self._evictions = 0
        self._retrievals = 0

    def store(
        self,
        *,
        owner_id: str,
        consent_id: str,
        source_id: str,
        context_id: str,
        content: str,
        occurred_at: int,
        tags: Sequence[str] = (),
        salience: float = 0.5,
    ) -> EpisodicTrace:
        if not all(value.strip() for value in (owner_id, consent_id, source_id, context_id, content)):
            raise ValueError("episodic memory requires owner, consent, source, context and content")
        if isinstance(occurred_at, bool) or not isinstance(occurred_at, int) or occurred_at < 0:
            raise ValueError("occurred_at must be a non-negative integer")
        if isinstance(salience, bool) or not math.isfinite(salience) or not 0.0 <= salience <= 1.0:
            raise ValueError("salience must be in [0, 1]")
        clean_tags = tuple(sorted({item.strip().casefold() for item in tags if item.strip()}))
        with self._lock:
            if any(item.source_id == source_id for item in self._items.values()):
                raise ValueError("episodic source already stored")
            if len(self._items) >= self.capacity:
                weakest = min(
                    self._items,
                    key=lambda key: (
                        self._items[key].salience * self._items[key].strength,
                        self._items[key].access_count,
                        self._items[key].occurred_at,
                        key,
                    ),
                )
                del self._items[weakest]
                self._evictions += 1
            item = EpisodicTrace(
                id=str(uuid4()), owner_id=owner_id, consent_id=consent_id,
                source_id=source_id, context_id=context_id, content=content,
                occurred_at=occurred_at, tags=clean_tags, salience=salience,
            )
            self._items[item.id] = item
            return item

    def retrieve(
        self,
        owner_id: str,
        query: str,
        *,
        context_id: str | None = None,
        limit: int = 5,
    ) -> tuple[EpisodicTrace, ...]:
        if not 1 <= limit <= 100:
            raise ValueError("episodic retrieval limit must be in [1, 100]")
        query_tokens = _tokens(query)
        if not query_tokens:
            return ()
        with self._lock:
            ranked: list[tuple[float, str]] = []
            for item in self._items.values():
                if item.owner_id != owner_id or (context_id and item.context_id != context_id):
                    continue
                candidate = _tokens(item.content) | frozenset(
                    token for tag in item.tags for token in _tokens(tag)
                )
                score = _overlap(query_tokens, candidate) * item.strength
                if score > 0:
                    ranked.append((score, item.id))
            ranked.sort(key=lambda pair: (-pair[0], pair[1]))
            results: list[EpisodicTrace] = []
            for _score, item_id in ranked[:limit]:
                item = self._items[item_id]
                strengthened = replace(
                    item,
                    strength=min(1.0, item.strength + self.retrieval_rate * (1.0 - item.strength)),
                    access_count=item.access_count + 1,
                )
                self._items[item_id] = strengthened
                results.append(strengthened)
            self._retrievals += len(results)
            return tuple(results)

    def forget_source(self, source_id: str) -> bool:
        with self._lock:
            selected = [key for key, item in self._items.items() if item.source_id == source_id]
            for key in selected:
                del self._items[key]
            return bool(selected)

    def forget_consent(self, owner_id: str, consent_id: str) -> int:
        with self._lock:
            selected = [
                key for key, item in self._items.items()
                if item.owner_id == owner_id and item.consent_id == consent_id
            ]
            for key in selected:
                del self._items[key]
            return len(selected)

    def forget_owner(self, owner_id: str) -> int:
        with self._lock:
            selected = [key for key, item in self._items.items() if item.owner_id == owner_id]
            for key in selected:
                del self._items[key]
            return len(selected)

    def reset(self) -> None:
        with self._lock:
            self._items.clear()
            self._evictions = 0
            self._retrievals = 0

    def snapshot(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "kind": "episodic",
                "count": len(self._items),
                "capacity": self.capacity,
                "retrievals": self._retrievals,
                "evictions": self._evictions,
                "state_storage": "volatile_process_memory",
            }


class SemanticMemoryStore:
    """Concept/attribute knowledge with evidence provenance and retrieval strengthening."""

    def __init__(self, *, capacity: int = 5_000, retrieval_rate: float = 0.08) -> None:
        _validate_capacity(capacity, "semantic capacity")
        if not math.isfinite(retrieval_rate) or not 0.0 < retrieval_rate <= 1.0:
            raise ValueError("retrieval_rate must be in (0, 1]")
        self.capacity = capacity
        self.retrieval_rate = retrieval_rate
        self._facts: dict[str, SemanticFact] = {}
        self._associations: dict[tuple[str, str, str], SemanticAssociation] = {}
        self._lock = RLock()
        self._evictions = 0

    def learn(
        self,
        *,
        owner_id: str,
        consent_id: str,
        source_id: str,
        concept: str,
        attributes: dict[str, str] | None = None,
        confidence: float = 0.5,
    ) -> SemanticFact:
        if not all(value.strip() for value in (owner_id, consent_id, source_id, concept)):
            raise ValueError("semantic fact requires owner, consent, source and concept")
        if isinstance(confidence, bool) or not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("semantic confidence must be in [0, 1]")
        attrs = tuple(sorted(
            (str(key).strip(), str(value).strip())
            for key, value in (attributes or {}).items()
            if str(key).strip() and str(value).strip()
        ))
        identity = (owner_id, consent_id, " ".join(concept.casefold().split()))
        with self._lock:
            existing = next(
                (
                    item for item in self._facts.values()
                    if (item.owner_id, item.consent_id, " ".join(item.concept.casefold().split())) == identity
                ),
                None,
            )
            if existing is not None:
                merged = dict(existing.attributes)
                merged.update(dict(attrs))
                updated = replace(
                    existing,
                    attributes=tuple(sorted(merged.items())),
                    source_ids=tuple(dict.fromkeys((*existing.source_ids, source_id))),
                    confidence=min(1.0, existing.confidence + 0.1 * (confidence - existing.confidence)),
                )
                self._facts[existing.id] = updated
                return updated
            if len(self._facts) >= self.capacity:
                weakest = min(
                    self._facts,
                    key=lambda key: (
                        self._facts[key].confidence * self._facts[key].strength,
                        self._facts[key].access_count,
                        key,
                    ),
                )
                del self._facts[weakest]
                self._evictions += 1
                self._associations = {
                    key: edge for key, edge in self._associations.items()
                    if weakest not in (edge.source_id, edge.target_id)
                }
            fact = SemanticFact(
                id=str(uuid4()), owner_id=owner_id, consent_id=consent_id,
                concept=concept.strip(), attributes=attrs, source_ids=(source_id,),
                confidence=confidence, strength=0.5,
            )
            self._facts[fact.id] = fact
            return fact

    def retrieve(self, owner_id: str, query: str, *, limit: int = 5) -> tuple[SemanticFact, ...]:
        if not 1 <= limit <= 100:
            raise ValueError("semantic retrieval limit must be in [1, 100]")
        query_tokens = _tokens(query)
        if not query_tokens:
            return ()
        with self._lock:
            ranked: list[tuple[float, str]] = []
            for item in self._facts.values():
                if item.owner_id != owner_id:
                    continue
                candidate_text = item.concept + " " + " ".join(
                    f"{key} {value}" for key, value in item.attributes
                )
                score = _overlap(query_tokens, _tokens(candidate_text)) * item.confidence * item.strength
                if score > 0:
                    ranked.append((score, item.id))
            ranked.sort(key=lambda pair: (-pair[0], pair[1]))
            results: list[SemanticFact] = []
            for _score, item_id in ranked[:limit]:
                item = self._facts[item_id]
                updated = replace(
                    item,
                    strength=min(1.0, item.strength + self.retrieval_rate * (1.0 - item.strength)),
                    access_count=item.access_count + 1,
                )
                self._facts[item_id] = updated
                results.append(updated)
            return tuple(results)

    def associate(self, owner_id: str, source_id: str, target_id: str, relation: str) -> SemanticAssociation:
        if not relation.strip() or source_id == target_id:
            raise ValueError("semantic association requires distinct concepts and a relation")
        with self._lock:
            source = self._facts.get(source_id)
            target = self._facts.get(target_id)
            if source is None or target is None or source.owner_id != owner_id or target.owner_id != owner_id:
                raise KeyError("semantic association endpoints must be owner-visible facts")
            if source.consent_id != target.consent_id:
                raise ValueError("semantic association cannot cross consent grants")
            key = (owner_id, source_id, target_id)
            current = self._associations.get(key)
            updated = SemanticAssociation(
                owner_id, source_id, target_id, relation.strip(),
                min(1.0, (current.strength if current else 0.0) + 0.15 * (1.0 - (current.strength if current else 0.0))),
            )
            if current is None and len(self._associations) >= self.capacity * 4:
                weakest = min(self._associations, key=lambda item: (self._associations[item].strength, item))
                del self._associations[weakest]
            self._associations[key] = updated
            return updated

    def related(self, owner_id: str, fact_id: str, *, limit: int = 10) -> tuple[tuple[SemanticFact, SemanticAssociation], ...]:
        with self._lock:
            edges = [
                edge for edge in self._associations.values()
                if edge.owner_id == owner_id and edge.source_id == fact_id
            ]
            edges.sort(key=lambda edge: (-edge.strength, edge.target_id))
            return tuple(
                (self._facts[edge.target_id], edge)
                for edge in edges[:limit] if edge.target_id in self._facts
            )

    def forget_source(self, source_id: str) -> int:
        with self._lock:
            selected = [key for key, item in self._facts.items() if source_id in item.source_ids]
            for key in selected:
                del self._facts[key]
                self._associations = {
                    edge_key: edge for edge_key, edge in self._associations.items()
                    if key not in (edge.source_id, edge.target_id)
                }
            return len(selected)

    def forget_consent(self, owner_id: str, consent_id: str) -> int:
        with self._lock:
            selected = [
                key for key, item in self._facts.items()
                if item.owner_id == owner_id and item.consent_id == consent_id
            ]
            for key in selected:
                del self._facts[key]
            self._associations = {
                key: edge for key, edge in self._associations.items()
                if key[0] != owner_id or (edge.source_id in self._facts and edge.target_id in self._facts)
            }
            return len(selected)

    def forget_owner(self, owner_id: str) -> int:
        with self._lock:
            selected = [key for key, item in self._facts.items() if item.owner_id == owner_id]
            for key in selected:
                del self._facts[key]
            self._associations = {
                key: edge for key, edge in self._associations.items()
                if edge.owner_id != owner_id
            }
            return len(selected)

    def reset(self) -> None:
        with self._lock:
            self._facts.clear()
            self._associations.clear()
            self._evictions = 0

    def snapshot(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "kind": "semantic",
                "count": len(self._facts),
                "associations": len(self._associations),
                "capacity": self.capacity,
                "evictions": self._evictions,
                "state_storage": "volatile_process_memory",
            }


class ProceduralMemoryStore:
    """Incremental task/skill memory with outcome-based confidence updates."""

    def __init__(self, *, capacity: int = 1_000) -> None:
        _validate_capacity(capacity, "procedural capacity")
        self.capacity = capacity
        self._items: dict[str, Procedure] = {}
        self._lock = RLock()
        self._evictions = 0

    def learn(
        self,
        *,
        owner_id: str,
        consent_id: str,
        source_id: str,
        task: str,
        steps: Sequence[str],
    ) -> Procedure:
        clean_steps = tuple(item.strip() for item in steps if item.strip())
        if not all(value.strip() for value in (owner_id, consent_id, source_id, task)) or not clean_steps:
            raise ValueError("procedure requires owner, consent, source, task and at least one step")
        identity = (owner_id, consent_id, " ".join(task.casefold().split()))
        with self._lock:
            existing = next(
                (
                    item for item in self._items.values()
                    if (item.owner_id, item.consent_id, " ".join(item.task.casefold().split())) == identity
                ),
                None,
            )
            if existing is not None:
                # Conflicting demonstrations remain separate evidence in the source list;
                # the current approved sequence is retained rather than silently overwritten.
                updated = replace(
                    existing,
                    source_ids=tuple(dict.fromkeys((*existing.source_ids, source_id))),
                )
                self._items[existing.id] = updated
                return updated
            if len(self._items) >= self.capacity:
                weakest = min(
                    self._items,
                    key=lambda key: (
                        self._items[key].strength * self._items[key].success_rate,
                        self._items[key].access_count,
                        key,
                    ),
                )
                del self._items[weakest]
                self._evictions += 1
            item = Procedure(
                id=str(uuid4()), owner_id=owner_id, consent_id=consent_id,
                task=task.strip(), steps=clean_steps, source_ids=(source_id,),
            )
            self._items[item.id] = item
            return item

    def record_outcome(self, owner_id: str, procedure_id: str, *, success: bool) -> Procedure:
        if not isinstance(success, bool):
            raise ValueError("success must be a boolean outcome")
        with self._lock:
            item = self._items.get(procedure_id)
            if item is None or item.owner_id != owner_id:
                raise KeyError(procedure_id)
            successes = item.successes + int(success)
            failures = item.failures + int(not success)
            observed = 1.0 if success else 0.0
            updated = replace(
                item,
                successes=successes,
                failures=failures,
                strength=min(1.0, max(0.0, item.strength + 0.1 * (observed - item.strength))),
            )
            self._items[procedure_id] = updated
            return updated

    def retrieve(self, owner_id: str, task: str, *, limit: int = 5) -> tuple[Procedure, ...]:
        if not 1 <= limit <= 100:
            raise ValueError("procedural retrieval limit must be in [1, 100]")
        query_tokens = _tokens(task)
        if not query_tokens:
            return ()
        with self._lock:
            ranked: list[tuple[float, str]] = []
            for item in self._items.values():
                if item.owner_id != owner_id:
                    continue
                score = _overlap(query_tokens, _tokens(item.task)) * item.strength * item.success_rate
                if score > 0:
                    ranked.append((score, item.id))
            ranked.sort(key=lambda pair: (-pair[0], pair[1]))
            results: list[Procedure] = []
            for _score, item_id in ranked[:limit]:
                item = self._items[item_id]
                updated = replace(item, access_count=item.access_count + 1)
                self._items[item_id] = updated
                results.append(updated)
            return tuple(results)

    def forget_source(self, source_id: str) -> int:
        with self._lock:
            selected = [key for key, item in self._items.items() if source_id in item.source_ids]
            for key in selected:
                del self._items[key]
            return len(selected)

    def forget_consent(self, owner_id: str, consent_id: str) -> int:
        with self._lock:
            selected = [
                key for key, item in self._items.items()
                if item.owner_id == owner_id and item.consent_id == consent_id
            ]
            for key in selected:
                del self._items[key]
            return len(selected)

    def forget_owner(self, owner_id: str) -> int:
        with self._lock:
            selected = [key for key, item in self._items.items() if item.owner_id == owner_id]
            for key in selected:
                del self._items[key]
            return len(selected)

    def reset(self) -> None:
        with self._lock:
            self._items.clear()
            self._evictions = 0

    def snapshot(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "kind": "procedural",
                "count": len(self._items),
                "capacity": self.capacity,
                "evictions": self._evictions,
                "state_storage": "volatile_process_memory",
            }


class SpatialMemoryStore:
    """Coordinate-indexed place memory with nearest-neighbour recall and explicit relations."""

    def __init__(self, *, capacity: int = 2_000) -> None:
        _validate_capacity(capacity, "spatial capacity")
        self.capacity = capacity
        self._places: dict[str, SpatialPlace] = {}
        self._relations: dict[tuple[str, str, str], SpatialRelation] = {}
        self._lock = RLock()
        self._evictions = 0

    def observe(
        self,
        *,
        owner_id: str,
        consent_id: str,
        source_id: str,
        label: str,
        coordinates: tuple[float, float, float],
        context_id: str = "default",
        tags: Sequence[str] = (),
    ) -> SpatialPlace:
        if not all(value.strip() for value in (owner_id, consent_id, source_id, label, context_id)):
            raise ValueError("spatial place requires owner, consent, source, label and context")
        if len(coordinates) != 3 or any(
            isinstance(value, bool) or not math.isfinite(value) or abs(value) > 1_000_000
            for value in coordinates
        ):
            raise ValueError("spatial coordinates must contain three finite values within +/-1,000,000")
        clean_tags = tuple(sorted({item.strip().casefold() for item in tags if item.strip()}))
        with self._lock:
            existing = next(
                (
                    item for item in self._places.values()
                    if item.owner_id == owner_id and item.consent_id == consent_id
                    and item.context_id == context_id and item.label.casefold() == label.casefold()
                ),
                None,
            )
            if existing is not None:
                updated = replace(
                    existing, coordinates=coordinates, source_id=source_id, tags=clean_tags
                )
                self._places[existing.id] = updated
                return updated
            if len(self._places) >= self.capacity:
                weakest = min(
                    self._places,
                    key=lambda key: (self._places[key].access_count, key),
                )
                del self._places[weakest]
                self._relations = {
                    edge_key: edge for edge_key, edge in self._relations.items()
                    if weakest not in (edge.source_id, edge.target_id)
                }
                self._evictions += 1
            item = SpatialPlace(
                id=str(uuid4()), owner_id=owner_id, consent_id=consent_id,
                source_id=source_id, label=label.strip(), coordinates=coordinates,
                context_id=context_id, tags=clean_tags,
            )
            self._places[item.id] = item
            return item

    def nearest(
        self,
        owner_id: str,
        coordinates: tuple[float, float, float],
        *,
        limit: int = 5,
        context_id: str | None = None,
    ) -> tuple[tuple[SpatialPlace, float], ...]:
        if len(coordinates) != 3 or any(not math.isfinite(value) for value in coordinates):
            raise ValueError("query coordinates must contain three finite values")
        if not 1 <= limit <= 100:
            raise ValueError("spatial retrieval limit must be in [1, 100]")
        with self._lock:
            candidates = [
                item for item in self._places.values()
                if item.owner_id == owner_id and (context_id is None or item.context_id == context_id)
            ]
            candidates.sort(
                key=lambda item: (
                    math.dist(item.coordinates, coordinates), item.id
                )
            )
            results: list[tuple[SpatialPlace, float]] = []
            for item in candidates[:limit]:
                updated = replace(item, access_count=item.access_count + 1)
                self._places[item.id] = updated
                results.append((updated, math.dist(updated.coordinates, coordinates)))
            return tuple(results)

    def connect(self, owner_id: str, source_id: str, target_id: str, relation: str) -> SpatialRelation:
        if not relation.strip() or source_id == target_id:
            raise ValueError("spatial relation requires distinct places and a relation")
        with self._lock:
            source = self._places.get(source_id)
            target = self._places.get(target_id)
            if source is None or target is None or source.owner_id != owner_id or target.owner_id != owner_id:
                raise KeyError("spatial relation endpoints must be owner-visible places")
            if source.consent_id != target.consent_id:
                raise ValueError("spatial relations cannot cross consent grants")
            edge = SpatialRelation(owner_id, source_id, target_id, relation.strip())
            key = (owner_id, source_id, target_id)
            if key not in self._relations and len(self._relations) >= self.capacity * 4:
                weakest = min(self._relations, key=lambda item: (self._relations[item].strength, item))
                del self._relations[weakest]
            self._relations[key] = edge
            return edge

    def retrieve(self, owner_id: str, query: str, *, limit: int = 5) -> tuple[SpatialPlace, ...]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return ()
        with self._lock:
            ranked = [
                (_overlap(query_tokens, _tokens(item.label + " " + " ".join(item.tags))), item.id)
                for item in self._places.values() if item.owner_id == owner_id
            ]
            ranked = [item for item in ranked if item[0] > 0]
            ranked.sort(key=lambda item: (-item[0], item[1]))
            results: list[SpatialPlace] = []
            for _score, item_id in ranked[:limit]:
                item = self._places[item_id]
                updated = replace(item, access_count=item.access_count + 1)
                self._places[item_id] = updated
                results.append(updated)
            return tuple(results)

    def forget_source(self, source_id: str) -> int:
        with self._lock:
            selected = [key for key, item in self._places.items() if item.source_id == source_id]
            for key in selected:
                del self._places[key]
            self._relations = {
                key: edge for key, edge in self._relations.items()
                if edge.source_id in self._places and edge.target_id in self._places
            }
            return len(selected)

    def forget_consent(self, owner_id: str, consent_id: str) -> int:
        with self._lock:
            selected = [
                key for key, item in self._places.items()
                if item.owner_id == owner_id and item.consent_id == consent_id
            ]
            for key in selected:
                del self._places[key]
            self._relations = {
                key: edge for key, edge in self._relations.items()
                if edge.owner_id != owner_id or (edge.source_id in self._places and edge.target_id in self._places)
            }
            return len(selected)

    def forget_owner(self, owner_id: str) -> int:
        with self._lock:
            selected = [key for key, item in self._places.items() if item.owner_id == owner_id]
            for key in selected:
                del self._places[key]
            self._relations = {
                key: edge for key, edge in self._relations.items()
                if edge.owner_id != owner_id
            }
            return len(selected)

    def reset(self) -> None:
        with self._lock:
            self._places.clear()
            self._relations.clear()
            self._evictions = 0

    def snapshot(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "kind": "spatial",
                "count": len(self._places),
                "relations": len(self._relations),
                "capacity": self.capacity,
                "evictions": self._evictions,
                "state_storage": "volatile_process_memory",
            }


class InMemoryCognitiveMemory:
    """Owns distinct volatile memory systems and their lifecycle."""

    def __init__(
        self,
        *,
        working_capacity: int = 7,
        max_working_sessions: int = 128,
        episodic_capacity: int = 2_000,
        semantic_capacity: int = 5_000,
        procedural_capacity: int = 1_000,
        spatial_capacity: int = 2_000,
    ) -> None:
        self.working = WorkingMemorySessions(
            capacity=working_capacity, max_sessions=max_working_sessions
        )
        self.episodic = EpisodicMemoryStore(capacity=episodic_capacity)
        self.semantic = SemanticMemoryStore(capacity=semantic_capacity)
        self.procedural = ProceduralMemoryStore(capacity=procedural_capacity)
        self.spatial = SpatialMemoryStore(capacity=spatial_capacity)
        self.relational = RelationalMemoryNetwork()
        self.episodic_replay = EpisodicReplayMemory()

    def ingest(self, record: MemoryRecord) -> str | None:
        """Project a consent-scoped source record into its distinct cognitive store."""
        common = {
            "owner_id": record.owner_id,
            "consent_id": record.consent_id,
            "source_id": record.id,
        }
        if record.kind is MemoryKind.EPISODIC:
            self.episodic.store(
                **common,
                context_id=str(record.metadata.get("context_id", record.owner_id)),
                content=record.content,
                occurred_at=int(record.metadata.get("tick", 0)),
                tags=tuple(str(item) for item in record.metadata.get("tags", ()) if str(item)),
                salience=float(record.metadata.get("salience", 0.5)),
            )
            return "episodic"
        if record.kind in {MemoryKind.SEMANTIC, MemoryKind.PREFERENCE, MemoryKind.DECISION}:
            attributes = record.metadata.get("attributes", {})
            if not isinstance(attributes, dict):
                attributes = {}
            self.semantic.learn(
                **common,
                concept=str(record.metadata.get("concept", record.content)),
                attributes={str(key): str(value) for key, value in attributes.items()},
                confidence=float(record.metadata.get("confidence", 0.5)),
            )
            return "semantic"
        if record.kind is MemoryKind.PROCEDURAL:
            raw_steps = record.metadata.get("steps", (record.content,))
            if not isinstance(raw_steps, (list, tuple)):
                raw_steps = (record.content,)
            self.procedural.learn(
                **common,
                task=str(record.metadata.get("task", record.content)),
                steps=tuple(str(item) for item in raw_steps),
            )
            return "procedural"
        if record.kind is MemoryKind.SPATIAL:
            raw_coordinates = record.metadata.get("coordinates")
            if not isinstance(raw_coordinates, (list, tuple)) or len(raw_coordinates) != 3:
                raise ValueError("spatial memory metadata requires three coordinates")
            self.spatial.observe(
                **common,
                label=str(record.metadata.get("label", record.content)),
                coordinates=tuple(float(value) for value in raw_coordinates),
                context_id=str(record.metadata.get("context_id", "default")),
                tags=tuple(str(item) for item in record.metadata.get("tags", ()) if str(item)),
            )
            return "spatial"
        return None

    def recall(self, owner_id: str, query: str, *, context_id: str | None = None) -> CognitiveMemoryRecall:
        return CognitiveMemoryRecall(
            episodic=self.episodic.retrieve(owner_id, query, context_id=context_id),
            semantic=self.semantic.retrieve(owner_id, query),
            procedural=self.procedural.retrieve(owner_id, query),
            spatial=self.spatial.retrieve(owner_id, query),
        )

    def forget_source(self, source_id: str) -> dict[str, int]:
        counts = {
            "episodic": int(self.episodic.forget_source(source_id)),
            "semantic": self.semantic.forget_source(source_id),
            "procedural": self.procedural.forget_source(source_id),
            "spatial": self.spatial.forget_source(source_id),
        }
        # The association/replay indexes retain identifiers without consent data.
        # Rebuild them from future co-activations rather than keep dangling links.
        self.relational.reset()
        self.episodic_replay.reset()
        return counts

    def forget_consent(self, owner_id: str, consent_id: str) -> dict[str, int]:
        counts = {
            "episodic": self.episodic.forget_consent(owner_id, consent_id),
            "semantic": self.semantic.forget_consent(owner_id, consent_id),
            "procedural": self.procedural.forget_consent(owner_id, consent_id),
            "spatial": self.spatial.forget_consent(owner_id, consent_id),
        }
        self.working.clear_owner(owner_id)
        # Link stores intentionally lack consent metadata; reset them on revocation.
        self.relational.reset()
        self.episodic_replay.reset()
        return counts

    def forget_owner(self, owner_id: str) -> dict[str, int]:
        counts = {
            "episodic": self.episodic.forget_owner(owner_id),
            "semantic": self.semantic.forget_owner(owner_id),
            "procedural": self.procedural.forget_owner(owner_id),
            "spatial": self.spatial.forget_owner(owner_id),
        }
        self.working.clear_owner(owner_id)
        # These associative structures retain source IDs but not consent metadata;
        # clear them on owner erasure rather than risk retaining an orphaned link.
        self.relational.reset()
        self.episodic_replay.reset()
        return counts

    def reset(self) -> None:
        self.working.reset()
        self.episodic.reset()
        self.semantic.reset()
        self.procedural.reset()
        self.spatial.reset()
        self.relational.reset()
        self.episodic_replay.reset()

    def snapshot(self) -> dict[str, object]:
        return {
            "working": self.working.snapshot(),
            "episodic": self.episodic.snapshot(),
            "semantic": self.semantic.snapshot(),
            "procedural": self.procedural.snapshot(),
            "spatial": self.spatial.snapshot(),
            "relational": self.relational.snapshot(),
            "episodic_replay": self.episodic_replay.snapshot(),
            "storage": "bounded_process_memory",
            "restart_policy": "runtime_experience_and_learned_state_reset",
        }
