"""Unified hybrid (lexical + semantic + temporal) long-term memory search.

Ranking strategy: weighted Reciprocal Rank Fusion (RRF; Cormack, Clarke &
Büttcher, 2009). BM25 scores and cosine similarities live on different scales,
so they are never added directly; each channel contributes ``w / (k + rank)``.

Channels:

* ``lexical`` — BM25 over content with stop-word removal and light suffix
  stemming.
* ``semantic`` — cosine similarity between unit embeddings; candidates below a
  minimum similarity (tuned on a held-out tuning split) are not retrieved,
  which lets irrelevant queries return nothing.
* ``temporal`` — only when the query carries temporal intent. It re-ranks items
  already found by a content channel and never retrieves on its own.

Hard filters (owner, consent, memory type, time, source, metadata) are applied
before ranking. Results are deterministic for the same data, configuration and
``reference_time``.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from artificial_human.core.models import MemoryKind
from artificial_human.memory.index import SemanticIndex
from artificial_human.memory.repository import MemoryLink, MemoryRecord
from artificial_human.memory.contracts import MemoryStore
from artificial_human.memory.temporal import (
    TemporalIntent,
    TemporalQuery,
    TemporalStatus,
    TimeRange,
    interpret_temporal_query,
    parse_stored_timestamp,
    parse_timestamp,
)

_WORD = re.compile(r"[a-z0-9_]+")
STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers herself him himself his how
    i if in into is it its itself just me more most my myself no nor not now of off on once
    only or other our ours ourselves out over own same she should so some such than that the
    their theirs them themselves then there these they this those through to too under until
    up very was we were what when where which while who whom why will with would you your
    yours yourself yourselves user owner owners users""".split()
)


def _stem(token: str) -> str:
    """Conservative suffix stripping so 'boils'/'boiling'/'boiled' share a stem."""
    for suffix in ("ingly", "edly", "ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stem = token[: -len(suffix)]
            if suffix in ("ing", "ed") and len(stem) > 3 and stem[-1] == stem[-2]:
                stem = stem[:-1]
            return stem
    return token


# Words that ask *about* time or events without naming a topic.
GENERIC_EVENT_TERMS = frozenset(
    _stem(word) for word in (
        "happen", "happened", "do", "done", "event", "events", "go", "went", "anything",
        "something", "things", "thing", "list", "tell", "show", "remember", "recall",
    )
)


def lexical_terms(text: str) -> list[str]:
    return [
        _stem(token) for token in _WORD.findall(text.lower())
        if len(token) >= 2 and token not in STOPWORDS
    ]


def bm25(
    query: str, documents: Sequence[tuple[str, str]], *, k1: float = 1.5, b: float = 0.75
) -> list[tuple[str, float, tuple[str, ...]]]:
    """Return ``(id, score, matched_terms)`` for documents with a positive score."""
    query_terms = sorted(set(lexical_terms(query)))
    if not query_terms or not documents:
        return []
    tokenised = [(doc_id, lexical_terms(text)) for doc_id, text in documents]
    total = len(tokenised)
    average = sum(len(terms) for _, terms in tokenised) / total
    term_sets = [set(terms) for _, terms in tokenised]
    frequency = {term: sum(term in terms for terms in term_sets) for term in query_terms}
    ranked: list[tuple[str, float, tuple[str, ...]]] = []
    for doc_id, terms in tokenised:
        counts = Counter(terms)
        score = 0.0
        matched: list[str] = []
        for term in query_terms:
            tf = counts[term]
            if not tf:
                continue
            idf = math.log(1 + (total - frequency[term] + 0.5) / (frequency[term] + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(terms) / max(average, 1)))
            matched.append(term)
        if score > 0:
            ranked.append((doc_id, score, tuple(matched)))
    # Stable sort: equal scores keep the caller's (deterministic) document order.
    ranked.sort(key=lambda item: -item[1])
    return ranked


@dataclass(frozen=True, slots=True)
class SearchFilters:
    kinds: tuple[MemoryKind, ...] = ()
    time_range: TimeRange | None = None
    valid_at: datetime | None = None
    known_at: datetime | None = None
    source_prefix: str | None = None
    fact_key: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    include_superseded: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "kinds": [kind.value for kind in self.kinds],
            "time_range": self.time_range.as_dict() if self.time_range else None,
            "valid_at": self.valid_at.isoformat() if self.valid_at else None,
            "known_at": self.known_at.isoformat() if self.known_at else None,
            "source_prefix": self.source_prefix,
            "fact_key": self.fact_key,
            "metadata": dict(self.metadata),
            "include_superseded": self.include_superseded,
        }


@dataclass(frozen=True, slots=True)
class SearchConfig:
    use_lexical: bool = True
    use_semantic: bool = True
    use_temporal_intent: bool = True
    rrf_k: int = 60
    lexical_weight: float = 1.0
    semantic_weight: float = 1.0
    temporal_weight: float = 0.5
    # Selected on the benchmark tuning split for BAAI/bge-small-en-v1.5 (see docs/MEMORY.md).
    # Other embedding models should be re-tuned with `artificial-human memory-eval`.
    semantic_min_similarity: float = 0.57
    channel_depth: int = 50
    expand_contradictions: bool = True

    @classmethod
    def from_env(cls) -> SearchConfig:
        """Read ranking overrides; defaults are documented in docs/MEMORY.md."""
        raw = os.getenv("ARTIFICIAL_HUMAN_MEMORY_MIN_SIMILARITY", "").strip()
        return cls(semantic_min_similarity=float(raw)) if raw else cls()

    def __post_init__(self) -> None:
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        if not -1.0 <= self.semantic_min_similarity <= 1.0:
            raise ValueError("semantic_min_similarity must be between -1 and 1")
        if self.channel_depth < 1:
            raise ValueError("channel_depth must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "use_lexical": self.use_lexical, "use_semantic": self.use_semantic,
            "use_temporal_intent": self.use_temporal_intent, "rrf_k": self.rrf_k,
            "weights": {
                "lexical": self.lexical_weight, "semantic": self.semantic_weight,
                "temporal": self.temporal_weight,
            },
            "semantic_min_similarity": self.semantic_min_similarity,
            "channel_depth": self.channel_depth,
            "expand_contradictions": self.expand_contradictions,
        }


@dataclass(frozen=True, slots=True)
class TemporalProfile:
    """Derived temporal facts about one record at a reference time."""

    status: TemporalStatus
    effective_time: datetime
    validity_start: datetime | None
    validity_end: datetime | None
    validity_known: bool
    superseded_by: str | None
    supersedes: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "effective_time": self.effective_time.isoformat(),
            "validity_start": self.validity_start.isoformat() if self.validity_start else None,
            "validity_end": self.validity_end.isoformat() if self.validity_end else None,
            "validity_known": self.validity_known,
            "superseded_by": self.superseded_by,
            "supersedes": self.supersedes,
        }


@dataclass(frozen=True, slots=True)
class MemorySearchResult:
    memory: MemoryRecord
    score: float
    rank: int
    channels: tuple[str, ...]
    temporal: TemporalProfile
    explanation: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MemorySearchResponse:
    query: str
    results: tuple[MemorySearchResult, ...]
    strategy: str
    temporal_query: TemporalQuery
    filters: SearchFilters
    config: SearchConfig
    reference_time: datetime
    semantic: dict[str, Any]
    candidate_count: int

    def explain(self) -> dict[str, Any]:
        """Identifiers, provenance and ranking evidence; excludes memory content."""
        return {
            "strategy": self.strategy,
            "reference_time": self.reference_time.isoformat(),
            "temporal_query": self.temporal_query.as_dict(),
            "filters": self.filters.as_dict(),
            "semantic": self.semantic,
            "candidate_count": self.candidate_count,
            "results": [
                {"memory_id": item.memory.id, "rank": item.rank, **item.explanation}
                for item in self.results
            ],
        }


class _LinkGraph:
    def __init__(self, links: Sequence[MemoryLink], known_at: datetime | None) -> None:
        self.superseded_by: dict[str, MemoryLink] = {}
        self.supersedes: dict[str, MemoryLink] = {}
        self.contradicts: dict[str, set[str]] = {}
        for link in links:
            if known_at is not None and parse_stored_timestamp(link.created_at) > known_at:
                continue
            if link.relation == "supersedes":
                self.superseded_by[link.to_memory_id] = link
                self.supersedes[link.from_memory_id] = link
            else:
                self.contradicts.setdefault(link.from_memory_id, set()).add(link.to_memory_id)
                self.contradicts.setdefault(link.to_memory_id, set()).add(link.from_memory_id)


def _optional(value: str | None) -> datetime | None:
    return parse_stored_timestamp(value) if value else None


def temporal_profile(
    record: MemoryRecord, graph: _LinkGraph, at: datetime, known_at: datetime | None = None
) -> TemporalProfile:
    occurred = _optional(record.occurred_at)
    valid_from = _optional(record.valid_from)
    valid_to = _optional(record.valid_to)
    recorded = parse_stored_timestamp(record.created_at)
    older_link = graph.supersedes.get(record.id)
    newer_link = graph.superseded_by.get(record.id)
    implicit_start = (
        _optional(older_link.effective_at) if older_link and older_link.effective_at else None
    )
    # A validity interval exists only for facts. An event's occurred_at is a point
    # in time, not the start of an open-ended interval.
    start = valid_from or implicit_start
    superseded_at = (
        _optional(newer_link.effective_at) if newer_link and newer_link.effective_at else None
    )
    end_candidates = [moment for moment in (valid_to, superseded_at) if moment is not None]
    end = min(end_candidates) if end_candidates else None

    if known_at is not None and recorded > known_at:
        status = TemporalStatus.NOT_YET_RECORDED
    elif superseded_at is not None and superseded_at <= at:
        status = TemporalStatus.SUPERSEDED
    elif valid_to is not None and valid_to <= at:
        status = TemporalStatus.EXPIRED
    elif (begins := start or occurred) is not None and begins > at:
        status = TemporalStatus.NOT_YET_VALID
    else:
        status = TemporalStatus.VALID
    return TemporalProfile(
        status=status,
        effective_time=occurred or valid_from or implicit_start or recorded,
        validity_start=start,
        validity_end=end,
        validity_known=any(
            item is not None for item in (valid_from, valid_to, occurred, older_link, newer_link)
        ),
        superseded_by=newer_link.from_memory_id if newer_link else None,
        supersedes=older_link.to_memory_id if older_link else None,
    )


def strip_cues(text: str, cues: Sequence[str]) -> str:
    """Remove temporal cue phrases so they do not act as topical keywords."""
    stripped = text.lower()
    for cue in sorted(cues, key=len, reverse=True):
        stripped = re.sub(r"\b" + re.escape(cue.lower()) + r"\b", " ", stripped)
    return stripped


def _in_range(profile: TemporalProfile, window: TimeRange) -> bool:
    if profile.validity_start is None and profile.validity_end is None:
        return window.contains(profile.effective_time)
    return window.overlaps(profile.validity_start, profile.validity_end)


class HybridMemorySearch:
    """Owner-scoped, consent-filtered, explainable long-term memory search."""

    def __init__(
        self,
        repository: MemoryStore,
        index: SemanticIndex | None = None,
        config: SearchConfig | None = None,
    ) -> None:
        self.repository = repository
        self.index = index
        self.config = config or SearchConfig()

    def _passes_filters(
        self, record: MemoryRecord, profile: TemporalProfile, filters: SearchFilters
    ) -> str | None:
        """Return the name of the first failing filter, or ``None``."""
        if filters.known_at is not None and profile.status is TemporalStatus.NOT_YET_RECORDED:
            return "known_at"
        if filters.valid_at is not None and profile.status is not TemporalStatus.VALID:
            return "valid_at"
        if filters.time_range is not None and not _in_range(profile, filters.time_range):
            return "time_range"
        if not filters.include_superseded and profile.superseded_by is not None:
            return "include_superseded"
        if filters.source_prefix and not record.source.startswith(filters.source_prefix):
            return "source_prefix"
        if filters.fact_key and record.fact_key != filters.fact_key:
            return "fact_key"
        for key, expected in filters.metadata.items():
            if record.metadata.get(key) != expected:
                return "metadata"
        return None

    @staticmethod
    def _temporal_exclusion(profile: TemporalProfile, intent: TemporalQuery) -> bool:
        if intent.intent is TemporalIntent.CURRENT:
            return profile.status is not TemporalStatus.VALID
        if intent.intent is TemporalIntent.RANGE and intent.time_range is not None:
            return not _in_range(profile, intent.time_range)
        if intent.intent is TemporalIntent.HISTORICAL:
            # The record that is currently valid *and* known to have replaced or
            # started after something else is the present answer, not history.
            return profile.status is TemporalStatus.VALID and profile.validity_known and (
                profile.supersedes is not None or profile.validity_start is not None
            ) and profile.superseded_by is None and profile.validity_end is None
        return False

    @staticmethod
    def _temporal_order(
        ids: Sequence[str], profiles: Mapping[str, TemporalProfile], intent: TemporalIntent
    ) -> list[str]:
        def end_or_effective(memory_id: str) -> float:
            profile = profiles[memory_id]
            moment = profile.validity_end or profile.effective_time
            return moment.timestamp()

        if intent is TemporalIntent.HISTORICAL:
            return sorted(
                ids,
                key=lambda item: (
                    profiles[item].status is TemporalStatus.VALID,
                    -end_or_effective(item),
                ),
            )
        if intent is TemporalIntent.CHRONOLOGICAL:
            return sorted(ids, key=lambda item: profiles[item].effective_time)
        if intent in (TemporalIntent.RECENT, TemporalIntent.CURRENT):
            return sorted(
                ids, key=lambda item: -profiles[item].effective_time.timestamp()
            )
        return []

    @staticmethod
    def _temporal_only_order(
        candidates: Sequence[MemoryRecord], profiles: Mapping[str, TemporalProfile],
        temporal_query: TemporalQuery, now: datetime,
    ) -> list[str]:
        """Rank by time alone for queries with no topical content ("What happened in May?").

        Events (episodic records) come first, then other records whose time
        point or validity change falls in the requested window.
        """
        def kind_rank(record: MemoryRecord) -> int:
            return 0 if record.kind is MemoryKind.EPISODIC else 1

        intent = temporal_query.intent
        if intent is TemporalIntent.RANGE and temporal_query.time_range is not None:
            window = temporal_query.time_range
            selected: list[tuple[int, datetime, int, str]] = []
            for position, record in enumerate(candidates):
                profile = profiles[record.id]
                moments = [
                    moment for moment in (
                        profile.effective_time, profile.validity_start, profile.validity_end
                    )
                    if moment is not None and window.contains(moment)
                ]
                if moments:
                    selected.append((kind_rank(record), min(moments), position, record.id))
            return [memory_id for *_, memory_id in sorted(selected)]
        past = [
            record for record in candidates if profiles[record.id].effective_time <= now
        ]
        if intent is TemporalIntent.RECENT:
            past.sort(key=lambda record: (
                kind_rank(record), -profiles[record.id].effective_time.timestamp()
            ))
        else:
            past.sort(key=lambda record: (
                kind_rank(record), profiles[record.id].effective_time
            ))
        return [record.id for record in past]

    def search(
        self,
        owner_id: str,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
        config: SearchConfig | None = None,
        reference_time: datetime | None = None,
    ) -> MemorySearchResponse:
        if not query.strip():
            raise ValueError("A search query is required")
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        filters = filters or SearchFilters()
        config = config or self.config
        now = parse_timestamp(reference_time) if reference_time else datetime.now(UTC)
        status_time = filters.valid_at or now

        records = self.repository.iter_visible(owner_id, kinds=filters.kinds or None)
        graph = _LinkGraph(self.repository.links(owner_id), filters.known_at)
        profiles = {
            item.id: temporal_profile(item, graph, status_time, filters.known_at)
            for item in records
        }
        temporal_query = (
            interpret_temporal_query(query, reference_time=now)
            if config.use_temporal_intent
            else TemporalQuery()
        )
        if filters.valid_at is not None or filters.time_range is not None:
            # Explicit temporal filters take precedence over inferred intent.
            if temporal_query.intent in (TemporalIntent.CURRENT, TemporalIntent.RANGE):
                temporal_query = TemporalQuery(cues=temporal_query.cues)
        candidates = [
            item for item in records
            if self._passes_filters(item, profiles[item.id], filters) is None
            and not self._temporal_exclusion(profiles[item.id], temporal_query)
        ]
        by_id = {item.id: item for item in candidates}
        # Ties are broken by recorded time, then id (the candidate order), never by
        # random identifiers alone, so identical data yields identical rankings.
        order_of = {item.id: index for index, item in enumerate(candidates)}
        lexical_query = strip_cues(query, temporal_query.cues)
        temporal_only = temporal_query.intent in (
            TemporalIntent.RANGE, TemporalIntent.RECENT, TemporalIntent.CHRONOLOGICAL
        ) and not [
            term for term in lexical_terms(lexical_query) if term not in GENERIC_EVENT_TERMS
        ]

        channels: dict[str, list[str]] = {}
        evidence: dict[str, dict[str, dict[str, Any]]] = {}
        if config.use_lexical:
            ranked = bm25(
                lexical_query,
                [(item.id, f"{item.content} {item.kind.value}") for item in candidates],
            )[: config.channel_depth]
            channels["lexical"] = [doc_id for doc_id, _, _ in ranked]
            for position, (doc_id, score, matched) in enumerate(ranked, start=1):
                evidence.setdefault(doc_id, {})["lexical"] = {
                    "rank": position, "bm25": round(score, 4), "matched_terms": list(matched),
                }

        semantic_state: dict[str, Any] = {"requested": config.use_semantic, "used": False}
        semantic_configured = self.index is not None and self.index.configured
        if config.use_semantic and semantic_configured and self.index is not None and candidates:
            embedded = self.index.query_vector(owner_id, query)
            if embedded is None:
                semantic_state["reason"] = "embedding backend unavailable or not permitted"
            else:
                info, vector = embedded
                similarities = self.index.similarities(owner_id, info, vector, list(by_id))
                accepted = sorted(
                    (
                        (memory_id, value) for memory_id, value in similarities.items()
                        if value >= config.semantic_min_similarity
                    ),
                    key=lambda item: (-item[1], order_of[item[0]]),
                )[: config.channel_depth]
                channels["semantic"] = [memory_id for memory_id, _ in accepted]
                for position, (memory_id, value) in enumerate(accepted, start=1):
                    evidence.setdefault(memory_id, {})["semantic"] = {
                        "rank": position, "cosine_similarity": round(value, 4),
                    }
                semantic_state.update({
                    "used": True, "model": info.public(),
                    "indexed_candidates": len(similarities),
                    "unindexed_candidates": len(by_id) - len(similarities),
                    "min_similarity": config.semantic_min_similarity,
                })
        elif config.use_semantic and not semantic_configured:
            semantic_state["reason"] = "no embedding backend configured"
        elif config.use_semantic:
            semantic_state["reason"] = "no candidate memories"

        content_hits = sorted(
            {doc_id for ids in channels.values() for doc_id in ids}, key=order_of.__getitem__
        )
        if temporal_only:
            temporal_ids = self._temporal_only_order(
                candidates, profiles, temporal_query, now
            )[: config.channel_depth]
        else:
            temporal_ids = self._temporal_order(content_hits, profiles, temporal_query.intent)
        if temporal_ids:
            channels["temporal"] = temporal_ids
            for position, memory_id in enumerate(temporal_ids, start=1):
                evidence.setdefault(memory_id, {})["temporal"] = {
                    "rank": position, "intent": temporal_query.intent.value,
                    "temporal_only_query": temporal_only,
                }

        weights = {
            "lexical": config.lexical_weight, "semantic": config.semantic_weight,
            "temporal": config.temporal_weight,
        }
        fused: dict[str, float] = {}
        contributions: dict[str, dict[str, float]] = {}
        for channel, ids in channels.items():
            for position, memory_id in enumerate(ids, start=1):
                value = weights[channel] / (config.rrf_k + position)
                fused[memory_id] = fused.get(memory_id, 0.0) + value
                contributions.setdefault(memory_id, {})[channel] = round(value, 6)
        ordered = sorted(fused, key=lambda memory_id: (-fused[memory_id], order_of[memory_id]))

        conflicts = self._conflicts(candidates, profiles, graph)
        if config.expand_contradictions:
            ordered = self._expand(ordered, conflicts, fused, contributions, order_of)
        ordered = ordered[:top_k]

        results = []
        for rank, memory_id in enumerate(ordered, start=1):
            record = by_id[memory_id]
            used = tuple(
                channel for channel in ("lexical", "semantic", "temporal", "contradiction")
                if channel in contributions.get(memory_id, {})
            )
            results.append(
                MemorySearchResult(
                    memory=record, score=round(fused[memory_id], 6), rank=rank, channels=used,
                    temporal=profiles[memory_id],
                    explanation={
                        "kind": record.kind.value,
                        "source": record.source,
                        "consent_id": record.consent_id,
                        "epistemic_status": record.epistemic_status,
                        "fact_key": record.fact_key,
                        "channels": list(used),
                        "fusion": {
                            "method": "weighted_reciprocal_rank_fusion",
                            "k": config.rrf_k,
                            "score": round(fused[memory_id], 6),
                            "contributions": contributions.get(memory_id, {}),
                        },
                        "evidence": evidence.get(memory_id, {}),
                        "temporal": profiles[memory_id].as_dict(),
                        "conflicts_with": sorted(conflicts.get(memory_id, set())),
                    },
                )
            )
        strategy = "+".join(
            channel for channel in ("lexical", "semantic", "temporal") if channel in channels
        ) or "none"
        return MemorySearchResponse(
            query=query, results=tuple(results), strategy=strategy,
            temporal_query=temporal_query, filters=filters, config=config,
            reference_time=now, semantic=semantic_state, candidate_count=len(candidates),
        )

    @staticmethod
    def _conflicts(
        candidates: Sequence[MemoryRecord], profiles: Mapping[str, TemporalProfile],
        graph: _LinkGraph,
    ) -> dict[str, set[str]]:
        """Explicit contradictions plus same-fact records that are valid simultaneously."""
        ids = {item.id for item in candidates}
        conflicts = {
            memory_id: {other for other in others if other in ids}
            for memory_id, others in graph.contradicts.items() if memory_id in ids
        }
        groups: dict[str, list[str]] = {}
        for item in candidates:
            if item.fact_key and profiles[item.id].status is TemporalStatus.VALID:
                groups.setdefault(item.fact_key, []).append(item.id)
        for members in groups.values():
            if len(members) < 2:
                continue
            for memory_id in members:
                conflicts.setdefault(memory_id, set()).update(
                    other for other in members if other != memory_id
                )
        return {key: value for key, value in conflicts.items() if value}

    @staticmethod
    def _expand(
        ordered: list[str], conflicts: Mapping[str, set[str]], fused: dict[str, float],
        contributions: dict[str, dict[str, float]], order_of: Mapping[str, int],
    ) -> list[str]:
        """Place conflicting evidence beside each retrieved record instead of hiding it."""
        expanded: list[str] = []
        for memory_id in ordered:
            if memory_id not in expanded:
                expanded.append(memory_id)
            for other in sorted(conflicts.get(memory_id, set()), key=order_of.__getitem__):
                if other in expanded:
                    continue
                expanded.append(other)
                if other not in fused:
                    fused[other] = fused[memory_id]
                    contributions.setdefault(other, {})["contradiction"] = 0.0
        return expanded

    def timeline(
        self,
        owner_id: str,
        *,
        filters: SearchFilters | None = None,
        reference_time: datetime | None = None,
        limit: int = 100,
    ) -> list[tuple[MemoryRecord, TemporalProfile]]:
        """Chronological recall: records ordered by event/validity/recorded time."""
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        filters = filters or SearchFilters()
        now = parse_timestamp(reference_time) if reference_time else datetime.now(UTC)
        records = self.repository.iter_visible(owner_id, kinds=filters.kinds or None)
        graph = _LinkGraph(self.repository.links(owner_id), filters.known_at)
        status_time = filters.valid_at or now
        items = [
            (item, temporal_profile(item, graph, status_time, filters.known_at))
            for item in records
        ]
        kept = [
            (record, profile) for record, profile in items
            if self._passes_filters(record, profile, filters) is None
        ]
        kept.sort(key=lambda pair: pair[1].effective_time)  # stable: recorded order breaks ties
        return kept[:limit]

    def with_config(self, **changes: Any) -> HybridMemorySearch:
        return HybridMemorySearch(self.repository, self.index, replace(self.config, **changes))
