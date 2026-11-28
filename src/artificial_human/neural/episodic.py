"""Bounded neural engram experiment for episodic pattern separation and completion.

This deterministic LIF engineering model is not a hippocampal simulation. It
stores keyed feature digests and provenance only; event text remains in the
consent-governed in-memory episodic store.
"""
from __future__ import annotations

import hashlib
import hmac
import math
import re
import secrets
from collections.abc import Sequence
from dataclasses import dataclass, replace
from threading import RLock

from artificial_human.neural.simulation import (
    LIFNeuron,
    LIFParameters,
    NeuralSimulation,
    Polarity,
    Synapse,
)


@dataclass(frozen=True, slots=True)
class EpisodicEngram:
    """Opaque sparse event code with source and consent provenance."""

    owner_id: str
    consent_id: str
    source_id: str
    context_digest: str
    feature_ids: tuple[str, ...]
    sparse_code: tuple[str, ...]
    occurred_at: int
    salience: float
    strength: float = 0.5
    access_count: int = 0


@dataclass(frozen=True, slots=True)
class EpisodicNeuralMatch:
    source_id: str
    score: float
    symbolic_score: float
    neural_activation: float
    completed_feature_count: int
    expected_completion_count: int
    completion_ratio: float
    spike_count: int
    latency_steps: int | None


@dataclass(frozen=True, slots=True)
class EpisodicNeuralRecall:
    matches: tuple[EpisodicNeuralMatch, ...]
    candidate_count: int
    activated_count: int
    network_spikes: int
    interference_index: float
    pattern_separation_index: float
    symbolic_top_source_id: str | None
    neural_top_source_id: str | None
    simulation_steps: int

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_count": self.candidate_count,
            "activated_count": self.activated_count,
            "network_spikes": self.network_spikes,
            "interference_index": self.interference_index,
            "pattern_separation_index": self.pattern_separation_index,
            "symbolic_top_source_id": self.symbolic_top_source_id,
            "neural_top_source_id": self.neural_top_source_id,
            "simulation_steps": self.simulation_steps,
            "matches": [
                {
                    "source_id": item.source_id,
                    "score": item.score,
                    "symbolic_score": item.symbolic_score,
                    "neural_activation": item.neural_activation,
                    "completed_feature_count": item.completed_feature_count,
                    "expected_completion_count": item.expected_completion_count,
                    "completion_ratio": item.completion_ratio,
                    "spike_count": item.spike_count,
                    "latency_steps": item.latency_steps,
                }
                for item in self.matches
            ],
        }


class EpisodicNeuralCircuit:
    """RAM-only sparse engrams with LIF cue activation and recurrent completion.

    Each engram contains keyed token digests, a keyed context digest, two
    identity features and source/consent identifiers. The per-instance HMAC
    key is volatile and is never exported in snapshots.

    Recall compares a deterministic Dice-overlap baseline with a small LIF
    circuit: cue-feature neurons activate candidate engrams; active engrams
    excite their own missing-feature neurons; a shared inhibitory unit provides
    competition. The circuit does not use benchmark labels or an LLM.
    """

    def __init__(
        self,
        *,
        capacity: int = 2_000,
        max_features: int = 16,
        max_candidates: int = 12,
        secret: bytes | None = None,
    ) -> None:
        for name, value in (
            ("capacity", capacity),
            ("max_features", max_features),
            ("max_candidates", max_candidates),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_features > 64 or max_candidates > 12:
            raise ValueError("feature and candidate bounds exceed the supported circuit size")
        if secret is not None and (not isinstance(secret, bytes) or len(secret) < 16):
            raise ValueError("secret must contain at least 16 bytes")
        self.capacity = capacity
        self.max_features = max_features
        self.max_candidates = max_candidates
        self._secret = secret or secrets.token_bytes(32)
        self._items: dict[tuple[str, str], EpisodicEngram] = {}
        self._lock = RLock()
        self._evictions = 0
        self._encodings = 0
        self._successful_strengthenings = 0

    def _digest(self, kind: str, value: str) -> str:
        payload = f"{kind}:{value}".encode("utf-8")
        return hmac.new(self._secret, payload, hashlib.sha256).hexdigest()[:20]

    def _features(self, content: str, tags: Sequence[str] = ()) -> tuple[str, ...]:
        tokens = set(re.findall(r"[a-z0-9]+", content.casefold()))
        for tag in tags:
            tokens.update(re.findall(r"[a-z0-9]+", tag.casefold()))
        return tuple(
            self._digest("token", token)
            for token in sorted(tokens)[: self.max_features]
        )

    def encode(
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
    ) -> EpisodicEngram:
        """Encode an already consent-approved event without retaining its text."""
        if not all(
            isinstance(value, str) and value.strip()
            for value in (owner_id, consent_id, source_id, context_id, content)
        ):
            raise ValueError("episodic engram requires owner, consent, source, context and content")
        if any(
            len(value) > 256
            for value in (owner_id, consent_id, source_id, context_id)
        ):
            raise ValueError("episodic provenance identifiers exceed 256 characters")
        if isinstance(occurred_at, bool) or not isinstance(occurred_at, int) or occurred_at < 0:
            raise ValueError("occurred_at must be a non-negative integer")
        if (
            isinstance(salience, bool)
            or not math.isfinite(salience)
            or not 0.0 <= salience <= 1.0
        ):
            raise ValueError("salience must be finite and in [0, 1]")
        if len(tags) > 64 or any(
            not isinstance(tag, str) or len(tag) > 256 for tag in tags
        ):
            raise ValueError("tags exceed the supported bounds")

        feature_ids = self._features(content, tags)
        if not feature_ids:
            raise ValueError("episodic event must contain at least one text feature")
        context_digest = self._digest("context", f"{owner_id}:{context_id}")
        identity_features = {
            self._digest("identity", f"{owner_id}:{source_id}:{index}")
            for index in range(2)
        }
        sparse_code = tuple(
            sorted(set(feature_ids) | identity_features | {context_digest})
        )
        item = EpisodicEngram(
            owner_id=owner_id,
            consent_id=consent_id,
            source_id=source_id,
            context_digest=context_digest,
            feature_ids=feature_ids,
            sparse_code=sparse_code,
            occurred_at=occurred_at,
            salience=salience,
        )
        key = (owner_id, source_id)
        with self._lock:
            existing = self._items.get(key)
            if existing is not None and existing.consent_id != consent_id:
                raise ValueError("an engram source cannot be reassigned to another consent grant")
            if existing is None and len(self._items) >= self.capacity:
                weakest = min(
                    self._items,
                    key=lambda candidate: (
                        self._items[candidate].salience * self._items[candidate].strength,
                        self._items[candidate].access_count,
                        self._items[candidate].occurred_at,
                        candidate,
                    ),
                )
                del self._items[weakest]
                self._evictions += 1
            self._items[key] = item
            self._encodings += 1
            return item

    @staticmethod
    def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
        first, second = set(left), set(right)
        union = first | second
        return len(first & second) / len(union) if union else 0.0

    def _pattern_separation(self, items: Sequence[EpisodicEngram]) -> float:
        if len(items) < 2:
            return 0.0
        semantic: list[float] = []
        coded: list[float] = []
        for index, first in enumerate(items):
            for second in items[index + 1 :]:
                semantic.append(self._jaccard(first.feature_ids, second.feature_ids))
                coded.append(self._jaccard(first.sparse_code, second.sparse_code))
        difference = sum(semantic) / len(semantic) - sum(coded) / len(coded)
        return round(min(1.0, max(0.0, difference)), 6)

    def _simulate(
        self,
        query_features: tuple[str, ...],
        items: Sequence[EpisodicEngram],
    ) -> tuple[dict[str, tuple[int, int | None, int]], int, int]:
        """Run a bounded LIF cue-to-engram-to-completion experiment."""
        parameters = LIFParameters(
            resting_potential_mV=-65.0,
            reset_potential_mV=-65.0,
            threshold_mV=-40.0,
            membrane_time_constant_ms=4.0,
            membrane_resistance=1.0,
            refractory_period_ms=2.0,
        )
        feature_ids = sorted(
            set(query_features).union(*(set(item.feature_ids) for item in items))
        )
        completion_ids: dict[str, list[str]] = {}
        simulation = NeuralSimulation(
            dt_ms=1.0,
            max_neurons=512,
            max_synapses=1_024,
            max_pending_events=1_024,
            max_abs_external_current=800.0,
        )
        feature_neurons: dict[str, str] = {}
        for index, feature_id in enumerate(feature_ids):
            neuron_id = f"feature-{index}"
            feature_neurons[feature_id] = neuron_id
            simulation.add_neuron(LIFNeuron(neuron_id, parameters))
        candidate_neurons: dict[str, str] = {}
        for index, item in enumerate(items):
            neuron_id = f"engram-{index}"
            candidate_neurons[item.source_id] = neuron_id
            simulation.add_neuron(LIFNeuron(neuron_id, parameters))
            missing_features = [
                feature for feature in item.feature_ids if feature not in query_features
            ]
            targets: list[str] = []
            for feature_index, _feature_id in enumerate(missing_features):
                target_id = f"completion-{index}-{feature_index}"
                simulation.add_neuron(LIFNeuron(target_id, parameters))
                simulation.add_synapse(
                    Synapse(
                        identifier=f"complete-{index}-{feature_index}",
                        source=neuron_id,
                        target=target_id,
                        weight=120.0,
                        polarity=Polarity.EXCITATORY,
                        delay_steps=1,
                        time_constant_ms=4.0,
                        maximum_weight=200.0,
                    )
                )
                targets.append(target_id)
            completion_ids[item.source_id] = targets
            for feature_index, feature_id in enumerate(item.feature_ids):
                if feature_id not in query_features:
                    continue
                simulation.add_synapse(
                    Synapse(
                        identifier=f"cue-{index}-{feature_index}",
                        source=feature_neurons[feature_id],
                        target=neuron_id,
                        weight=40.0,
                        polarity=Polarity.EXCITATORY,
                        delay_steps=1,
                        time_constant_ms=4.0,
                        maximum_weight=100.0,
                    )
                )

        inhibitory_id = "shared-inhibitory"
        simulation.add_neuron(LIFNeuron(inhibitory_id, parameters))
        for index, item in enumerate(items):
            candidate_id = candidate_neurons[item.source_id]
            simulation.add_synapse(
                Synapse(
                    identifier=f"engram-to-inhibitory-{index}",
                    source=candidate_id,
                    target=inhibitory_id,
                    weight=120.0,
                    polarity=Polarity.EXCITATORY,
                    delay_steps=1,
                    time_constant_ms=4.0,
                    maximum_weight=200.0,
                )
            )
            for other_index, other in enumerate(items):
                if other_index == index:
                    continue
                simulation.add_synapse(
                    Synapse(
                        identifier=f"lateral-inhibition-{index}-{other_index}",
                        source=inhibitory_id,
                        target=candidate_neurons[other.source_id],
                        weight=120.0,
                        polarity=Polarity.INHIBITORY,
                        delay_steps=1,
                        time_constant_ms=4.0,
                        maximum_weight=200.0,
                    )
                )

        cue_currents = {feature_neurons[item]: 600.0 for item in query_features}
        total_spikes = 0
        for step_index in range(6):
            result = simulation.step(cue_currents if step_index == 0 else None)
            total_spikes += len(result.spikes)

        by_source: dict[str, tuple[int, int | None, int]] = {}
        for item in items:
            neuron = simulation.neurons[candidate_neurons[item.source_id]]
            completion_count = sum(
                simulation.neurons[target_id].spike_count > 0
                for target_id in completion_ids[item.source_id]
            )
            latency = (
                int(round(neuron.last_spike_time_ms / simulation.dt_ms))
                if neuron.last_spike_time_ms is not None
                else None
            )
            by_source[item.source_id] = (
                neuron.spike_count,
                latency,
                int(completion_count),
            )
        return by_source, total_spikes, simulation.step_index

    def recall(
        self,
        owner_id: str,
        query: str,
        *,
        context_id: str | None = None,
        limit: int = 5,
    ) -> EpisodicNeuralRecall:
        """Complete partial cues through bounded LIF engram activation."""
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id is required")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be non-empty")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer in [1, 100]")
        if context_id is not None and (
            not isinstance(context_id, str)
            or not context_id.strip()
            or len(context_id) > 256
        ):
            raise ValueError("context_id must be a non-empty string within 256 characters")
        query_features = self._features(query)
        if not query_features:
            return EpisodicNeuralRecall((), 0, 0, 0, 0.0, 0.0, None, None, 0)
        context_digest = (
            self._digest("context", f"{owner_id}:{context_id}")
            if context_id is not None
            else None
        )
        with self._lock:
            eligible = [
                item
                for item in self._items.values()
                if item.owner_id == owner_id
                and (context_digest is None or item.context_digest == context_digest)
            ]
            ranked: list[tuple[float, EpisodicEngram, int]] = []
            query_set = set(query_features)
            for item in eligible:
                overlap_count = len(query_set & set(item.feature_ids))
                if overlap_count == 0:
                    continue
                symbolic_score = (
                    2.0 * overlap_count / (len(query_set) + len(item.feature_ids))
                )
                ranked.append((symbolic_score * item.strength, item, overlap_count))
            ranked.sort(
                key=lambda candidate: (
                    -candidate[0],
                    -candidate[1].salience,
                    candidate[1].source_id,
                )
            )
            candidates = ranked[: self.max_candidates]
            candidate_items = [item for _score, item, _overlap in candidates]
            neural, network_spikes, simulation_steps = self._simulate(
                query_features, candidate_items
            )
            symbolic_scores = [score for score, _item, _overlap in candidates]
            interference = 0.0
            if len(symbolic_scores) > 1:
                gap = max(0.0, symbolic_scores[0] - symbolic_scores[1])
                interference = round(min(1.0, max(0.0, 1.0 - gap)), 6)
            matches: list[EpisodicNeuralMatch] = []
            for symbolic_score, item, _overlap in candidates:
                spike_count, latency, completion_count = neural[item.source_id]
                expected = max(
                    0,
                    len(item.feature_ids) - len(query_set & set(item.feature_ids)),
                )
                activation = 1.0 if spike_count else 0.0
                completion_ratio = completion_count / expected if expected else 0.0
                score = symbolic_score * (
                    0.55 + 0.30 * activation + 0.15 * completion_ratio
                )
                matches.append(
                    EpisodicNeuralMatch(
                        source_id=item.source_id,
                        score=round(min(1.0, max(0.0, score)), 6),
                        symbolic_score=round(symbolic_score, 6),
                        neural_activation=activation,
                        completed_feature_count=completion_count,
                        expected_completion_count=expected,
                        completion_ratio=round(completion_ratio, 6),
                        spike_count=spike_count,
                        latency_steps=latency,
                    )
                )
                key = (item.owner_id, item.source_id)
                self._items[key] = replace(item, access_count=item.access_count + 1)
            matches.sort(
                key=lambda item: (
                    -item.score,
                    item.latency_steps if item.latency_steps is not None else 10_000,
                    item.source_id,
                )
            )
            symbolic_top = candidates[0][1].source_id if candidates else None
            neural_top = next(
                (item.source_id for item in matches if item.neural_activation > 0.0),
                None,
            )
            return EpisodicNeuralRecall(
                matches=tuple(matches[:limit]),
                candidate_count=len(candidates),
                activated_count=sum(item.neural_activation > 0.0 for item in matches),
                network_spikes=network_spikes,
                interference_index=interference,
                pattern_separation_index=self._pattern_separation(candidate_items),
                symbolic_top_source_id=symbolic_top,
                neural_top_source_id=neural_top,
                simulation_steps=simulation_steps,
            )

    def strengthen(self, owner_id: str, source_id: str, *, rate: float = 0.1) -> bool:
        """Strengthen an engram after it survives attention selection."""
        if isinstance(rate, bool) or not math.isfinite(rate) or not 0.0 < rate <= 1.0:
            raise ValueError("strengthening rate must be in (0, 1]")
        with self._lock:
            key = (owner_id, source_id)
            item = self._items.get(key)
            if item is None:
                return False
            self._items[key] = replace(
                item,
                strength=min(1.0, item.strength + rate * (1.0 - item.strength)),
            )
            self._successful_strengthenings += 1
            return True

    def forget_source(self, source_id: str) -> int:
        with self._lock:
            selected = [
                key for key, item in self._items.items() if item.source_id == source_id
            ]
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
            selected = [
                key for key, item in self._items.items() if item.owner_id == owner_id
            ]
            for key in selected:
                del self._items[key]
            return len(selected)

    def reset(self) -> None:
        with self._lock:
            self._items.clear()
            self._evictions = 0
            self._encodings = 0
            self._successful_strengthenings = 0

    def snapshot(self) -> dict[str, int | str | bool]:
        with self._lock:
            return {
                "kind": "neural_episodic",
                "count": len(self._items),
                "capacity": self.capacity,
                "max_features_per_engram": self.max_features,
                "max_recall_candidates": self.max_candidates,
                "encodings": self._encodings,
                "evictions": self._evictions,
                "successful_strengthenings": self._successful_strengthenings,
                "state_storage": "volatile_process_memory",
                "retains_raw_content": False,
                "feature_codes_keyed_with_volatile_secret": True,
            }
