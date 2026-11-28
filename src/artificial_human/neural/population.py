"""Explicit, deterministic distributed population codes.

The implementation is a small computational encoder/decoder with configurable
tuning curves. It stores population activations and metadata, never raw inputs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from artificial_human.neural.simulation import NeuralSimulation, SimulationStep


@dataclass(frozen=True, slots=True)
class PopulationCode:
    """Distributed activity pattern and metadata, without its source feature vector."""

    activations: tuple[float, ...]
    uncertainty: float
    timestamp: int
    feature_count: int
    units_per_feature: int
    preferred_values: tuple[float, ...]
    observed_features: tuple[bool, ...]
    scheme: str = "gaussian"

    def __post_init__(self) -> None:
        if (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, int)
            or self.timestamp < 0
        ):
            raise ValueError("timestamp must be a non-negative integer")
        if (
            isinstance(self.feature_count, bool)
            or not isinstance(self.feature_count, int)
            or not 1 <= self.feature_count <= 64
        ):
            raise ValueError("feature_count must be an integer in [1, 64]")
        if (
            isinstance(self.units_per_feature, bool)
            or not isinstance(self.units_per_feature, int)
            or not 2 <= self.units_per_feature <= 128
        ):
            raise ValueError("units_per_feature must be an integer in [2, 128]")
        if len(self.activations) != self.feature_count * self.units_per_feature:
            raise ValueError("activation count does not match population dimensions")
        if len(self.preferred_values) != self.units_per_feature:
            raise ValueError("preferred_values do not match units_per_feature")
        if len(self.observed_features) != self.feature_count:
            raise ValueError("observed_features do not match feature_count")
        if not all(isinstance(value, bool) for value in self.observed_features):
            raise ValueError("observed_features must contain booleans")
        if (
            isinstance(self.uncertainty, bool)
            or not math.isfinite(self.uncertainty)
            or not 0.0 <= self.uncertainty <= 1.0
        ):
            raise ValueError("uncertainty must be finite and in [0, 1]")
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in self.activations):
            raise ValueError("activations must be finite and in [0, 1]")
        if not all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in self.preferred_values):
            raise ValueError("preferred_values must be finite and in [-1, 1]")
        if not self.scheme:
            raise ValueError("coding scheme name is required")

    @property
    def values(self) -> tuple[float, ...]:
        """Compatibility alias for prior OrganismLoop consumers."""
        return self.activations

    def decode(self) -> tuple[float | None, ...]:
        """Decode observed features into normalized [-1, 1] coordinates.

        The decoder is an activity-weighted preferred-value average; it does
        not invert the input tanh or reconstruct original sensor units.
        """
        decoded: list[float | None] = []
        for feature_index, observed in enumerate(self.observed_features):
            if not observed:
                decoded.append(None)
                continue
            start = feature_index * self.units_per_feature
            responses = self.activations[start : start + self.units_per_feature]
            total = sum(responses)
            if total <= 1e-15:
                decoded.append(None)
                continue
            estimate = sum(
                response * preferred
                for response, preferred in zip(
                    responses, self.preferred_values, strict=True
                )
            ) / total
            decoded.append(min(1.0, max(-1.0, estimate)))
        return tuple(decoded)

    def mask_features(self, *feature_indices: int) -> PopulationCode:
        """Return a partial cue with selected feature banks hidden."""
        if len(set(feature_indices)) != len(feature_indices):
            raise ValueError("feature indices must be unique")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < self.feature_count
            for index in feature_indices
        ):
            raise ValueError("masked feature index is out of range")
        hidden = set(feature_indices)
        activations = tuple(
            0.0 if feature_index in hidden else value
            for feature_index in range(self.feature_count)
            for value in self.activations[
                feature_index * self.units_per_feature :
                (feature_index + 1) * self.units_per_feature
            ]
        )
        observed = tuple(
            was_observed and index not in hidden
            for index, was_observed in enumerate(self.observed_features)
        )
        return PopulationCode(
            activations=activations,
            uncertainty=self.uncertainty,
            timestamp=self.timestamp,
            feature_count=self.feature_count,
            units_per_feature=self.units_per_feature,
            preferred_values=self.preferred_values,
            observed_features=observed,
            scheme=self.scheme,
        )

    def sparsify(self, top_k: int = 2) -> PopulationCode:
        """Keep only the strongest top-k units in each observed feature bank."""
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= self.units_per_feature
        ):
            raise ValueError("top_k must be in [1, units_per_feature]")
        sparse = [0.0] * len(self.activations)
        for feature_index, observed in enumerate(self.observed_features):
            if not observed:
                continue
            start = feature_index * self.units_per_feature
            bank = self.activations[start : start + self.units_per_feature]
            winners = sorted(
                range(self.units_per_feature),
                key=lambda index: (-bank[index], index),
            )[:top_k]
            for index in winners:
                sparse[start + index] = bank[index]
        return PopulationCode(
            activations=tuple(sparse),
            uncertainty=self.uncertainty,
            timestamp=self.timestamp,
            feature_count=self.feature_count,
            units_per_feature=self.units_per_feature,
            preferred_values=self.preferred_values,
            observed_features=self.observed_features,
            scheme=f"{self.scheme}-top{top_k}",
        )

    def cosine_similarity(self, other: PopulationCode) -> float:
        """Compare only feature banks observed in both patterns."""
        if (
            self.feature_count != other.feature_count
            or self.units_per_feature != other.units_per_feature
            or self.preferred_values != other.preferred_values
        ):
            raise ValueError("population codes must share a compatible geometry")
        indices = [
            index
            for feature_index in range(self.feature_count)
            if self.observed_features[feature_index]
            and other.observed_features[feature_index]
            for index in range(
                feature_index * self.units_per_feature,
                (feature_index + 1) * self.units_per_feature,
            )
        ]
        if not indices:
            return 0.0
        left = [self.activations[index] for index in indices]
        right = [other.activations[index] for index in indices]
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm <= 1e-15 or right_norm <= 1e-15:
            return 0.0
        similarity = sum(a * b for a, b in zip(left, right, strict=True))
        return min(1.0, max(-1.0, similarity / (left_norm * right_norm)))


class PopulationEncoder:
    """Encode numeric features with overlapping Gaussian tuning curves."""

    def __init__(
        self,
        units_per_feature: int = 9,
        tuning_width: float = 0.28,
    ) -> None:
        if (
            isinstance(units_per_feature, bool)
            or not isinstance(units_per_feature, int)
            or not 2 <= units_per_feature <= 128
        ):
            raise ValueError("units_per_feature must be an integer in [2, 128]")
        if (
            isinstance(tuning_width, bool)
            or not math.isfinite(tuning_width)
            or not 0.01 <= tuning_width <= 2.0
        ):
            raise ValueError("tuning_width must be finite and in [0.01, 2.0]")
        self.units_per_feature = units_per_feature
        self.tuning_width = tuning_width
        self.preferred_values = tuple(
            -1.0 + 2.0 * index / (units_per_feature - 1)
            for index in range(units_per_feature)
        )

    def encode(
        self,
        features: tuple[float, ...],
        *,
        timestamp: int = 0,
        confidence: float = 1.0,
    ) -> PopulationCode:
        """Return a population activity pattern; source features are not retained."""
        if not 1 <= len(features) <= 64:
            raise ValueError("features must contain between 1 and 64 values")
        if not all(math.isfinite(value) for value in features):
            raise ValueError("features must be finite")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, int)
            or timestamp < 0
        ):
            raise ValueError("timestamp must be a non-negative integer")
        if (
            isinstance(confidence, bool)
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError("confidence must be finite and in [0, 1]")
        activations: list[float] = []
        for feature in features:
            normalized = math.tanh(feature)
            for preferred in self.preferred_values:
                scaled = (normalized - preferred) / self.tuning_width
                response = math.exp(-0.5 * scaled * scaled) * confidence
                activations.append(round(response, 12))
        return PopulationCode(
            activations=tuple(activations),
            uncertainty=round(1.0 - confidence, 12),
            timestamp=timestamp,
            feature_count=len(features),
            units_per_feature=self.units_per_feature,
            preferred_values=self.preferred_values,
            observed_features=tuple(True for _ in features),
        )

    def decode(self, code: PopulationCode) -> tuple[float | None, ...]:
        if code.units_per_feature != self.units_per_feature:
            raise ValueError("code was created with a different population size")
        if code.preferred_values != self.preferred_values:
            raise ValueError("code was created with different preferred values")
        return code.decode()


class OneHotBaselineEncoder:
    """Nearest-preferred-value one-hot encoder used as a quantizing baseline."""

    def __init__(self, units_per_feature: int = 9) -> None:
        self._geometry = PopulationEncoder(units_per_feature, tuning_width=0.28)
        self.units_per_feature = units_per_feature
        self.preferred_values = self._geometry.preferred_values

    def encode(
        self,
        features: tuple[float, ...],
        *,
        timestamp: int = 0,
        confidence: float = 1.0,
    ) -> PopulationCode:
        dense = self._geometry.encode(
            features, timestamp=timestamp, confidence=confidence
        )
        activations = [0.0] * len(dense.activations)
        for feature_index in range(dense.feature_count):
            start = feature_index * self.units_per_feature
            bank = dense.activations[start : start + self.units_per_feature]
            winner = max(
                range(self.units_per_feature),
                key=lambda index: (bank[index], -index),
            )
            activations[start + winner] = confidence
        return PopulationCode(
            activations=tuple(activations),
            uncertainty=dense.uncertainty,
            timestamp=dense.timestamp,
            feature_count=dense.feature_count,
            units_per_feature=self.units_per_feature,
            preferred_values=self.preferred_values,
            observed_features=dense.observed_features,
            scheme="one-hot",
        )


@dataclass(frozen=True, slots=True)
class FeatureVectorBaseline:
    """Direct normalized feature baseline, without a distributed population."""

    values: tuple[float, ...]
    uncertainty: float
    timestamp: int


class FeatureVectorBaselineEncoder:
    """Hand-coded tanh feature-vector baseline for comparison."""

    def encode(
        self,
        features: tuple[float, ...],
        *,
        timestamp: int = 0,
        confidence: float = 1.0,
    ) -> FeatureVectorBaseline:
        if not 1 <= len(features) <= 64 or not all(math.isfinite(x) for x in features):
            raise ValueError("features must contain 1..64 finite values")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, int)
            or timestamp < 0
            or isinstance(confidence, bool)
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError("invalid timestamp or confidence")
        return FeatureVectorBaseline(
            values=tuple(round(math.tanh(value), 12) for value in features),
            uncertainty=round(1.0 - confidence, 12),
            timestamp=timestamp,
        )


@dataclass(frozen=True, slots=True)
class StoredPopulationPattern:
    identifier: str
    code: PopulationCode


@dataclass(frozen=True, slots=True)
class PatternCompletion:
    identifier: str
    completed_code: PopulationCode
    similarity: float
    completed_features: tuple[float | None, ...]


class PopulationAssociativeMemory:
    """Bounded in-memory code store with overlap-aware cosine cue retrieval."""

    def __init__(self, capacity: int = 128) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or not 1 <= capacity <= 10_000:
            raise ValueError("capacity must be an integer in [1, 10000]")
        self.capacity = capacity
        self._patterns: list[StoredPopulationPattern] = []
        self._geometry: tuple[int, int, tuple[float, ...]] | None = None

    def store(self, identifier: str, code: PopulationCode) -> None:
        if not identifier or not identifier.strip():
            raise ValueError("pattern identifier is required")
        if any(pattern.identifier == identifier for pattern in self._patterns):
            raise ValueError(f"duplicate pattern identifier: {identifier}")
        geometry = (
            code.feature_count,
            code.units_per_feature,
            code.preferred_values,
        )
        if self._geometry is not None and geometry != self._geometry:
            raise ValueError("population code geometry does not match stored patterns")
        if len(self._patterns) >= self.capacity:
            raise RuntimeError("population associative memory capacity reached")
        self._geometry = geometry
        self._patterns.append(StoredPopulationPattern(identifier, code))

    def complete(self, cue: PopulationCode) -> PatternCompletion | None:
        if not any(cue.observed_features):
            return None
        if not self._patterns:
            return None
        cue_geometry = (
            cue.feature_count,
            cue.units_per_feature,
            cue.preferred_values,
        )
        if cue_geometry != self._geometry:
            raise ValueError("population cue geometry does not match stored patterns")
        best: StoredPopulationPattern | None = None
        best_similarity = -1.0
        for pattern in self._patterns:
            similarity = cue.cosine_similarity(pattern.code)
            if similarity > best_similarity:
                best = pattern
                best_similarity = similarity
        if best is None:
            return None
        return PatternCompletion(
            identifier=best.identifier,
            completed_code=best.code,
            similarity=round(best_similarity, 12),
            completed_features=best.code.decode(),
        )

    def clear(self) -> None:
        self._patterns.clear()
        self._geometry = None

    def __len__(self) -> int:
        return len(self._patterns)


class PopulationProjector:
    """Drive named LIF units from a population code without passing raw features."""

    def __init__(
        self,
        neuron_ids: tuple[str, ...],
        *,
        current_scale: float = 450.0,
    ) -> None:
        if not neuron_ids or len(set(neuron_ids)) != len(neuron_ids):
            raise ValueError("neuron_ids must be non-empty and unique")
        if any(not identifier or not identifier.strip() for identifier in neuron_ids):
            raise ValueError("neuron identifiers must be non-empty")
        if (
            isinstance(current_scale, bool)
            or not math.isfinite(current_scale)
            or not 0.0 < current_scale <= 1_000.0
        ):
            raise ValueError("current_scale must be finite and in (0, 1000]")
        self.neuron_ids = neuron_ids
        self.current_scale = current_scale

    def drive(
        self,
        code: PopulationCode,
        simulation: NeuralSimulation,
    ) -> SimulationStep:
        if len(self.neuron_ids) != len(code.activations):
            raise ValueError("projector neuron count must equal population unit count")
        if set(self.neuron_ids) != set(simulation.neurons):
            raise ValueError("projector neuron IDs must match simulation topology")
        currents = {
            identifier: activation * self.current_scale
            for identifier, activation in zip(
                self.neuron_ids, code.activations, strict=True
            )
        }
        return simulation.step(currents)


def run_population_coding_comparison() -> dict[str, object]:
    """Run deterministic probes for generalization, interference, separation and completion."""
    dense_encoder = PopulationEncoder()
    one_hot_encoder = OneHotBaselineEncoder()

    near = dense_encoder.encode((0.0, 0.5))
    moderately_different = dense_encoder.encode((0.2, 0.5))
    distant = dense_encoder.encode((0.8, 0.5))
    dense_near_similarity = near.cosine_similarity(moderately_different)
    dense_distant_similarity = near.cosine_similarity(distant)
    sparse_similarity = near.sparsify(top_k=2).cosine_similarity(
        moderately_different.sparsify(top_k=2)
    )

    first = dense_encoder.encode((0.01, -0.8), timestamp=1)
    second = dense_encoder.encode((0.03, 0.8), timestamp=2)
    dense_memory = PopulationAssociativeMemory(capacity=2)
    dense_memory.store("pattern-a", first)
    dense_memory.store("pattern-b", second)
    dense_retrieval = dense_memory.complete(second.mask_features(1))

    first_one_hot = one_hot_encoder.encode((0.01, -0.8), timestamp=1)
    second_one_hot = one_hot_encoder.encode((0.03, 0.8), timestamp=2)
    one_hot_memory = PopulationAssociativeMemory(capacity=2)
    one_hot_memory.store("pattern-a", first_one_hot)
    one_hot_memory.store("pattern-b", second_one_hot)
    one_hot_retrieval = one_hot_memory.complete(second_one_hot.mask_features(1))

    complete_memory = PopulationAssociativeMemory(capacity=2)
    complete_memory.store("target-pattern", first)
    complete_memory.store("distractor-pattern", dense_encoder.encode((-0.7, 0.8)))
    completion = complete_memory.complete(first.mask_features(1))

    code = dense_encoder.encode((0.0, 0.5), timestamp=7)
    neuron_ids = tuple(
        f"population-unit-{index}" for index in range(len(code.activations))
    )
    simulation = NeuralSimulation(
        max_neurons=len(neuron_ids),
        max_synapses=1,
        max_pending_events=1,
    )
    from artificial_human.neural.simulation import LIFNeuron

    for identifier in neuron_ids:
        simulation.add_neuron(LIFNeuron(identifier))
    spikes = PopulationProjector(neuron_ids).drive(code, simulation).spikes

    return {
        "scheme": "independent Gaussian tuning banks per normalized numeric feature",
        "generalization": {
            "near_similarity": round(dense_near_similarity, 8),
            "distant_similarity": round(dense_distant_similarity, 8),
            "near_is_more_similar": dense_near_similarity > dense_distant_similarity,
        },
        "one_hot_interference": {
            "same_bin_collision": first_one_hot.activations[:9] == second_one_hot.activations[:9],
            "dense_retrieval": dense_retrieval.identifier if dense_retrieval else None,
            "one_hot_retrieval": one_hot_retrieval.identifier if one_hot_retrieval else None,
        },
        "pattern_separation": {
            "dense_similarity": round(dense_near_similarity, 8),
            "top_two_similarity": round(sparse_similarity, 8),
            "overlap_reduced": sparse_similarity < dense_near_similarity,
        },
        "pattern_completion": {
            "retrieved_identifier": completion.identifier if completion else None,
            "missing_feature_recovered": (
                completion.completed_features[1] is not None if completion else False
            ),
            "cue_missing_features": [1],
        },
        "lif_projection": {
            "population_units": len(code.activations),
            "spike_count": len(spikes),
            "spiking_units": list(spikes),
            "timestamp_preserved": code.timestamp,
        },
        "hand_coded_feature_baseline": list(
            FeatureVectorBaselineEncoder().encode((0.0, 0.5)).values
        ),
        "limitations": [
            "Gaussian tuning is a computational choice, not fitted biological response data.",
            "The decoder returns normalized feature estimates, not original sensor units.",
            "Top-k separation is a task-specific trade-off and may reduce smooth generalization.",
            "The associative store is a small bounded code-retrieval experiment, not a full hippocampal circuit.",
        ],
    }


class _FeatureEncoder(Protocol):
    def encode(
        self,
        features: tuple[float, ...],
        *,
        timestamp: int = 0,
        confidence: float = 1.0,
    ) -> PopulationCode: ...
