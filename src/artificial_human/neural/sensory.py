"""Permission-scoped, synthetic-only sensory encoding.

This module never opens devices. Callers supply bounded synthetic signals or
local test fixtures after explicitly granting a modality-scoped permission.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from artificial_human.neural.population import PopulationCode, PopulationEncoder
from artificial_human.neural.simulation import NeuralSimulation, SimulationStep


class SensoryModality(StrEnum):
    VISION = "vision"
    AUDITION = "audition"
    TOUCH = "touch"
    VESTIBULAR = "vestibular"
    PROPRIOCEPTION = "proprioception"
    INTEROCEPTION = "interoception"


class SensoryProvenance(StrEnum):
    SYNTHETIC = "synthetic"
    LOCAL_FIXTURE = "local_fixture"


class RetentionPolicy(StrEnum):
    EPHEMERAL = "ephemeral"
    SESSION = "session"


@dataclass(frozen=True, slots=True)
class SensoryEvent:
    modality: SensoryModality
    values: tuple[float | None, ...]
    timestamp: int
    confidence: float = 1.0
    provenance: SensoryProvenance = SensoryProvenance.SYNTHETIC

    def __post_init__(self) -> None:
        if not isinstance(self.modality, SensoryModality):
            raise ValueError("modality must be a SensoryModality")
        if not isinstance(self.provenance, SensoryProvenance):
            raise ValueError("provenance must be explicit")
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, int) or self.timestamp < 0:
            raise ValueError("timestamp must be a non-negative integer")
        if not self.values or len(self.values) > 256:
            raise ValueError("sensory event must contain 1..256 samples")
        if not any(value is not None for value in self.values):
            raise ValueError("sensory event must contain at least one observed sample")
        if any(
            value is not None
            and (isinstance(value, bool) or not math.isfinite(value) or abs(value) > 1_000_000.0)
            for value in self.values
        ):
            raise ValueError("sensory values must be finite and within [-1000000, 1000000], or None")
        if isinstance(self.confidence, bool) or not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class SensoryPermission:
    modality: SensoryModality
    purpose: str
    granted_at: int
    expires_at: int
    retention: RetentionPolicy
    max_events: int
    max_samples: int


class PermissionDeniedError(PermissionError):
    """Raised when a modality has no active, explicit grant."""


class SensoryPermissionLedger:
    """In-memory least-privilege grants; an empty ledger denies every modality."""

    def __init__(self) -> None:
        self._grants: dict[SensoryModality, SensoryPermission] = {}
        self._uses: dict[SensoryModality, int] = {}

    def grant(
        self,
        modality: SensoryModality,
        *,
        purpose: str,
        granted_at: int,
        expires_at: int,
        retention: RetentionPolicy = RetentionPolicy.EPHEMERAL,
        max_events: int = 128,
        max_samples: int = 256,
        explicit_consent: bool = False,
    ) -> SensoryPermission:
        if not explicit_consent:
            raise PermissionDeniedError("explicit consent is required to grant a modality")
        if not isinstance(modality, SensoryModality) or not isinstance(retention, RetentionPolicy):
            raise ValueError("invalid modality or retention policy")
        if not purpose.strip():
            raise ValueError("permission purpose is required")
        if (
            isinstance(granted_at, bool) or not isinstance(granted_at, int)
            or isinstance(expires_at, bool) or not isinstance(expires_at, int)
            or granted_at < 0 or expires_at <= granted_at
        ):
            raise ValueError("permission time bounds are invalid")
        if isinstance(max_events, bool) or not isinstance(max_events, int) or not 1 <= max_events <= 10_000:
            raise ValueError("max_events must be in [1, 10000]")
        if isinstance(max_samples, bool) or not isinstance(max_samples, int) or not 1 <= max_samples <= 256:
            raise ValueError("max_samples must be in [1, 256]")
        permission = SensoryPermission(
            modality, purpose.strip(), granted_at, expires_at, retention, max_events, max_samples
        )
        self._grants[modality] = permission
        self._uses[modality] = 0
        return permission

    def revoke(self, modality: SensoryModality) -> None:
        self._grants.pop(modality, None)
        self._uses.pop(modality, None)

    def permission_for(self, modality: SensoryModality) -> SensoryPermission | None:
        return self._grants.get(modality)

    def authorize(self, event: SensoryEvent) -> SensoryPermission:
        permission = self._grants.get(event.modality)
        if permission is None:
            raise PermissionDeniedError(f"{event.modality.value} is disabled by default")
        if not permission.granted_at <= event.timestamp < permission.expires_at:
            raise PermissionDeniedError("event timestamp is outside the permission window")
        if len(event.values) > permission.max_samples:
            raise PermissionDeniedError("event exceeds the granted sample budget")
        used = self._uses[event.modality]
        if used >= permission.max_events:
            raise PermissionDeniedError("permission event budget is exhausted")
        self._uses[event.modality] = used + 1
        return permission


@dataclass(frozen=True, slots=True)
class EncodedPercept:
    modality: SensoryModality
    code: PopulationCode
    timestamp: int
    provenance: SensoryProvenance
    missing_samples: int
    permission_purpose: str


class ModalityEncoder(Protocol):
    modality: SensoryModality

    def encode(self, event: SensoryEvent, encoder: PopulationEncoder) -> EncodedPercept: ...


def _observed(event: SensoryEvent) -> tuple[float, ...]:
    return tuple(value for value in event.values if value is not None)


def _moments(values: tuple[float, ...]) -> tuple[float, float, float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    rms = math.sqrt(sum(value * value for value in values) / len(values))
    peak = max(abs(value) for value in values)
    return mean, math.sqrt(variance), rms, peak


def _percept(
    event: SensoryEvent,
    encoder: PopulationEncoder,
    features: tuple[float, ...],
    *,
    quality: float = 1.0,
) -> EncodedPercept:
    missing = len(event.values) - len(_observed(event))
    confidence = event.confidence * quality * (1.0 - missing / len(event.values))
    code = encoder.encode(features, timestamp=event.timestamp, confidence=confidence)
    return EncodedPercept(
        event.modality, code, event.timestamp, event.provenance, missing, ""
    )


class VisionEncoder:
    modality = SensoryModality.VISION

    def encode(self, event: SensoryEvent, encoder: PopulationEncoder) -> EncodedPercept:
        if event.modality is not self.modality:
            raise ValueError("vision encoder received another modality")
        values = _observed(event)
        mean, contrast, rms, peak = _moments(values)
        adjacent = sum(abs(b - a) for a, b in zip(values, values[1:])) / max(1, len(values) - 1)
        return _percept(event, encoder, (mean, contrast, adjacent, peak), quality=1.0 / (1.0 + rms * 0.01))


class AuditionEncoder:
    modality = SensoryModality.AUDITION

    def encode(self, event: SensoryEvent, encoder: PopulationEncoder) -> EncodedPercept:
        if event.modality is not self.modality:
            raise ValueError("audition encoder received another modality")
        values = _observed(event)
        mean, spread, rms, peak = _moments(values)
        crossings = sum((a < 0 <= b) or (b < 0 <= a) for a, b in zip(values, values[1:]))
        zero_crossing_rate = crossings / max(1, len(values) - 1)
        return _percept(event, encoder, (rms, spread, zero_crossing_rate, peak), quality=1.0 / (1.0 + abs(mean)))


class TouchEncoder:
    modality = SensoryModality.TOUCH

    def encode(self, event: SensoryEvent, encoder: PopulationEncoder) -> EncodedPercept:
        if event.modality is not self.modality:
            raise ValueError("touch encoder received another modality")
        values = _observed(event)
        mean, spread, rms, peak = _moments(values)
        contact_fraction = sum(value > 0.0 for value in values) / len(values)
        return _percept(event, encoder, (mean, peak, contact_fraction, spread))


class VestibularEncoder:
    modality = SensoryModality.VESTIBULAR

    def encode(self, event: SensoryEvent, encoder: PopulationEncoder) -> EncodedPercept:
        if event.modality is not self.modality:
            raise ValueError("vestibular encoder received another modality")
        values = _observed(event)
        mean, spread, rms, peak = _moments(values)
        change = sum(abs(b - a) for a, b in zip(values, values[1:])) / max(1, len(values) - 1)
        return _percept(event, encoder, (mean, spread, rms, change))


class ProprioceptionEncoder:
    modality = SensoryModality.PROPRIOCEPTION

    def encode(self, event: SensoryEvent, encoder: PopulationEncoder) -> EncodedPercept:
        if event.modality is not self.modality:
            raise ValueError("proprioception encoder received another modality")
        values = _observed(event)
        mean, spread, rms, peak = _moments(values)
        return _percept(event, encoder, (mean, spread, peak, rms))


class InteroceptionEncoder:
    modality = SensoryModality.INTEROCEPTION

    def encode(self, event: SensoryEvent, encoder: PopulationEncoder) -> EncodedPercept:
        if event.modality is not self.modality:
            raise ValueError("interoception encoder received another modality")
        values = _observed(event)
        mean, spread, rms, peak = _moments(values)
        return _percept(event, encoder, (mean, rms, spread, peak), quality=1.0 / (1.0 + spread))


class SensoryEncoderSuite:
    """Independent deterministic modality encoders sharing a population-code interface."""

    def __init__(self, units_per_feature: int = 9) -> None:
        self.population_encoder = PopulationEncoder(units_per_feature=units_per_feature)
        encoders: tuple[ModalityEncoder, ...] = (
            VisionEncoder(), AuditionEncoder(), TouchEncoder(), VestibularEncoder(),
            ProprioceptionEncoder(), InteroceptionEncoder(),
        )
        self._encoders = {encoder.modality: encoder for encoder in encoders}

    def encode(self, event: SensoryEvent, permission: SensoryPermission) -> EncodedPercept:
        if permission.modality is not event.modality:
            raise PermissionDeniedError("permission does not cover this sensory modality")
        result = self._encoders[event.modality].encode(event, self.population_encoder)
        return EncodedPercept(
            result.modality, result.code, result.timestamp, result.provenance,
            result.missing_samples, permission.purpose,
        )


@dataclass(frozen=True, slots=True)
class AlignedPercepts:
    timestamp: int
    window: int
    percepts: tuple[tuple[SensoryModality, EncodedPercept | None], ...]

    def get(self, modality: SensoryModality) -> EncodedPercept | None:
        return next((percept for key, percept in self.percepts if key is modality), None)


class SensoryPipeline:
    """Permission-gated, bounded in-memory ingestion; stores percepts, never raw events."""

    def __init__(
        self,
        *,
        ledger: SensoryPermissionLedger | None = None,
        suite: SensoryEncoderSuite | None = None,
        capacity: int = 256,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or not 1 <= capacity <= 10_000:
            raise ValueError("capacity must be an integer in [1, 10000]")
        self.ledger = ledger or SensoryPermissionLedger()
        self.suite = suite or SensoryEncoderSuite()
        self.capacity = capacity
        self._session_percepts: deque[EncodedPercept] = deque(maxlen=capacity)

    def ingest(self, event: SensoryEvent) -> EncodedPercept:
        permission = self.ledger.authorize(event)
        percept = self.suite.encode(event, permission)
        if permission.retention is RetentionPolicy.SESSION:
            self._session_percepts.append(percept)
        return percept

    def align(self, timestamp: int, *, window: int) -> AlignedPercepts:
        if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
            raise ValueError("alignment timestamp must be a non-negative integer")
        if isinstance(window, bool) or not isinstance(window, int) or window < 0:
            raise ValueError("alignment window must be a non-negative integer")
        aligned: list[tuple[SensoryModality, EncodedPercept | None]] = []
        for modality in SensoryModality:
            candidates = [
                percept for percept in self._session_percepts
                if percept.modality is modality
                and 0 <= timestamp - percept.timestamp <= window
            ]
            aligned.append((modality, max(candidates, key=lambda item: item.timestamp) if candidates else None))
        return AlignedPercepts(timestamp, window, tuple(aligned))

    def revoke(self, modality: SensoryModality) -> None:
        """Revoke future ingestion and immediately purge retained percepts for that modality."""
        self.ledger.revoke(modality)
        self._session_percepts = deque(
            (percept for percept in self._session_percepts if percept.modality is not modality),
            maxlen=self.capacity,
        )

    def clear_session(self) -> None:
        self._session_percepts.clear()

    @property
    def retained_count(self) -> int:
        return len(self._session_percepts)


def project_percept(
    percept: EncodedPercept,
    simulation: NeuralSimulation,
    neuron_ids: tuple[str, ...],
    *,
    current_scale: float = 450.0,
) -> SimulationStep:
    """Project a permission-approved population percept into the LIF substrate."""
    from artificial_human.neural.population import PopulationProjector

    return PopulationProjector(neuron_ids, current_scale=current_scale).drive(percept.code, simulation)
