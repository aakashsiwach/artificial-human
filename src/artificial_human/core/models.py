from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryKind(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    PREFERENCE = "preference"
    DECISION = "decision"
    WORKING = "working"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class Observation:
    source: str
    subject: str
    value: str
    scope: str = "manual"
    observed_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True, slots=True)
class HealthSignal:
    name: str
    value: float
    threshold: float
    direction: str = "above"
    source: str = "manual"
    observed_at: datetime = field(default_factory=utc_now)

    @property
    def abnormal(self) -> bool:
        if self.direction == "above":
            return self.value > self.threshold
        if self.direction == "below":
            return self.value < self.threshold
        raise ValueError("direction must be 'above' or 'below'")


@dataclass(frozen=True, slots=True)
class Incident:
    signal: HealthSignal
    severity: Severity
    summary: str
    containment_recommendation: str
    investigation_recommendation: str
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class RepairProposal:
    incident_id: str
    summary: str
    steps: tuple[str, ...]
    safety_checks: tuple[str, ...]
    status: str = "pending_review"
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=utc_now)
