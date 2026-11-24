from __future__ import annotations

from dataclasses import dataclass

from artificial_human.core.models import HealthSignal, Incident, Severity


@dataclass(frozen=True, slots=True)
class SignalPolicy:
    threshold: float
    direction: str
    severity: Severity
    explanation: str


DEFAULT_POLICIES: dict[str, SignalPolicy] = {
    "api_error_rate": SignalPolicy(0.05, "above", Severity.WARNING, "API error rate is elevated."),
    "heartbeat_age_seconds": SignalPolicy(60.0, "above", Severity.CRITICAL, "A monitored heartbeat is stale."),
    "queue_depth": SignalPolicy(100.0, "above", Severity.WARNING, "The work queue is growing."),
    "memory_rejection_count": SignalPolicy(10.0, "above", Severity.WARNING, "Many memory writes were rejected."),
    "available_memory_percent": SignalPolicy(10.0, "below", Severity.CRITICAL, "Available memory is critically low."),
}


class ImmuneSystem:
    """Turns configured health signals into incidents; it does not mutate the host."""

    def __init__(self, policies: dict[str, SignalPolicy] | None = None) -> None:
        self.policies = policies or DEFAULT_POLICIES
        self._incidents: list[Incident] = []

    def evaluate(self, name: str, value: float, *, source: str = "manual") -> tuple[HealthSignal | None, Incident | None]:
        policy = self.policies.get(name)
        if policy is None:
            return None, None
        signal = HealthSignal(
            name=name, value=value, threshold=policy.threshold,
            direction=policy.direction, source=source,
        )
        if not signal.abnormal:
            return signal, None
        incident = Incident(
            signal=signal,
            severity=policy.severity,
            summary=policy.explanation,
            containment_recommendation=(
                "Pause non-essential work and preserve diagnostic evidence; request operator review."
            ),
            investigation_recommendation=(
                "Inspect recent changes, correlated events, resource usage, and dependency health."
            ),
        )
        self._incidents.append(incident)
        return signal, incident

    def incidents(self) -> tuple[Incident, ...]:
        return tuple(self._incidents)
