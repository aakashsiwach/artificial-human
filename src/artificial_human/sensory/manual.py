from artificial_human.core.models import Observation


class ManualSensor:
    """Accepts observations explicitly supplied by an owner or trusted caller."""

    def capture(self, *, source: str, subject: str, value: str, scope: str = "manual") -> Observation:
        if not source.strip() or not subject.strip() or not value.strip():
            raise ValueError("source, subject, and value are required")
        return Observation(source=source, subject=subject, value=value, scope=scope)
