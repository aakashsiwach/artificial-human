"""Temporal representation and rule-based temporal query interpretation.

The memory store distinguishes several kinds of time (a bitemporal model):

* ``occurred_at`` — when an event happened (event time).
* ``recorded_at`` — when the system learned about it (transaction time; the
  ``created_at`` column, kept for backward compatibility).
* ``valid_from`` / ``valid_to`` — the interval during which a fact held.
* ``updated_at`` — when the record itself was last corrected.
* supersession links — when a newer record replaced an older one.

Temporal *intent* is inferred from query text with transparent rules. This is
an engineering heuristic, not a model of how the brain encodes time.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum


class TemporalStatus(StrEnum):
    VALID = "valid"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    NOT_YET_VALID = "not_yet_valid"
    NOT_YET_RECORDED = "not_yet_recorded"


class TemporalIntent(StrEnum):
    NONE = "none"
    CURRENT = "current"
    HISTORICAL = "historical"
    RANGE = "range"
    RECENT = "recent"
    CHRONOLOGICAL = "chronological"


def parse_timestamp(value: str | datetime) -> datetime:
    """Parse an ISO-8601 timestamp and require an explicit timezone."""
    if isinstance(value, datetime):
        parsed = value
    else:
        text = value.strip()
        if not text:
            raise ValueError("Timestamp cannot be empty")
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as error:
            raise ValueError(f"Invalid ISO-8601 timestamp: {value!r}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timestamps must include a timezone offset (for example +00:00 or Z)")
    return parsed.astimezone(UTC)


def to_utc_iso(value: str | datetime) -> str:
    """Normalise a timezone-aware timestamp to the UTC storage convention."""
    return parse_timestamp(value).isoformat(timespec="microseconds")


def optional_utc_iso(value: str | datetime | None) -> str | None:
    return None if value is None else to_utc_iso(value)


@dataclass(frozen=True, slots=True)
class TimeRange:
    """Half-open interval ``[start, end)``; either bound may be open."""

    start: datetime | None = None
    end: datetime | None = None

    def __post_init__(self) -> None:
        if self.start and self.end and self.start >= self.end:
            raise ValueError("Time range start must be earlier than its end")

    def contains(self, moment: datetime) -> bool:
        if self.start is not None and moment < self.start:
            return False
        return not (self.end is not None and moment >= self.end)

    def overlaps(self, start: datetime | None, end: datetime | None) -> bool:
        if self.end is not None and start is not None and start >= self.end:
            return False
        return not (self.start is not None and end is not None and end <= self.start)

    def as_dict(self) -> dict[str, str | None]:
        return {
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
        }


@dataclass(frozen=True, slots=True)
class TemporalQuery:
    intent: TemporalIntent = TemporalIntent.NONE
    time_range: TimeRange | None = None
    cues: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "intent": self.intent.value,
            "time_range": self.time_range.as_dict() if self.time_range else None,
            "cues": list(self.cues),
        }


_MONTHS = {name.lower(): index for index, name in enumerate(calendar.month_name) if name}
_MONTH_PATTERN = re.compile(
    r"\b(?:in|during|throughout|of)\s+(" + "|".join(_MONTHS) + r")(?:\s+(\d{4}))?\b",
    re.IGNORECASE,
)
_YEAR_PATTERN = re.compile(r"\b(?:in|during|throughout)\s+((?:19|20)\d{2})\b", re.IGNORECASE)
_CURRENT_CUES = (
    "now", "currently", "current", "these days", "at present", "presently",
    "nowadays", "today", "still",
)
_HISTORICAL_CUES = (
    "previously", "previous", "formerly", "former", "used to", "before that",
    "in the past", "originally", "earlier", "prior", "past",
)
_RECENT_CUES = ("most recent", "latest", "recently", "lately", "last time", "newest")
_CHRONOLOGICAL_CUES = (
    "chronological", "in order", "timeline", "sequence", "first", "earliest", "order of",
)


def _has_cue(text: str, cue: str) -> bool:
    return re.search(r"\b" + re.escape(cue) + r"\b", text) is not None


def _month_range(month: int, year: int) -> TimeRange:
    start = datetime(year, month, 1, tzinfo=UTC)
    end = datetime(year + 1, 1, 1, tzinfo=UTC) if month == 12 else datetime(
        year, month + 1, 1, tzinfo=UTC
    )
    return TimeRange(start, end)


def interpret_temporal_query(text: str, *, reference_time: datetime) -> TemporalQuery:
    """Infer temporal intent from explicit cues in the query text.

    Rules, in priority order: explicit month/year ranges, relative ranges
    ("yesterday", "last week/month/year"), historical cues, current cues,
    recency cues, chronological cues. A bare month without a year resolves to
    the most recent such month that has started at ``reference_time``.
    """
    lowered = text.lower()
    reference = parse_timestamp(reference_time)

    month_match = _MONTH_PATTERN.search(lowered)
    if month_match:
        month = _MONTHS[month_match.group(1).lower()]
        year = int(month_match.group(2)) if month_match.group(2) else reference.year
        if not month_match.group(2) and datetime(year, month, 1, tzinfo=UTC) > reference:
            year -= 1
        return TemporalQuery(
            TemporalIntent.RANGE, _month_range(month, year), (month_match.group(0),)
        )
    year_match = _YEAR_PATTERN.search(lowered)
    if year_match:
        year = int(year_match.group(1))
        return TemporalQuery(
            TemporalIntent.RANGE,
            TimeRange(datetime(year, 1, 1, tzinfo=UTC), datetime(year + 1, 1, 1, tzinfo=UTC)),
            (year_match.group(0),),
        )
    day_start = reference.replace(hour=0, minute=0, second=0, microsecond=0)
    if _has_cue(lowered, "yesterday"):
        return TemporalQuery(
            TemporalIntent.RANGE, TimeRange(day_start - timedelta(days=1), day_start),
            ("yesterday",),
        )
    for unit, days in (("week", 7), ("month", 31), ("year", 366)):
        cue = f"last {unit}"
        if _has_cue(lowered, cue) or _has_cue(lowered, f"past {unit}"):
            return TemporalQuery(
                TemporalIntent.RANGE,
                TimeRange(reference - timedelta(days=days), reference),
                (cue,),
            )

    historical = tuple(cue for cue in _HISTORICAL_CUES if _has_cue(lowered, cue))
    if historical:
        return TemporalQuery(TemporalIntent.HISTORICAL, None, historical)
    current = tuple(cue for cue in _CURRENT_CUES if _has_cue(lowered, cue))
    if current:
        return TemporalQuery(TemporalIntent.CURRENT, None, current)
    recent = tuple(cue for cue in _RECENT_CUES if _has_cue(lowered, cue))
    if recent:
        return TemporalQuery(TemporalIntent.RECENT, None, recent)
    chronological = tuple(cue for cue in _CHRONOLOGICAL_CUES if _has_cue(lowered, cue))
    if chronological:
        return TemporalQuery(TemporalIntent.CHRONOLOGICAL, None, chronological)
    return TemporalQuery()


def parse_stored_timestamp(value: str) -> datetime:
    """Parse a stored timestamp; legacy values without an offset are treated as UTC."""
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
