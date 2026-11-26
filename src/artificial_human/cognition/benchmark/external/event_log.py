"""Bounded, privacy-conscious reader for external XES event logs.

The reader accepts only a local XES or XES.GZ file. It keeps case identifiers
hashed, stores no event text, and reads event attributes only when explicitly
allowlisted by the operator.
"""
from __future__ import annotations

import gzip
import hashlib
import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from artificial_human.cognition.benchmark.external import ExternalDatasetError, sha256_file


MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_CASES = 20_000
MAX_EVENTS = 250_000
MAX_EVENTS_PER_CASE = 128
_SENSITIVE_PARTS = frozenset({
    "account", "address", "auth", "card", "cookie", "cvv", "email", "ip",
    "name", "operator", "patient", "password", "phone", "resource", "secret", "session", "ssn", "token", "user",
})


@dataclass(frozen=True, slots=True)
class EventLogEvent:
    """One source-labelled activity with a normalized timestamp."""

    event_type: str
    occurred_at: str
    attributes: dict[str, str | int | float | bool]


@dataclass(frozen=True, slots=True)
class EventLogCase:
    """A trace whose source identifier is represented only by a digest."""

    case_digest: str
    events: tuple[EventLogEvent, ...]


@dataclass(frozen=True, slots=True)
class EventLogData:
    """Validated event log metadata and bounded case traces."""

    cases: tuple[EventLogCase, ...]
    event_count: int
    activity_count: int
    sha256: str


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _activity_slug(label: str, seen: dict[str, str]) -> str:
    normalized = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", ".", normalized.casefold()).strip(".")
    if not slug:
        raise ExternalDatasetError("XES activity labels must contain ASCII letters or digits")
    if not slug[0].isalpha():
        slug = "event." + slug
    if len(slug) > 64:
        slug = slug[:54].rstrip(".") + "." + hashlib.sha256(
            label.encode("utf-8")
        ).hexdigest()[:8]
    prior = seen.get(slug)
    if prior is not None and prior != label:
        suffix = hashlib.sha256(label.encode("utf-8")).hexdigest()[:8]
        slug = slug[:55].rstrip(".") + "." + suffix
    seen[slug] = label
    return slug


def _attribute_name(key: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
    if not normalized or len(normalized) > 40 or not normalized[0].isalpha():
        raise ExternalDatasetError(f"XES attribute key cannot be normalized safely: {key!r}")
    parts = set(normalized.split("_"))
    if parts.intersection(_SENSITIVE_PARTS):
        raise ExternalDatasetError(f"Sensitive XES attribute key is not eligible: {key!r}")
    return normalized


def _parse_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as error:
        raise ExternalDatasetError("XES event timestamp is not valid ISO-8601") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def _primitive(kind: str, value: str) -> str | int | float | bool | None:
    if kind == "string":
        return value if 0 < len(value) <= 120 else None
    if kind == "int":
        try:
            return int(value)
        except ValueError:
            return None
    if kind == "float":
        try:
            result = float(value)
        except ValueError:
            return None
        return result if result == result and abs(result) != float("inf") else None
    if kind == "boolean":
        lowered = value.casefold()
        return lowered == "true" if lowered in {"true", "false"} else None
    return None


def _values(element: ET.Element) -> dict[str, tuple[str, str]]:
    values: dict[str, tuple[str, str]] = {}
    for child in element:
        kind = _local_name(child.tag)
        key, value = child.attrib.get("key"), child.attrib.get("value")
        if key is not None and value is not None:
            values[key] = (kind, value)
    return values


def _open_xes(path: Path) -> BinaryIO:
    try:
        return gzip.open(path, "rb") if path.name.casefold().endswith(".xes.gz") else path.open("rb")
    except OSError as error:
        raise ExternalDatasetError(f"Cannot open local XES file: {error}") from error


def load_xes(path: Path, *, attribute_keys: tuple[str, ...] = ()) -> EventLogData:
    """Read bounded XES traces; no network access or raw identifiers leave this function."""
    if not path.is_file():
        raise ExternalDatasetError("The XES path must point to an existing local file")
    name = path.name.casefold()
    if not (name.endswith(".xes") or name.endswith(".xes.gz")):
        raise ExternalDatasetError("Only .xes and .xes.gz event logs are supported")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ExternalDatasetError(f"XES file exceeds the {MAX_FILE_BYTES}-byte safety limit")

    selected: dict[str, str] = {}
    for key in attribute_keys:
        if not key or key in {"concept:name", "time:timestamp"}:
            raise ExternalDatasetError("Activity and timestamp keys cannot be extra attributes")
        safe = _attribute_name(key)
        if safe in selected.values():
            raise ExternalDatasetError("Selected XES attribute keys must normalize uniquely")
        selected[key] = safe

    dataset_digest = sha256_file(path)
    cases: list[EventLogCase] = []
    case_digests: set[str] = set()
    seen_activities: dict[str, str] = {}
    event_count = 0
    root: ET.Element | None = None
    try:
        with _open_xes(path) as stream:
            for action, element in ET.iterparse(stream, events=("start", "end")):
                if action == "start":
                    if root is None:
                        root = element
                    continue
                if _local_name(element.tag) != "trace":
                    continue

                trace_values = _values(element)
                case_value = trace_values.get("concept:name") or trace_values.get(
                    "case:concept:name"
                )
                if case_value is None:
                    raise ExternalDatasetError("Every XES trace must have a case concept:name")
                case_id = case_value[1]
                if not case_id:
                    raise ExternalDatasetError("XES case identifiers cannot be empty")
                # Salt the case digest with the file fingerprint to prevent cross-dataset linkage.
                case_digest = hashlib.sha256(
                    (dataset_digest + ":" + case_id).encode("utf-8")
                ).hexdigest()[:24]
                if case_digest in case_digests:
                    raise ExternalDatasetError("XES trace case identifiers must be unique")
                case_digests.add(case_digest)

                events: list[EventLogEvent] = []
                for child in element:
                    if _local_name(child.tag) != "event":
                        continue
                    raw = _values(child)
                    activity = raw.get("concept:name")
                    timestamp = raw.get("time:timestamp")
                    if activity is None or timestamp is None:
                        raise ExternalDatasetError(
                            "Every XES event must contain concept:name and time:timestamp"
                        )
                    event_type = _activity_slug(activity[1], seen_activities)
                    attributes: dict[str, str | int | float | bool] = {}
                    for source_key, safe_key in selected.items():
                        item = raw.get(source_key)
                        if item is None:
                            continue
                        primitive = _primitive(item[0], item[1])
                        if primitive is not None:
                            attributes[safe_key] = primitive
                    events.append(EventLogEvent(
                        event_type=event_type,
                        occurred_at=_parse_timestamp(timestamp[1]),
                        attributes=attributes,
                    ))
                    event_count += 1
                    if event_count > MAX_EVENTS:
                        raise ExternalDatasetError(
                            f"XES log exceeds the {MAX_EVENTS}-event safety limit"
                        )
                    if len(events) > MAX_EVENTS_PER_CASE:
                        raise ExternalDatasetError(
                            f"An XES trace exceeds the {MAX_EVENTS_PER_CASE}-event safety limit"
                        )
                events.sort(key=lambda item: item.occurred_at)
                if events:
                    cases.append(EventLogCase(case_digest=case_digest, events=tuple(events)))
                if len(cases) > MAX_CASES:
                    raise ExternalDatasetError(
                        f"XES log exceeds the {MAX_CASES}-case safety limit"
                    )
                element.clear()
                if root is not None:
                    root.clear()
    except ET.ParseError as error:
        raise ExternalDatasetError("XES file is not well-formed XML") from error
    except (OSError, EOFError) as error:
        raise ExternalDatasetError(f"Cannot read XES file: {error}") from error

    if len(cases) < 4:
        raise ExternalDatasetError("At least four non-empty XES cases are required")
    return EventLogData(
        cases=tuple(cases),
        event_count=event_count,
        activity_count=len(seen_activities),
        sha256=dataset_digest,
    )
