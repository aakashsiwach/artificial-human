from __future__ import annotations

from datetime import UTC, datetime

import pytest

from artificial_human.core.models import MemoryKind
from artificial_human.memory.repository import MemoryNotFoundError, MemoryRepository
from artificial_human.memory.temporal import (
    TemporalIntent,
    TimeRange,
    interpret_temporal_query,
    parse_timestamp,
    to_utc_iso,
)

REFERENCE = datetime(2026, 6, 15, 12, tzinfo=UTC)


def test_timestamps_require_timezone_and_are_stored_in_utc() -> None:
    with pytest.raises(ValueError, match="timezone"):
        parse_timestamp("2026-01-01T10:00:00")
    with pytest.raises(ValueError, match="Invalid"):
        parse_timestamp("not a date")
    assert to_utc_iso("2026-01-01T10:00:00+05:30") == "2026-01-01T04:30:00.000000+00:00"
    assert to_utc_iso("2026-01-01T10:00:00Z") == "2026-01-01T10:00:00.000000+00:00"


@pytest.mark.parametrize(
    ("query", "intent"),
    [
        ("Where does the user work now?", TemporalIntent.CURRENT),
        ("Which city do I currently live in?", TemporalIntent.CURRENT),
        ("Where did the user work previously?", TemporalIntent.HISTORICAL),
        ("Where did I live in the past?", TemporalIntent.HISTORICAL),
        ("What did I do most recently?", TemporalIntent.RECENT),
        ("Show my trips in chronological order", TemporalIntent.CHRONOLOGICAL),
        ("What is the boiling point of water?", TemporalIntent.NONE),
    ],
)
def test_temporal_intent_cues(query: str, intent: TemporalIntent) -> None:
    assert interpret_temporal_query(query, reference_time=REFERENCE).intent is intent


def test_month_without_year_resolves_to_most_recent_started_month() -> None:
    january = interpret_temporal_query("Where did I work in January?", reference_time=REFERENCE)
    assert january.intent is TemporalIntent.RANGE
    assert january.time_range == TimeRange(
        datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    )
    december = interpret_temporal_query("What happened in December?", reference_time=REFERENCE)
    assert december.time_range is not None
    assert december.time_range.start == datetime(2025, 12, 1, tzinfo=UTC)
    explicit = interpret_temporal_query("in March 2024", reference_time=REFERENCE)
    assert explicit.time_range is not None
    assert explicit.time_range.start == datetime(2024, 3, 1, tzinfo=UTC)


def test_relative_ranges() -> None:
    yesterday = interpret_temporal_query("What did I do yesterday?", reference_time=REFERENCE)
    assert yesterday.time_range == TimeRange(
        datetime(2026, 6, 14, tzinfo=UTC), datetime(2026, 6, 15, tzinfo=UTC)
    )
    last_week = interpret_temporal_query("anything last week", reference_time=REFERENCE)
    assert last_week.intent is TemporalIntent.RANGE
    assert interpret_temporal_query("in 2025", reference_time=REFERENCE).time_range == TimeRange(
        datetime(2025, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)
    )


def test_time_range_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError):
        TimeRange(datetime(2026, 2, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC))


def _repository() -> tuple[MemoryRepository, str]:
    repository = MemoryRepository(":memory:")
    return repository, repository.grant_consent("owner").id


def test_memory_stores_distinct_temporal_fields_in_utc() -> None:
    repository, consent = _repository()
    record = repository.add(
        owner_id="owner", kind=MemoryKind.EPISODIC, content="Gave a talk",
        source="journal", consent_id=consent, occurred_at="2026-03-12T18:30:00+01:00",
    )
    assert record.occurred_at == "2026-03-12T17:30:00.000000+00:00"
    assert record.recorded_at == record.created_at
    assert record.valid_from is None
    with pytest.raises(ValueError, match="earlier"):
        repository.add(
            owner_id="owner", kind=MemoryKind.SEMANTIC, content="Bad interval", source="s",
            consent_id=consent, valid_from="2026-02-01T00:00:00Z",
            valid_to="2026-01-01T00:00:00Z",
        )
    with pytest.raises(ValueError, match="timezone"):
        repository.add(
            owner_id="owner", kind=MemoryKind.SEMANTIC, content="Naive", source="s",
            consent_id=consent, valid_from="2026-02-01T00:00:00",
        )


def test_supersession_preserves_history_and_provenance() -> None:
    repository, consent = _repository()
    old = repository.add(
        owner_id="owner", kind=MemoryKind.SEMANTIC, content="Works at Company A",
        source="statement:2025", consent_id=consent, fact_key="owner.employer",
        valid_from="2025-01-01T00:00:00Z",
    )
    new, link = repository.supersede(
        owner_id="owner", memory_id=old.id, content="Moved to Company B",
        source="statement:2026", consent_id=consent, valid_from="2026-03-01T00:00:00Z",
    )
    assert repository.get("owner", old.id) == old  # the old record is untouched
    assert new.fact_key == "owner.employer"
    assert new.kind is MemoryKind.SEMANTIC
    assert (link.from_memory_id, link.to_memory_id, link.relation) == (
        new.id, old.id, "supersedes"
    )
    assert link.effective_at == "2026-03-01T00:00:00.000000+00:00"
    assert [item.source for item in repository.version_chain("owner", old.id)] == [
        "statement:2025", "statement:2026"
    ]
    assert [item.id for item in repository.version_chain("owner", new.id)] == [old.id, new.id]
    with pytest.raises(ValueError, match="already superseded"):
        repository.supersede(
            owner_id="owner", memory_id=old.id, content="Branch", source="s",
            consent_id=consent,
        )


def test_inferred_memory_cannot_supersede_asserted_memory() -> None:
    repository, consent = _repository()
    asserted = repository.add(
        owner_id="owner", kind=MemoryKind.SEMANTIC, content="Lives in Pune", source="owner",
        consent_id=consent,
    )
    with pytest.raises(ValueError, match="inferred"):
        repository.supersede(
            owner_id="owner", memory_id=asserted.id, content="Probably lives in Mumbai",
            source="model guess", consent_id=consent, epistemic_status="inferred",
        )
    with pytest.raises(ValueError, match="epistemic_status"):
        repository.add(
            owner_id="owner", kind=MemoryKind.SEMANTIC, content="x", source="s",
            consent_id=consent, epistemic_status="certain",
        )


def test_contradictions_are_linked_not_resolved() -> None:
    repository, consent = _repository()
    first = repository.add(
        owner_id="owner", kind=MemoryKind.SEMANTIC, content="Allergic to peanuts",
        source="medical form", consent_id=consent,
    )
    second = repository.add(
        owner_id="owner", kind=MemoryKind.SEMANTIC, content="No food allergies",
        source="chat", consent_id=consent,
    )
    link = repository.link_contradiction("owner", first.id, second.id, note="check")
    assert repository.link_contradiction("owner", second.id, first.id).id == link.id
    assert {item.id for item in repository.iter_visible("owner")} == {first.id, second.id}
    with pytest.raises(ValueError):
        repository.link_contradiction("owner", first.id, first.id)
    with pytest.raises(MemoryNotFoundError):
        repository.link_contradiction("someone-else", first.id, second.id)


def test_correction_updates_in_place_and_records_update_time() -> None:
    repository, consent = _repository()
    record = repository.add(
        owner_id="owner", kind=MemoryKind.PREFERENCE, content="Prefers tea",
        source="owner", consent_id=consent,
    )
    corrected = repository.correct(
        "owner", record.id, content="Prefers green tea", valid_from="2025-01-01T00:00:00Z"
    )
    assert corrected.content == "Prefers green tea"
    assert corrected.updated_at is not None
    assert corrected.source == "owner" and corrected.created_at == record.created_at
    assert corrected.valid_from == "2025-01-01T00:00:00.000000+00:00"
    cleared = repository.correct("owner", record.id, valid_from=None)
    assert cleared.valid_from is None
    with pytest.raises(MemoryNotFoundError):
        repository.correct("other", record.id, content="hijack")


def test_injected_clock_controls_recorded_time() -> None:
    moment = datetime(2025, 5, 1, 9, tzinfo=UTC)
    repository = MemoryRepository(":memory:", clock=lambda: moment)
    consent = repository.grant_consent("owner")
    record = repository.add(
        owner_id="owner", kind=MemoryKind.SEMANTIC, content="Fact", source="s",
        consent_id=consent.id,
    )
    assert record.recorded_at == "2025-05-01T09:00:00+00:00"
