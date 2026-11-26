from __future__ import annotations

import json
from pathlib import Path

import pytest

from artificial_human.cognition.benchmark.external import ExternalDatasetError
from artificial_human.cognition.benchmark.external.event_log import load_xes
from artificial_human.cognition.benchmark.external.schema_event_log import (
    run_schema_event_log_suite,
)
from artificial_human.cognition.benchmark.runner import resolve_suites


def _write_xes(path: Path, *, case_count: int = 8) -> Path:
    traces: list[str] = []
    for case_index in range(case_count):
        day = case_index + 1
        events = []
        for minute, activity in enumerate(("Ticket created", "Work started", "Ticket closed")):
            events.append(
                "<event>"
                f'<string key="concept:name" value="{activity}"/>'
                f'<date key="time:timestamp" value="2026-01-{day:02d}T10:{minute:02d}:00Z"/>'
                '<string key="org:resource" value="Private operator"/>'
                '<string key="region" value="IN"/>'
                "</event>"
            )
        traces.append(
            "<trace>"
            f'<string key="concept:name" value="private-case-{case_index}"/>'
            + "".join(events)
            + "</trace>"
        )
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<log xmlns="http://www.xes-standard.org/">'
        + "".join(traces)
        + "</log>",
        encoding="utf-8",
    )
    return path


def test_xes_reader_hashes_case_ids_and_excludes_attributes_by_default(tmp_path: Path) -> None:
    path = _write_xes(tmp_path / "events.xes")
    log = load_xes(path)

    assert len(log.cases) == 8
    assert log.event_count == 24
    assert log.activity_count == 3
    assert len(log.sha256) == 64
    assert all(case.case_digest != "private-case-0" for case in log.cases)
    assert all(not event.attributes for case in log.cases for event in case.events)
    assert log.cases[0].events[0].event_type == "ticket.created"


def test_xes_reader_requires_allowlisting_and_rejects_sensitive_fields(tmp_path: Path) -> None:
    path = _write_xes(tmp_path / "events.xes")
    log = load_xes(path, attribute_keys=("region",))
    assert all(
        event.attributes == {"region": "IN"}
        for case in log.cases for event in case.events
    )
    with pytest.raises(ExternalDatasetError, match="Sensitive"):
        load_xes(path, attribute_keys=("org:resource",))


def test_xes_reader_rejects_unsupported_or_malformed_inputs(tmp_path: Path) -> None:
    csv_path = tmp_path / "events.csv"
    csv_path.write_text("case,activity,time\n1,A,2026-01-01T00:00:00Z\n", encoding="utf-8")
    with pytest.raises(ExternalDatasetError, match="Only .xes"):
        load_xes(csv_path)

    malformed = tmp_path / "malformed.xes"
    malformed.write_text("<log><trace>", encoding="utf-8")
    with pytest.raises(ExternalDatasetError, match="well-formed"):
        load_xes(malformed)


def test_external_schema_suite_uses_case_holdout_and_reports_aggregates_only(
    tmp_path: Path,
) -> None:
    path = _write_xes(tmp_path / "external.xes")
    result = run_schema_event_log_suite(path, seed=17, max_cases=8)

    assert result.suite == "schema-event-log"
    assert result.evidence_category.startswith("independently authored")
    assert result.splits == {"train_cases": 6, "eval_cases": 2}
    assert result.systems["schema_induction"]["training_proposal_count"] >= 1
    assert result.systems["schema_induction"]["heldout_eval_case_coverage"] == 1.0
    assert all(item["passed"] for item in result.checks)

    rendered = json.dumps(result.as_dict())
    for private_value in ("private-case-", "Private operator", "Ticket created"):
        assert private_value not in rendered


def test_external_schema_suite_is_deterministic_and_does_not_mutate_inputs(
    tmp_path: Path,
) -> None:
    path = _write_xes(tmp_path / "external.xes")
    first = run_schema_event_log_suite(path, seed=29, max_cases=8).as_dict()
    second = run_schema_event_log_suite(path, seed=29, max_cases=8).as_dict()
    assert first == second


def test_external_schema_suite_is_opt_in() -> None:
    with pytest.raises(ValueError, match="schema-event-log"):
        resolve_suites(["schema-event-log"])
    assert resolve_suites(
        ["schema-event-log"], available_external=["schema-event-log"]
    ) == ["schema-event-log"]
