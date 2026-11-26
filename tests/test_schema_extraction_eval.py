from artificial_human.cognition.benchmark.schema_extraction_eval import (
    PROBE_COUNT,
    run_schema_extraction_suite,
)


def test_schema_extraction_suite_measures_generalisation_and_controls() -> None:
    result = run_schema_extraction_suite(seed=71)
    assert result.suite == "schema-extraction"
    assert result.splits == {"tuning": 0, "eval": PROBE_COUNT}

    metrics = result.systems["schema_induction"]
    assert metrics["recurrent_schema_detection_rate"] == 1.0
    assert metrics["variable_attribute_slot_rate"] == 1.0
    assert metrics["stable_attribute_constant_rate"] == 1.0
    assert metrics["sensitive_attribute_exclusion_rate"] == 1.0
    assert metrics["singleton_false_positive_rate"] == 0.0
    assert metrics["contradiction_false_positive_rate"] == 0.0
    assert metrics["preview_silent_write_violation_rate"] == 0.0
    assert metrics["approved_schema_provenance_rate"] == 1.0
    assert all(item["passed"] for item in result.checks)


def test_schema_extraction_suite_is_deterministic() -> None:
    first = run_schema_extraction_suite(seed=2026)
    second = run_schema_extraction_suite(seed=2026)
    assert first.systems == second.systems
    assert first.by_condition == second.by_condition
    assert first.extra == second.extra
    assert first.checks == second.checks
