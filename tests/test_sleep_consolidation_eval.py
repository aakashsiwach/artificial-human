from artificial_human.cognition.benchmark.sleep_consolidation_eval import (
    PROBE_COUNT,
    run_sleep_consolidation_suite,
)


def test_sleep_consolidation_suite_requires_recurrence_and_owner_review() -> None:
    result = run_sleep_consolidation_suite(seed=19)

    assert result.suite == "sleep-consolidation"
    assert result.splits == {"tuning": 0, "eval": PROBE_COUNT}
    metrics = result.systems["replay_pattern_detection"]
    assert metrics["recurring_pattern_proposal_rate"] == 1.0
    assert metrics["singleton_false_positive_rate"] == 0.0
    assert metrics["contradiction_false_positive_rate"] == 0.0
    approval = result.systems["owner_review"]
    assert approval["silent_write_violation_rate"] == 0.0
    assert approval["approved_inferred_provenance_rate"] == 1.0
    assert all(item["passed"] for item in result.checks)


def test_sleep_consolidation_suite_is_deterministic() -> None:
    first = run_sleep_consolidation_suite(seed=2026)
    second = run_sleep_consolidation_suite(seed=2026)

    assert first.systems == second.systems
    assert first.by_condition == second.by_condition
    assert first.extra == second.extra
    assert first.checks == second.checks
