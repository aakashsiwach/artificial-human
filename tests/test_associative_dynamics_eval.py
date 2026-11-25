from artificial_human.cognition.benchmark.associative_dynamics_eval import (
    PROBE_COUNT,
    run_associative_dynamics_suite,
)


def test_associative_dynamics_suppresses_conflicts_and_measures_interference() -> None:
    result = run_associative_dynamics_suite(seed=19)

    assert result.suite == "associative-dynamics"
    assert result.splits == {"tuning": 0, "eval": PROBE_COUNT}
    assert result.systems["contradiction_suppression"]["suppression_rate"] == 1.0
    assert result.systems["contradiction_suppression"]["contradiction_recall@1"] == 0.0
    assert result.systems["no_retrieval_reinforcement"]["target_recall_after_distractors@1"] == 0.0
    assert result.systems["retrieval_reinforced"]["target_recall_after_distractors@1"] == 1.0
    assert all(item["passed"] for item in result.checks)


def test_associative_dynamics_suite_is_deterministic() -> None:
    first = run_associative_dynamics_suite(seed=2026)
    second = run_associative_dynamics_suite(seed=2026)

    assert first.systems == second.systems
    assert first.by_condition == second.by_condition
    assert first.extra == second.extra
    assert first.checks == second.checks
