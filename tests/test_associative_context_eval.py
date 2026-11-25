from artificial_human.cognition.benchmark.associative_context_eval import (
    run_associative_context_suite,
)
from artificial_human.cognition.benchmark.associative_memory_eval import PROBES


def test_context_scoped_associations_prevent_cross_context_recall_leakage() -> None:
    result = run_associative_context_suite(seed=17)

    assert result.suite == "associative-context"
    assert result.splits == {"tuning": 0, "eval": len(PROBES)}
    assert result.systems["session_scoped"]["target_recall@5"] == 1.0
    assert result.systems["session_scoped"]["cross_context_leakage@5"] == 0.0
    assert result.systems["shared_global_control"]["target_recall@5"] == 1.0
    assert all(item["passed"] for item in result.checks)
    assert result.extra["cleanup"]["remaining_associations"] == 0
    assert result.extra["cleanup"]["remaining_contexts"] == 0


def test_context_scoped_association_suite_is_deterministic() -> None:
    first = run_associative_context_suite(seed=2026)
    second = run_associative_context_suite(seed=2026)

    assert first.systems == second.systems
    assert first.by_condition == second.by_condition
    assert first.extra == second.extra
    assert first.checks == second.checks
