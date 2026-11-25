from artificial_human.cognition.benchmark.episodic_replay_eval import (
    PROBE_COUNT,
    run_episodic_replay_suite,
)


def test_episodic_replay_suite_measures_neighbour_recall_and_isolation() -> None:
    result = run_episodic_replay_suite(seed=77)

    assert result.suite == "episodic-replay"
    assert result.splits == {"tuning": 0, "eval": PROBE_COUNT}
    assert result.systems["temporal_replay"]["exact_neighbour_recall@2"] == 1.0
    assert result.systems["no_replay_baseline"]["exact_neighbour_recall@2"] == 0.0
    assert result.systems["wrong_context_control"]["cross_context_leakage@2"] == 0.0
    assert all(item["passed"] for item in result.checks)


def test_episodic_replay_suite_is_deterministic() -> None:
    first = run_episodic_replay_suite(seed=2026)
    second = run_episodic_replay_suite(seed=2026)

    assert first.systems == second.systems
    assert first.by_condition == second.by_condition
    assert first.extra == second.extra
    assert first.checks == second.checks
