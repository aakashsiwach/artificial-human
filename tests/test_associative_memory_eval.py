from artificial_human.cognition.benchmark.associative_memory_eval import (
    RANDOM_LINK_REPETITIONS,
    PROBES,
    run_associative_memory_suite,
)


def test_associative_memory_retrieval_beats_keyword_and_random_link_controls() -> None:
    result = run_associative_memory_suite(seed=17)

    assert result.suite == "associative-memory"
    assert len(PROBES) == 9
    assert result.splits == {"tuning": 0, "eval": len(PROBES)}
    assert result.extra["random_link_repetitions"] == RANDOM_LINK_REPETITIONS

    learned = result.systems["learned_associations"]
    keyword = result.systems["keyword_only"]
    random_link = result.systems["random_link"]

    assert learned["target_recall@5"] == 1.0
    assert keyword["target_recall@5"] == 0.0
    assert learned["target_mrr"] > keyword["target_mrr"]
    assert learned["target_mrr"] > random_link["target_mrr"]
    assert random_link["target_recall@5"] < learned["target_recall@5"]
    assert all(item["passed"] for item in result.checks)


def test_associative_memory_suite_is_reproducible_for_a_fixed_seed() -> None:
    first = run_associative_memory_suite(seed=2026)
    second = run_associative_memory_suite(seed=2026)

    assert first.systems == second.systems
    assert first.by_condition == second.by_condition
    assert first.extra == second.extra
    assert first.checks == second.checks
