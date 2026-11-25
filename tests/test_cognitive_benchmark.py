from __future__ import annotations

import json

import pytest
from conftest import ConceptEmbedding

from artificial_human.cli import main
from artificial_human.cognition.attention import AttentionSystem
from artificial_human.cognition.benchmark import runner as runner_module
from artificial_human.cognition.benchmark.attention_eval import SYSTEMS, run_attention_suite
from artificial_human.cognition.benchmark.common import derive_seed, measured
from artificial_human.cognition.benchmark.datasets import (
    ATTENTION_CONDITIONS,
    generate_attention_tasks,
    prediction_items,
    validate_split_isolation,
)
from artificial_human.cognition.benchmark.integration_eval import SCENARIOS, run_integration_suite
from artificial_human.cognition.benchmark.memory_eval import run_memory_suite
from artificial_human.cognition.benchmark.metrics import (
    auroc,
    choose_threshold,
    detection_rates,
    kendall_tau,
    pairwise_accuracy,
    pearson,
)
from artificial_human.cognition.benchmark.prediction_eval import (
    existing_novelty,
    jaccard_distance,
    run_prediction_suite,
)
from artificial_human.cognition.benchmark.report import render_markdown
from artificial_human.cognition.benchmark.runner import (
    InvariantViolation,
    deterministic_view,
    resolve_suites,
    run_cognitive_evaluation,
    write_reports,
)
from artificial_human.cognition.benchmark.working_memory_eval import (
    ExistingWorkspace,
    run_working_memory_suite,
)
from artificial_human.cognition.perception import PerceptionSystem
from artificial_human.cognition.prediction import PredictionSystem
from artificial_human.memory.benchmark.runner import (
    _validate_dataset,
    compact_report,
    load_dataset,
    run_benchmark,
)

# ----------------------------------------------------------------- metrics


def test_pairwise_accuracy() -> None:
    grades = {"a": 2, "b": 1, "c": 0}
    assert pairwise_accuracy(["a", "b", "c"], grades) == 1.0
    assert pairwise_accuracy(["c", "b", "a"], grades) == 0.0
    assert pairwise_accuracy(["b", "a", "c"], grades) == pytest.approx(2 / 3)
    assert pairwise_accuracy(["x", "y"], {"x": 1, "y": 1}) == 1.0
    with pytest.raises(ValueError, match="missing"):
        pairwise_accuracy(["a"], grades)


def test_auroc_and_detection_rates() -> None:
    assert auroc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0]) == 0.0
    assert auroc([0.5, 0.5], [1, 0]) == 0.5
    for scores, labels in (([0.1], [1]), ([0.1, 0.2], [1])):
        with pytest.raises(ValueError):
            auroc(scores, labels)
    rates = detection_rates([0.9, 0.2, 0.7, 0.1], [1, 1, 0, 0], threshold=0.5)
    assert rates["false_positive_rate"] == 0.5
    assert rates["false_negative_rate"] == 0.5
    assert rates["balanced_accuracy"] == 0.5


def test_choose_threshold_maximises_balanced_accuracy() -> None:
    threshold = choose_threshold([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    assert 0.2 < threshold <= 0.8
    assert detection_rates([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], threshold)["accuracy"] == 1.0
    with pytest.raises(ValueError):
        choose_threshold([], [])


def test_pearson_and_kendall_tau() -> None:
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 1, 1], [1, 2, 3]) is None
    with pytest.raises(ValueError):
        pearson([1], [1])
    assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert kendall_tau(["c", "b", "a"], ["a", "b", "c"]) == -1.0
    assert kendall_tau(["a"], ["a", "b"]) == 1.0


# ---------------------------------------------------------------- datasets


def test_attention_tasks_are_deterministic_seeded_and_split() -> None:
    first, second = generate_attention_tasks(7), generate_attention_tasks(7)
    assert first == second
    assert first != generate_attention_tasks(8)
    validate_split_isolation({
        split: [task.id for task in first if task.split == split] for split in ("tuning", "eval")
    })
    assert {task.condition for task in first} == set(ATTENTION_CONDITIONS)
    for task in first:
        ids = [candidate.reference_id for candidate in task.candidates]
        assert len(ids) == len(set(ids)) == len(task.grades)
        assert any(task.grades.values())
    assert derive_seed(1, "a") == derive_seed(1, "a") != derive_seed(1, "b")


def test_cross_source_condition_uses_architecture_normalisation() -> None:
    task = next(t for t in generate_attention_tasks(3) if t.condition == "cross_source_normalisation")
    for kind in ("memory", "learning_example"):
        values = [c.relevance for c in task.candidates if c.kind == kind]
        assert max(values) == 1.0
    assert all(task.grades[c.reference_id] == 0 for c in task.candidates
               if c.kind == "learning_example")


def test_split_isolation_rejects_shared_tasks_and_prediction_split_is_fixed() -> None:
    with pytest.raises(ValueError, match="both"):
        validate_split_isolation({"tuning": ["x"], "eval": ["x"]})
    items = prediction_items()
    assert sum(item.split == "tuning" for item in items) == 18
    assert sum(item.split == "eval" for item in items) == 36
    assert len({item.id for item in items}) == len(items)


def test_malformed_memory_dataset_is_rejected() -> None:
    dataset = load_dataset()
    broken = {**dataset, "queries": [{**dataset["queries"][0], "relevant": {"nope": 1}}]}
    with pytest.raises(ValueError, match="unknown memory"):
        _validate_dataset(broken)
    source_query = next(q for q in dataset["queries"] if "expected_source" in q)
    wrong = {**dataset, "queries": [{**source_query, "expected_source": "synthetic:other"}]}
    with pytest.raises(ValueError, match="source"):
        _validate_dataset(wrong)
    with pytest.raises(ValueError, match="Unknown suite"):
        resolve_suites(["telepathy"])


# ------------------------------------------------------------------ suites


def test_attention_suite_uses_the_exact_existing_implementation() -> None:
    task = generate_attention_tasks(5)[0]
    expected = [
        item.reference_id
        for item in AttentionSystem().select(list(task.candidates), limit=len(task.candidates))
    ]
    assert SYSTEMS["existing_attention"](task.candidates) == expected
    result = run_attention_suite(5)
    assert set(result.systems) == {"existing_attention", "relevance_only", "salience_only",
                                   "input_order"}
    assert result.extra["consistency"]["repeat_run_identical"] == 1.0
    assert result.by_condition["salient_distractors"]["relevance_only"]["ndcg@k"] > (
        result.by_condition["salient_distractors"]["existing_attention"]["ndcg@k"]
    )


def test_working_memory_suite_measures_real_workspace() -> None:
    workspace = ExistingWorkspace(2)
    for step, item in enumerate(("a", "b", "c")):
        workspace.encode(item, item, 0.5, step)
    assert len(workspace.contents()) == 2
    result = run_working_memory_suite(11)
    assert all(values["capacity_violations"] == 0 for values in result.systems.values())
    assert all(item["passed"] for item in result.checks)
    assert result.by_condition["similarity_interference"]["existing_working_memory"][
        "difference"] == 0.0
    curve = result.extra["delayed_retention_curves"]["existing_working_memory"]
    assert curve["retention_by_intervening_items"]["0"] == 1.0


def test_prediction_suite_tunes_on_tuning_split_only() -> None:
    result = run_prediction_suite()
    items = prediction_items()
    tuning = [item for item in items if item.split == "tuning"]
    assert result.systems["existing_prediction"]["threshold_from_tuning_split"] == choose_threshold(
        [existing_novelty(item) for item in tuning], [item.unexpected for item in tuning]
    )
    item = items[0]
    direct = PredictionSystem().assess(PerceptionSystem().perceive(item.text), list(item.context))
    assert existing_novelty(item) == direct.novelty
    assert jaccard_distance(item) == 0.0
    assert result.by_condition["exact_repetition"]["existing_prediction"]["mean_novelty"] == 0.0
    assert "embedding_distance_reference" not in result.systems
    with_reference = run_prediction_suite(ConceptEmbedding())
    assert "embedding_distance_reference" in with_reference.systems


def test_memory_suite_reuses_benchmark_and_reports_new_metrics() -> None:
    result = run_memory_suite(None)
    lexical = result.systems["lexical"]
    for key in ("temporal_answer_accuracy", "source_attribution_accuracy",
                "contradiction_recall@5", "owner_leaks"):
        assert key in lexical
    assert lexical["owner_leaks"] == 0
    assert result.checks[0]["passed"]
    assert "source_attribution" in result.by_condition


def test_integration_scenarios_run_offline_and_guard_core_invariants() -> None:
    result = run_integration_suite(None)
    assert set(result.by_condition) == set(SCENARIOS)
    by_name = {(c["scenario"], c["name"]): c["passed"] for c in result.checks}
    assert all(passed for (scenario, _), passed in by_name.items()
               if scenario in ("B_temporal_knowledge", "E_contradiction_awareness"))
    assert by_name[("A_retain_and_recall", "conversation turns were not written to long-term memory")]
    assert by_name[("D_working_memory_interference", "capacity is never exceeded")]
    assert by_name[("F_context_switching_multi_turn", "sessions do not share working memory")]
    hybrid = run_integration_suite(ConceptEmbedding())
    assert set(hybrid.systems) == {"lexical_temporal", "hybrid_temporal"}


# ----------------------------------------------------------- runner/report


def test_full_evaluation_is_deterministic_for_a_seed() -> None:
    first = run_cognitive_evaluation(["attention", "working-memory", "prediction"], seed=3)
    second = run_cognitive_evaluation(["attention", "working-memory", "prediction"], seed=3)
    other = run_cognitive_evaluation(["attention"], seed=4)
    assert deterministic_view(first) == deterministic_view(second)
    assert deterministic_view(first)["suites"]["attention"] != deterministic_view(other)["suites"][
        "attention"]
    assert first["network_used_by_evaluation"] is False


def test_report_contains_all_required_sections(tmp_path) -> None:
    report = run_cognitive_evaluation(["all"], seed=1)
    markdown = render_markdown(report)
    for number in range(1, 15):
        assert f"## {number}. " in markdown
    paths = write_reports(report, tmp_path, ["json"])
    assert [path.name for path in paths] == ["cognitive-evaluation-results.json"]
    loaded = json.loads(paths[0].read_text(encoding="utf-8"))
    assert loaded["suites_run"] == ["attention", "working-memory", "prediction", "memory",
                                    "associative-memory", "associative-context",
                                    "associative-dynamics", "episodic-replay",
                                    "sleep-consolidation", "integration"]
    assert paths[0].stat().st_size < 200_000  # compact output


def test_measured_records_runtime_and_memory() -> None:
    result = measured(lambda: run_working_memory_suite(1))
    assert result.runtime_seconds > 0
    assert result.peak_memory_kib > 0


def test_invariant_violation_is_raised_for_owner_leakage(monkeypatch) -> None:
    from artificial_human.cognition.benchmark.common import SuiteResult, check

    def leaky() -> SuiteResult:
        result = SuiteResult("memory", "t", "e", {})
        result.checks = [check("no cross-owner leakage in any variant", False)]
        return result

    monkeypatch.setattr(runner_module, "_suite_runner", lambda *args: leaky)
    with pytest.raises(InvariantViolation):
        run_cognitive_evaluation(["memory"])


# --------------------------------------------------------------------- CLI


def test_cli_cognition_eval_formats_and_exit_codes(tmp_path, capsys, monkeypatch) -> None:
    assert main(["cognition-eval", "--suite", "prediction", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["suites_run"] == ["prediction"]

    assert main(["cognition-eval", "--suite", "attention", "--suite", "working-memory",
                 "--seed", "9", "--output-dir", str(tmp_path)]) == 0
    assert (tmp_path / "cognitive-evaluation-report.md").exists()
    assert (tmp_path / "cognitive-evaluation-results.json").exists()

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(runner_module, "run_cognitive_evaluation", explode)
    assert main(["cognition-eval", "--suite", "attention"]) == 1

    def violate(*args, **kwargs):
        raise InvariantViolation("leak")

    monkeypatch.setattr(runner_module, "run_cognitive_evaluation", violate)
    assert main(["cognition-eval"]) == 3
    with pytest.raises(SystemExit):
        main(["cognition-eval", "--suite", "telepathy"])


# --------------------------------------------- memory benchmark compatibility


def test_memory_benchmark_compact_output_keeps_aggregates_and_failures_only() -> None:
    report = run_benchmark(None)
    compact = compact_report(report)
    for name, variant in compact["variants"].items():
        assert "queries" not in variant
        assert variant["overall"] == report["variants"][name]["overall"]
        assert all(len(item["ranked"]) <= 5 for item in variant["failures"])
    assert "table" not in compact["threshold_tuning"]
