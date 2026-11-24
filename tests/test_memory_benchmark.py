from __future__ import annotations

import json

import pytest
from conftest import ConceptEmbedding

from artificial_human.cli import main
from artificial_human.memory.benchmark.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from artificial_human.memory.benchmark.runner import (
    MAIN_VARIANTS,
    load_dataset,
    render_markdown,
    run_benchmark,
    write_reports,
)


def test_metric_definitions() -> None:
    relevant = {"a": 2, "b": 1}
    ranked = ["x", "a", "y", "b"]
    assert recall_at_k(ranked, relevant, 1) == 0.0
    assert recall_at_k(ranked, relevant, 4) == 1.0
    assert precision_at_k(ranked, relevant, 2) == 0.5
    assert reciprocal_rank(ranked, relevant, 10) == 0.5
    assert reciprocal_rank(["x"], relevant, 10) == 0.0
    assert ndcg_at_k(["a", "b"], relevant, 2) == pytest.approx(1.0)
    assert 0.0 < ndcg_at_k(["b", "a"], relevant, 2) < 1.0
    with pytest.raises(ValueError):
        recall_at_k(ranked, {}, 5)


def test_dataset_is_synthetic_split_and_consistent() -> None:
    dataset = load_dataset()
    splits = {query["split"] for query in dataset["queries"]}
    assert splits == {"tuning", "eval"}
    categories = {query["category"] for query in dataset["queries"] if query["split"] == "eval"}
    assert categories == {
        "exact_keyword", "paraphrase", "conceptual", "episodic", "temporal_chronological",
        "historical_fact", "contradiction", "type_filter", "irrelevant", "owner_isolation",
    }
    assert all(item["source"].startswith("synthetic:") for item in dataset["memories"])
    assert all(owner.startswith("bench-") for owner in dataset["owners"].values())


def test_lexical_only_benchmark_runs_offline_and_reports_metrics() -> None:
    report = run_benchmark(None)
    assert report["embedding_model"] is None
    assert set(report["variants"]) == {"lexical", "ablation_lexical_temporal"}
    assert "semantic" in report["skipped_variants"]
    for result in report["variants"].values():
        overall = result["overall"]
        assert {"recall@5", "precision@5", "mrr", "ndcg@5", "rejection_rate"} <= set(overall)
        assert overall["owner_leaks"] == 0
        assert overall["forbidden_returned"] == 0
    assert "Memory retrieval benchmark report" in render_markdown(report)


def test_benchmark_with_semantic_backend_tunes_on_tuning_split_and_is_deterministic() -> None:
    first = run_benchmark(ConceptEmbedding())
    second = run_benchmark(ConceptEmbedding())
    assert set(MAIN_VARIANTS) <= set(first["variants"])
    assert first["threshold_tuning"]["performed"] is True
    assert first["threshold_tuning"]["split"] == "tuning"
    eval_ids = {item["id"] for item in first["variants"]["hybrid"]["queries"]}
    assert all(query_id.startswith("eval-") for query_id in eval_ids)
    for name in first["variants"]:
        assert first["variants"][name]["overall"] == second["variants"][name]["overall"]
        assert first["variants"][name]["overall"]["owner_leaks"] == 0


def test_reports_are_written_as_json_and_markdown(tmp_path) -> None:
    report = run_benchmark(None)
    json_path, markdown_path = write_reports(report, tmp_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["benchmark"] == report["benchmark"]
    assert markdown_path.read_text(encoding="utf-8").startswith("# Memory retrieval")


def test_cli_memory_eval_and_simulation(tmp_path, capsys) -> None:
    assert main(["memory-eval", "--backend", "none", "--output-dir", str(tmp_path)]) == 0
    assert (tmp_path / "memory-retrieval-results.json").exists()
    assert main(["simulate"]) == 0
    output = capsys.readouterr().out
    assert "memory_demo" in output
    assert "moved to Company B" in output


def test_cli_memory_index_and_search(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ARTIFICIAL_HUMAN_DB_PATH", str(tmp_path / "cli.sqlite3"))
    monkeypatch.setenv("ARTIFICIAL_HUMAN_OWNER_ID", "cli-owner")
    monkeypatch.delenv("ARTIFICIAL_HUMAN_EMBEDDING_BACKEND", raising=False)
    from artificial_human.core.models import MemoryKind
    from artificial_human.memory.repository import MemoryRepository

    repository = MemoryRepository(tmp_path / "cli.sqlite3")
    consent = repository.grant_consent("cli-owner")
    repository.add(
        owner_id="cli-owner", kind=MemoryKind.SEMANTIC, content="Private tea preference",
        source="test", consent_id=consent.id,
    )
    assert main(["memory-index", "status"]) == 0
    assert '"retrieval_mode": "lexical_only"' in capsys.readouterr().out
    assert main(["memory-search", "tea"]) == 0
    output = capsys.readouterr().out
    assert '"strategy": "lexical"' in output
    assert "Private tea preference" not in output  # content hidden unless requested
    assert main(["memory-search", "tea", "--show-content"]) == 0
    assert "Private tea preference" in capsys.readouterr().out
