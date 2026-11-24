"""Long-term memory suite: wraps the existing memory retrieval benchmark.

The dataset, protocol (threshold tuned on the tuning split only), variants and
metrics live in ``artificial_human.memory.benchmark``. This module reuses them
instead of duplicating them, and summarises them for the cognitive report.
"""

from __future__ import annotations

from typing import Any

from artificial_human.cognition.benchmark.common import SuiteResult, check
from artificial_human.memory.benchmark.runner import MAIN_VARIANTS, run_benchmark
from artificial_human.memory.embeddings import EmbeddingProvider

HEADLINE = (
    "recall@5", "precision@5", "mrr", "ndcg@5", "rejection_rate",
    "temporal_answer_accuracy", "source_attribution_accuracy", "contradiction_recall@5",
    "owner_leaks",
)


def run_memory_suite(provider: EmbeddingProvider | None = None) -> SuiteResult:
    report = run_benchmark(provider)
    result = SuiteResult(
        suite="memory",
        title="Long-term memory recall (synthetic retrieval benchmark)",
        evidence_category="computational capability benchmark (synthetic, project-authored)",
        splits=dict(report["splits"]),
    )
    for name, variant in report["variants"].items():
        result.systems[name] = {key: variant["overall"].get(key) for key in HEADLINE}
        for category, summary in variant["by_category"].items():
            result.by_condition.setdefault(category, {})[name] = {
                "mrr": summary["mrr"] if summary["queries_with_relevant"] else None,
                "rejection_rate": summary["rejection_rate"],
            }
    leaks = sum(variant["overall"]["owner_leaks"] for variant in report["variants"].values())
    result.checks = [check("no cross-owner leakage in any variant", leaks == 0, leaks=leaks)]
    result.extra = {
        "dataset": {
            "name": report["benchmark"], "version": report["dataset_version"],
            "sha256": report["dataset_sha256"],
        },
        "embedding_model": report["embedding_model"],
        "semantic_threshold": report["threshold_tuning"]["selected_threshold"],
        "threshold_tuned_on": "tuning" if report["threshold_tuning"]["performed"] else None,
        "skipped_variants": report["skipped_variants"],
    }
    primary = next(
        (name for name in ("hybrid_temporal", "ablation_lexical_temporal") if name in report["variants"]),
        None,
    )
    if primary:
        result.failures = [
            {"variant": primary, "query": item["id"], "category": item["category"],
             "expected": list(item["relevant"]), "top_results": item["ranked"][:5]}
            for item in report["variants"][primary]["queries"]
            if (item["relevant"] and item["recall@5"] < 1.0)
            or (not item["relevant"] and not item["rejected"])
        ]
    shown = [name for name in MAIN_VARIANTS if name in result.systems] or list(result.systems)
    result.findings = [
        f"{name}: MRR {result.systems[name]['mrr']:.3f}, rejection "
        f"{_fmt(result.systems[name]['rejection_rate'])}, temporal answer accuracy "
        f"{_fmt(result.systems[name]['temporal_answer_accuracy'])}, source attribution "
        f"{_fmt(result.systems[name]['source_attribution_accuracy'])}."
        for name in shown
    ] + [
        "The benchmark is synthetic, English-only and project-authored. It is a regression "
        "signal, not independent validation. Multi-turn recall and context switching are "
        "measured through the full cognitive cycle in the integration suite.",
    ]
    return result


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{value:.3f}"
