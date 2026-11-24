"""Run the controlled memory-retrieval benchmark and write JSON + Markdown reports.

Protocol:

1. Ingest the synthetic dataset into a fresh in-memory store with a
   deterministic clock (so "recorded at" times are reproducible).
2. Choose the semantic similarity threshold using **only the tuning split**.
3. Evaluate every variant on the **eval split** with that fixed threshold.

Nothing about the eval split influences any parameter.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from artificial_human.core.models import MemoryKind
from artificial_human.memory.benchmark.metrics import (
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from artificial_human.memory.embeddings import EmbeddingProvider
from artificial_human.memory.repository import MemoryRepository
from artificial_human.memory.search import SearchConfig, SearchFilters
from artificial_human.memory.system import LongTermMemory
from artificial_human.memory.temporal import parse_timestamp

DATASET_PATH = Path(__file__).with_name("dataset.json")
K_VALUES = (1, 3, 5)
RETRIEVAL_DEPTH = 10
PRIMARY_K = 5
THRESHOLD_GRID = tuple(round(0.40 + 0.01 * step, 2) for step in range(46))

# Main comparison (required baselines) followed by ablations of the full system.
VARIANTS: dict[str, dict[str, Any]] = {
    "lexical": {"use_lexical": True, "use_semantic": False, "use_temporal_intent": False},
    "semantic": {"use_lexical": False, "use_semantic": True, "use_temporal_intent": False},
    "hybrid": {"use_lexical": True, "use_semantic": True, "use_temporal_intent": False},
    "hybrid_temporal": {"use_lexical": True, "use_semantic": True, "use_temporal_intent": True},
    "ablation_lexical_temporal": {
        "use_lexical": True, "use_semantic": False, "use_temporal_intent": True,
    },
    "ablation_semantic_temporal": {
        "use_lexical": False, "use_semantic": True, "use_temporal_intent": True,
    },
    "ablation_hybrid_temporal_no_similarity_threshold": {
        "use_lexical": True, "use_semantic": True, "use_temporal_intent": True,
        "semantic_min_similarity": -1.0,
    },
    "ablation_hybrid_temporal_no_contradiction_expansion": {
        "use_lexical": True, "use_semantic": True, "use_temporal_intent": True,
        "expand_contradictions": False,
    },
}
MAIN_VARIANTS = ("lexical", "semantic", "hybrid", "hybrid_temporal")


def load_dataset(path: Path | None = None) -> dict[str, Any]:
    raw = (path or DATASET_PATH).read_bytes()
    dataset: dict[str, Any] = json.loads(raw)
    # Hash with normalised line endings so Windows and Linux checkouts agree.
    dataset["_sha256"] = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    _validate_dataset(dataset)
    return dataset


def _validate_dataset(dataset: Mapping[str, Any]) -> None:
    keys = [item["key"] for item in dataset["memories"]]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate memory keys in benchmark dataset")
    known = set(keys)
    ids = [query["id"] for query in dataset["queries"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate query ids in benchmark dataset")
    texts: dict[str, set[str]] = {"tuning": set(), "eval": set()}
    for query in dataset["queries"]:
        if query["split"] not in texts:
            raise ValueError(f"Unknown split {query['split']!r}")
        texts[query["split"]].add(query["text"].strip().lower())
        for key in [*query["relevant"], *query.get("forbidden", [])]:
            if key not in known:
                raise ValueError(f"Query {query['id']} references unknown memory {key!r}")
    if texts["tuning"] & texts["eval"]:
        raise ValueError("Tuning and eval splits share query text")


@dataclass
class _Store:
    memory: LongTermMemory
    key_to_id: dict[str, str]
    id_to_key: dict[str, str]
    owners: dict[str, str]
    index_reports: dict[str, dict[str, Any]]


def build_store(dataset: Mapping[str, Any], provider: EmbeddingProvider | None) -> _Store:
    clock = {"now": parse_timestamp("2020-01-01T00:00:00+00:00")}
    repository = MemoryRepository(":memory:", clock=lambda: clock["now"])
    owners: dict[str, str] = dict(dataset["owners"])
    consents = {alias: repository.grant_consent(owner).id for alias, owner in owners.items()}
    memory = LongTermMemory(repository, provider)
    key_to_id: dict[str, str] = {}
    ordered = sorted(dataset["memories"], key=lambda item: parse_timestamp(item["recorded_at"]))
    for item in ordered:
        clock["now"] = parse_timestamp(item["recorded_at"])
        common = {
            "content": item["content"], "source": item["source"],
            "consent_id": consents[item["owner"]],
            "occurred_at": item.get("occurred_at"), "valid_from": item.get("valid_from"),
            "valid_to": item.get("valid_to"),
            "epistemic_status": item.get("epistemic_status", "asserted"),
        }
        if "supersedes" in item:
            record, _ = repository.supersede(
                owner_id=owners[item["owner"]], memory_id=key_to_id[item["supersedes"]],
                kind=MemoryKind(item["kind"]), **common,
            )
        else:
            record = repository.add(
                owner_id=owners[item["owner"]], kind=MemoryKind(item["kind"]),
                fact_key=item.get("fact_key"), **common,
            )
        key_to_id[item["key"]] = record.id
        for other in item.get("contradicts", []):
            repository.link_contradiction(
                owners[item["owner"]], record.id, key_to_id[other], note="benchmark"
            )
    reports = {
        alias: memory.reindex(owner).as_dict() for alias, owner in owners.items()
    }
    return _Store(memory, key_to_id, {v: k for k, v in key_to_id.items()}, owners, reports)


def _run_query(
    store: _Store, dataset: Mapping[str, Any], query: Mapping[str, Any], config: SearchConfig
) -> dict[str, Any]:
    owner = store.owners[query["owner"]]
    response = store.memory.search(
        owner, query["text"], top_k=RETRIEVAL_DEPTH,
        filters=SearchFilters(kinds=tuple(MemoryKind(kind) for kind in query.get("kinds", []))),
        config=config, reference_time=parse_timestamp(dataset["reference_time"]),
    )
    ranked = [store.id_to_key[item.memory.id] for item in response.results]
    leaked = [item.memory.id for item in response.results if item.memory.owner_id != owner]
    relevant: dict[str, int] = dict(query["relevant"])
    outcome: dict[str, Any] = {
        "id": query["id"], "category": query["category"], "text": query["text"],
        "ranked": ranked, "relevant": relevant,
        "owner_leaks": len(leaked),
        "forbidden_returned": sorted(set(query.get("forbidden", [])) & set(ranked)),
        "temporal_intent": response.temporal_query.intent.value,
        "strategy": response.strategy,
    }
    if relevant:
        for k in K_VALUES:
            outcome[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
            outcome[f"precision@{k}"] = precision_at_k(ranked, relevant, k)
            outcome[f"ndcg@{k}"] = ndcg_at_k(ranked, relevant, k)
        outcome["mrr"] = reciprocal_rank(ranked, relevant, RETRIEVAL_DEPTH)
    else:
        outcome["rejected"] = not ranked
        outcome["false_positives@5"] = len(ranked[:PRIMARY_K])
    return outcome


def _aggregate(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [item for item in outcomes if item["relevant"]]
    empty = [item for item in outcomes if not item["relevant"]]
    summary: dict[str, Any] = {"queries": len(outcomes), "queries_with_relevant": len(scored)}
    for k in K_VALUES:
        for metric in ("recall", "precision", "ndcg"):
            summary[f"{metric}@{k}"] = round(mean([item[f"{metric}@{k}"] for item in scored]), 4)
    summary["mrr"] = round(mean([item["mrr"] for item in scored]), 4)
    summary["irrelevant_queries"] = len(empty)
    summary["rejection_rate"] = (
        round(mean([1.0 if item["rejected"] else 0.0 for item in empty]), 4) if empty else None
    )
    summary["owner_leaks"] = sum(item["owner_leaks"] for item in outcomes)
    summary["forbidden_returned"] = sum(len(item["forbidden_returned"]) for item in outcomes)
    return summary


def evaluate(
    store: _Store, dataset: Mapping[str, Any], config: SearchConfig, split: str
) -> dict[str, Any]:
    queries = [query for query in dataset["queries"] if query["split"] == split]
    outcomes = [_run_query(store, dataset, query, config) for query in queries]
    categories = sorted({item["category"] for item in outcomes})
    return {
        "overall": _aggregate(outcomes),
        "by_category": {
            category: _aggregate([item for item in outcomes if item["category"] == category])
            for category in categories
        },
        "queries": outcomes,
    }


def tune_threshold(
    store: _Store, dataset: Mapping[str, Any], base: SearchConfig
) -> tuple[float, list[dict[str, float]]]:
    """Pick the similarity threshold maximising MRR + rejection rate on the tuning split."""
    table = []
    best: tuple[float, float] | None = None
    for threshold in THRESHOLD_GRID:
        result = evaluate(store, dataset, replace(base, semantic_min_similarity=threshold), "tuning")
        overall = result["overall"]
        objective = overall["mrr"] + (overall["rejection_rate"] or 0.0)
        table.append({
            "threshold": threshold, "mrr": overall["mrr"],
            "rejection_rate": overall["rejection_rate"] or 0.0,
            "objective": round(objective, 4),
        })
        if best is None or objective > best[1] + 1e-9:
            best = (threshold, objective)
    assert best is not None
    return best[0], table


def run_benchmark(
    provider: EmbeddingProvider | None,
    *,
    dataset_path: Path | None = None,
    variants: Sequence[str] | None = None,
    threshold: float | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    notify = progress or (lambda _message: None)
    notify("ingesting synthetic dataset")
    store = build_store(dataset, provider)
    model = store.memory.index.model_info()
    semantic_available = model is not None
    base = SearchConfig()

    tuning: dict[str, Any]
    if not semantic_available:
        tuning = {"performed": False, "reason": "no embedding backend"}
        chosen = base.semantic_min_similarity
    elif threshold is not None:
        tuning = {"performed": False, "reason": "threshold supplied explicitly"}
        chosen = threshold
    else:
        notify("tuning semantic similarity threshold on the tuning split")
        chosen, table = tune_threshold(store, dataset, replace(base, **VARIANTS["hybrid_temporal"]))
        tuning = {
            "performed": True, "split": "tuning",
            "objective": "mean reciprocal rank + rejection rate (tuning split only)",
            "grid": [THRESHOLD_GRID[0], THRESHOLD_GRID[-1], 0.01], "table": table,
        }
    tuning["selected_threshold"] = chosen

    results: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for name in variants or VARIANTS:
        settings = VARIANTS[name]
        if settings.get("use_semantic") and not semantic_available:
            skipped[name] = "requires an embedding backend"
            continue
        notify(f"evaluating {name}")
        config = replace(base, semantic_min_similarity=chosen)
        config = replace(config, **settings)
        evaluated = evaluate(store, dataset, config, "eval")
        evaluated["config"] = config.as_dict()
        results[name] = evaluated

    return {
        "benchmark": dataset["name"],
        "dataset_version": dataset["version"],
        "dataset_sha256": dataset["_sha256"],
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "reference_time": dataset["reference_time"],
        "k_values": list(K_VALUES),
        "retrieval_depth": RETRIEVAL_DEPTH,
        "embedding_model": model.public() if model else None,
        "index": store.index_reports,
        "splits": {
            split: sum(1 for query in dataset["queries"] if query["split"] == split)
            for split in ("tuning", "eval")
        },
        "threshold_tuning": tuning,
        "variants": results,
        "skipped_variants": skipped,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    variants: Mapping[str, Any] = report["variants"]
    model = report["embedding_model"]
    lines = [
        "# Memory retrieval benchmark report",
        "",
        f"- Benchmark: `{report['benchmark']}` v{report['dataset_version']} "
        f"(sha256 `{report['dataset_sha256'][:12]}`)",
        f"- Generated: {report['generated_at']} with Python {report['python']}",
        f"- Reference time for temporal queries: {report['reference_time']}",
        "- Embedding model: "
        + (f"`{model['model_id']}` ({model['backend']}, {model['dimensions']} dims, "
           f"{model['locality']}, {model['version']})" if model else "none (lexical only)"),
        f"- Queries: {report['splits']['tuning']} tuning, {report['splits']['eval']} eval "
        "(metrics below are eval split only)",
        f"- Semantic similarity threshold: {_fmt(report['threshold_tuning']['selected_threshold'])}"
        + (" (selected on tuning split)" if report["threshold_tuning"]["performed"] else ""),
        "",
        "Precision@K is the standard definition (relevant hits in top K divided by K); "
        "most queries have one or two relevant memories, so its ceiling is low. "
        "Rejection rate is the share of queries with no relevant memory that returned nothing.",
        "",
        "## Overall results (eval split)",
        "",
        "| Variant | Recall@1 | Recall@5 | Precision@1 | Precision@5 | MRR | nDCG@5 "
        "| Rejection | Owner leaks |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, result in variants.items():
        overall = result["overall"]
        lines.append(
            f"| {name} | {_fmt(overall['recall@1'])} | {_fmt(overall['recall@5'])} "
            f"| {_fmt(overall['precision@1'])} | {_fmt(overall['precision@5'])} "
            f"| {_fmt(overall['mrr'])} | {_fmt(overall['ndcg@5'])} "
            f"| {_fmt(overall['rejection_rate'])} | {overall['owner_leaks']} |"
        )
    if report["skipped_variants"]:
        lines += ["", "Skipped: " + ", ".join(
            f"`{name}` ({reason})" for name, reason in report["skipped_variants"].items()
        )]
    categories = sorted({
        category for result in variants.values() for category in result["by_category"]
    })
    shown = [name for name in variants if name in MAIN_VARIANTS] or list(variants)
    lines += [
        "",
        "## Per-category results (MRR, or rejection rate for categories without relevant items)",
        "",
        "| Category | " + " | ".join(shown) + " |",
        "| --- | " + " | ".join("---" for _ in shown) + " |",
    ]
    for category in categories:
        cells = []
        for name in shown:
            summary = variants[name]["by_category"].get(category)
            if summary is None:
                cells.append("n/a")
            elif summary["queries_with_relevant"]:
                extra = (
                    f" / rej {_fmt(summary['rejection_rate'])}"
                    if summary["irrelevant_queries"] else ""
                )
                cells.append(f"{_fmt(summary['mrr'])}{extra}")
            else:
                cells.append(f"rej {_fmt(summary['rejection_rate'])}")
        lines.append(f"| {category} | " + " | ".join(cells) + " |")

    main = [name for name in MAIN_VARIANTS if name in variants]
    if len(main) >= 2:
        lines += ["", "## Measured comparison", ""]
        reference = variants[main[0]]["overall"]
        for name in main[1:]:
            overall = variants[name]["overall"]
            lines.append(
                f"- `{name}` vs `{main[0]}`: MRR {overall['mrr'] - reference['mrr']:+.3f}, "
                f"nDCG@5 {overall['ndcg@5'] - reference['ndcg@5']:+.3f}, "
                f"Recall@5 {overall['recall@5'] - reference['recall@5']:+.3f}."
            )
        if "hybrid" in variants and "semantic" in variants:
            hybrid, semantic = variants["hybrid"]["overall"], variants["semantic"]["overall"]
            lines.append(
                f"- `hybrid` vs `semantic`: MRR {hybrid['mrr'] - semantic['mrr']:+.3f}, "
                f"nDCG@5 {hybrid['ndcg@5'] - semantic['ndcg@5']:+.3f}."
            )
        if "hybrid_temporal" in variants and "hybrid" in variants:
            full, hybrid = variants["hybrid_temporal"]["overall"], variants["hybrid"]["overall"]
            lines.append(
                f"- `hybrid_temporal` vs `hybrid`: MRR {full['mrr'] - hybrid['mrr']:+.3f}, "
                f"nDCG@5 {full['ndcg@5'] - hybrid['ndcg@5']:+.3f}."
            )
        lines.append(
            "- These differences come from one small synthetic dataset; treat them as "
            "a regression signal, not as evidence about general retrieval quality."
        )

    primary = "hybrid_temporal" if "hybrid_temporal" in variants else (
        "ablation_lexical_temporal" if "ablation_lexical_temporal" in variants
        else next(iter(variants), None)
    )
    if primary:
        failures = [
            item for item in variants[primary]["queries"]
            if (item["relevant"] and item["recall@5"] < 1.0)
            or (not item["relevant"] and not item["rejected"])
            or item["owner_leaks"] or item["forbidden_returned"]
        ]
        lines += ["", f"## Failure cases for `{primary}`", ""]
        if not failures:
            lines.append("None on this dataset.")
        for item in failures:
            expected = ", ".join(item["relevant"]) or "(nothing)"
            lines.append(
                f"- `{item['id']}` ({item['category']}) \"{item['text']}\" — expected "
                f"{expected}; top results: {', '.join(item['ranked'][:5]) or '(none)'}"
            )
    tuning = report["threshold_tuning"]
    if tuning["performed"]:
        lines += [
            "",
            "## Threshold tuning (tuning split only)",
            "",
            f"Objective: {tuning['objective']}. Ties keep the lowest threshold. "
            "Selected: " + _fmt(tuning["selected_threshold"]) + ".",
        ]
    lines += [
        "",
        "## Reproduce",
        "",
        "```",
        "pip install -e \".[dev,semantic]\"",
        "artificial-human memory-eval --backend fastembed --output-dir docs/benchmarks",
        "```",
        "",
    ]
    return "\n".join(lines)


def write_reports(report: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "memory-retrieval-results.json"
    markdown_path = output_dir / "memory-retrieval-report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path
