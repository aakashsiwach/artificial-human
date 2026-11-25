"""Evaluate context-scoped associative recall and cross-session leakage."""

from __future__ import annotations

from statistics import fmean
from typing import Any

from artificial_human.cognition.associative_memory import AssociativeKnowledgeNetwork
from artificial_human.cognition.benchmark.associative_memory_eval import PROBES, TOP_K
from artificial_human.cognition.benchmark.common import DEFAULT_SEED, SuiteResult, check


def _target_metrics(activations: tuple[Any, ...], target_fact_id: str, source_fact_id: str) -> dict[str, Any]:
    ranked_ids = tuple(item.fact.fact_id for item in activations)
    try:
        rank: int | None = ranked_ids.index(target_fact_id) + 1
    except ValueError:
        rank = None
    return {
        "target_rank": rank,
        "target_recall@5": float(rank is not None and rank <= TOP_K),
        "false_associations": sum(
            fact_id not in {source_fact_id, target_fact_id}
            for fact_id in ranked_ids[:TOP_K]
        ),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "target_recall@5": fmean(row["target_recall@5"] for row in rows),
        "false_associations_per_query": fmean(row["false_associations"] for row in rows),
    }


def run_associative_context_suite(seed: int = DEFAULT_SEED) -> SuiteResult:
    """Compare session-isolated links with a shared global-graph control."""
    scoped = AssociativeKnowledgeNetwork()
    shared = AssociativeKnowledgeNetwork()
    same_context_rows: list[dict[str, Any]] = []
    cross_context_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    by_probe: dict[str, dict[str, dict[str, Any]]] = {}
    contexts: list[str] = []

    # Train one pair in each conversation context. A held-out cue is then issued
    # in that same context and in a different context with no training history.
    for index, probe in enumerate(PROBES):
        training_context = f"conversation-{index:02d}"
        isolated_context = f"unseen-{index:02d}"
        contexts.append(training_context)

        scoped.process(probe.training_query, context_id=training_context, limit=TOP_K)
        shared.process(probe.training_query, context_id="global", limit=TOP_K)

        same = _target_metrics(
            scoped.process(probe.cue_query, context_id=training_context, limit=TOP_K),
            probe.target_fact_id,
            probe.source_fact_id,
        )
        cross = _target_metrics(
            scoped.process(probe.cue_query, context_id=isolated_context, limit=TOP_K),
            probe.target_fact_id,
            probe.source_fact_id,
        )
        global_result = _target_metrics(
            shared.process(probe.cue_query, context_id="global", limit=TOP_K),
            probe.target_fact_id,
            probe.source_fact_id,
        )

        same_context_rows.append(same)
        cross_context_rows.append(cross)
        global_rows.append(global_result)
        by_probe[probe.name] = {
            "same_context": same,
            "different_context": cross,
            "shared_global_control": global_result,
        }

    same_summary = _aggregate(same_context_rows)
    cross_summary = _aggregate(cross_context_rows)
    global_summary = _aggregate(global_rows)

    # The benchmark verifies cleanup as well as isolation: learned associations
    # are removed when their owning context is cleared.
    removed = sum(scoped.forget_context(context) for context in contexts)
    scoped_after_cleanup = scoped.snapshot()
    shared_before_reset = shared.snapshot()
    shared.reset()

    checks = [
        check(
            "same-context cues recover every co-activated target",
            same_summary["target_recall@5"] == 1.0,
            f"Recall@5={same_summary['target_recall@5']:.3f}",
        ),
        check(
            "different contexts do not retrieve another context's target",
            cross_summary["target_recall@5"] == 0.0,
            f"Cross-context leakage@5={cross_summary['target_recall@5']:.3f}",
        ),
        check(
            "shared-global control exposes why context partitioning is needed",
            global_summary["target_recall@5"] == 1.0,
            f"Shared graph target recall@5={global_summary['target_recall@5']:.3f}",
        ),
        check(
            "context cleanup deletes every scoped association",
            removed > 0 and scoped_after_cleanup["association_count"] == 0,
            f"Removed {removed} context associations",
        ),
        check(
            "associative state stores no raw user text",
            scoped_after_cleanup["retains_user_text"] is False
            and shared_before_reset["retains_user_text"] is False,
            "Only source fact IDs, hashed/non-content context labels, and weights are retained",
        ),
    ]

    return SuiteResult(
        suite="associative-context",
        title="Context-sensitive associative recall and source monitoring",
        evidence_category="synthetic_software_benchmark",
        splits={"tuning": 0, "eval": len(PROBES)},
        systems={
            "session_scoped": {
                **same_summary,
                "cross_context_leakage@5": cross_summary["target_recall@5"],
            },
            "shared_global_control": {
                **global_summary,
                "cross_context_leakage@5": global_summary["target_recall@5"],
            },
        },
        by_condition=by_probe,
        checks=checks,
        findings=[
            (
                "Session-scoped associations preserve cue completion inside the trained "
                "context while blocking the same target from an unrelated context."
            ),
            (
                "The shared-global control retrieves the same targets across the whole graph, "
                "demonstrating the cross-context leakage risk of unpartitioned associations."
            ),
            (
                "This probe tests software context isolation, not source credibility, "
                "contradiction resolution, or biological source-monitoring mechanisms."
            ),
        ],
        extra={
            "probe_count": len(PROBES),
            "context_count": len(contexts),
            "contexts_removed": removed,
            "top_k": TOP_K,
            "seed": seed,
            "cleanup": {
                "remaining_associations": scoped_after_cleanup["association_count"],
                "remaining_contexts": scoped_after_cleanup["association_context_count"],
            },
            "privacy": {
                "state_storage": scoped_after_cleanup["state_storage"],
                "retains_user_text": scoped_after_cleanup["retains_user_text"],
            },
        },
    )
