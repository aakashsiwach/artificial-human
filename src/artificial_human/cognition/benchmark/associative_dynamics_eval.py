"""Controlled contradiction-suppression and retrieval-interference evaluation."""

from __future__ import annotations

from statistics import fmean
from typing import Any

from artificial_human.cognition.benchmark.common import (
    DEFAULT_SEED,
    SuiteResult,
    check,
)
from artificial_human.cognition.relational_memory import RelationalMemoryNetwork


PROBE_COUNT = 9


def run_associative_dynamics_suite(seed: int = DEFAULT_SEED) -> SuiteResult:
    """Measure explicit contradiction suppression and retrieval-strengthened retention."""
    contradiction_hits: list[float] = []
    contradiction_suppressions: list[float] = []
    baseline_retention: list[float] = []
    reinforced_retention: list[float] = []
    baseline_evictions: list[float] = []
    reinforced_evictions: list[float] = []
    by_probe: dict[str, dict[str, dict[str, Any]]] = {}

    for index in range(PROBE_COUNT):
        suffix = f"{seed % 100000:05d}-{index:02d}"
        context = f"contradiction-{suffix}"
        first = f"record-a-{suffix}"
        second = f"record-b-{suffix}"

        conflict_network = RelationalMemoryNetwork(decay=1.0)
        conflict_network.learn([first, second], context_id=context)
        before = conflict_network.snapshot()["association_count"]
        recalled = conflict_network.activate(
            [first],
            context_id=context,
            contradictions=[(first, second)],
        )
        conflict_network.learn(
            [first, second],
            context_id=context,
            contradictions=[(first, second)],
        )
        conflict_state = conflict_network.snapshot()
        contradiction_hits.append(float(any(item.memory_id == second for item in recalled)))
        contradiction_suppressions.append(float(
            before == 1
            and conflict_state["association_count"] == 0
            and conflict_state["contradiction_suppressions"] == 1
        ))

        cue = f"target-cue-{suffix}"
        target = f"target-record-{suffix}"
        distractor_cue = f"distractor-cue-{suffix}"
        distractor_target = f"distractor-record-{suffix}"
        interference_context = f"interference-{suffix}"

        baseline = RelationalMemoryNetwork(
            learning_rate=0.08, retrieval_rate=0.0, decay=0.995, max_links=1
        )
        reinforced = RelationalMemoryNetwork(
            learning_rate=0.08, retrieval_rate=0.2, decay=0.995, max_links=1
        )
        for network in (baseline, reinforced):
            network.learn([cue, target], context_id=interference_context)

        # Both systems receive the same cue opportunities. Only the adaptive
        # variant strengthens the used association.
        for _ in range(5):
            baseline.activate([], context_id=interference_context)
            reinforced.activate([cue], context_id=interference_context)

        for network in (baseline, reinforced):
            network.learn(
                [distractor_cue, distractor_target],
                context_id=interference_context,
            )

        baseline_result = baseline.activate([cue], context_id=interference_context)
        reinforced_result = reinforced.activate([cue], context_id=interference_context)
        baseline_state = baseline.snapshot()
        reinforced_state = reinforced.snapshot()
        baseline_found = float(any(item.memory_id == target for item in baseline_result))
        reinforced_found = float(any(item.memory_id == target for item in reinforced_result))
        baseline_retention.append(baseline_found)
        reinforced_retention.append(reinforced_found)
        baseline_evictions.append(float(baseline_state["evicted_links"]))
        reinforced_evictions.append(float(reinforced_state["evicted_links"]))
        by_probe[f"probe_{index + 1:02d}"] = {
            "contradiction": {
                "target_recalled": contradiction_hits[-1],
                "link_suppressed": contradiction_suppressions[-1],
                "links_before": before,
                "links_after": conflict_state["association_count"],
            },
            "interference": {
                "baseline_target_recalled": baseline_found,
                "reinforced_target_recalled": reinforced_found,
                "baseline_evictions": baseline_state["evicted_links"],
                "reinforced_evictions": reinforced_state["evicted_links"],
                "baseline_final_links": baseline_state["association_count"],
                "reinforced_final_links": reinforced_state["association_count"],
            },
        }

    suppression_rate = fmean(contradiction_suppressions)
    contradiction_recall = fmean(contradiction_hits)
    baseline_summary = {
        "target_recall_after_distractors@1": fmean(baseline_retention),
        "mean_evicted_links": fmean(baseline_evictions),
    }
    reinforced_summary = {
        "target_recall_after_distractors@1": fmean(reinforced_retention),
        "mean_evicted_links": fmean(reinforced_evictions),
    }

    checks = [
        check(
            "explicit contradictions suppress previously learned positive links",
            suppression_rate == 1.0,
            f"Suppression rate={suppression_rate:.3f}",
        ),
        check(
            "contradictory records are not retrieved through their removed positive link",
            contradiction_recall == 0.0,
            f"Conflict-pair associative recall={contradiction_recall:.3f}",
        ),
        check(
            "bounded baseline shows distractor-induced association interference",
            baseline_summary["target_recall_after_distractors@1"] == 0.0,
            f"Baseline target retention={baseline_summary['target_recall_after_distractors@1']:.3f}",
        ),
        check(
            "retrieval-dependent strengthening protects used links under the same distractors",
            reinforced_summary["target_recall_after_distractors@1"] == 1.0,
            f"Reinforced target retention={reinforced_summary['target_recall_after_distractors@1']:.3f}",
        ),
        check(
            "both networks remain within their one-link capacity",
            all(
                row["interference"]["baseline_final_links"] <= 1
                and row["interference"]["reinforced_final_links"] <= 1
                for row in by_probe.values()
            ),
            "All controlled graphs stayed within max_links=1",
        ),
    ]

    return SuiteResult(
        suite="associative-dynamics",
        title="Contradiction-sensitive associations and retrieval interference",
        evidence_category="synthetic software-mechanism benchmark",
        splits={"tuning": 0, "eval": PROBE_COUNT},
        systems={
            "no_retrieval_reinforcement": baseline_summary,
            "retrieval_reinforced": reinforced_summary,
            "contradiction_suppression": {
                "suppression_rate": suppression_rate,
                "contradiction_recall@1": contradiction_recall,
            },
        },
        by_condition=by_probe,
        checks=checks,
        findings=[
            (
                "Explicit repository contradiction links remove a previously learned "
                "positive edge and prevent that pair from being relearned in the observed cycle."
            ),
            (
                "Under an identical one-link capacity and distractor sequence, repeated "
                "successful retrieval protects the reinforced target association."
            ),
            (
                "These are project-authored opaque-ID controls. They test graph rules and "
                "capacity effects, not factual truth, source credibility, or biological memory."
            ),
        ],
        extra={
            "probe_count": PROBE_COUNT,
            "seed": seed,
            "max_links_per_network": 1,
            "retrieval_rate_adaptive": 0.2,
            "retrieval_rate_baseline": 0.0,
            "privacy": {
                "memory_content_used": False,
                "persistent_user_data_used": False,
                "network_storage": "volatile_process_memory",
            },
        },
    )
