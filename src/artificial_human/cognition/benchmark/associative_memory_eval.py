"""Controlled evaluation of associative recall against simple retrieval controls.

The probe set is deliberately small and authored from source-defined facts. It
tests whether a co-activation edge can complete a cue when keyword retrieval
returns only the cue fact. It is not a measure of naturalistic or human memory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from statistics import fmean
from typing import Any, TypedDict

from artificial_human.cognition.associative_memory import AssociativeKnowledgeNetwork
from artificial_human.cognition.benchmark.common import (
    DEFAULT_SEED,
    SuiteResult,
    check,
    rng,
)
from artificial_human.cognition.core_knowledge import CORE_KNOWLEDGE, retrieve_core_knowledge

TOP_K = 5
RANDOM_LINK_REPETITIONS = 512

QueryMetrics = TypedDict(
    "QueryMetrics",
    {
        "target_rank": int | None,
        "target_recall@5": float,
        "target_mrr": float,
        "false_associations": float | int,
    },
)


@dataclass(frozen=True, slots=True)
class AssociativeProbe:
    """One controlled co-activation and held-out single-fact cue."""

    name: str
    source_fact_id: str
    target_fact_id: str
    training_query: str
    cue_query: str


# Each training query must retrieve exactly the source and target facts; each cue
# must retrieve only the source. This makes the target a relational completion.
PROBES: tuple[AssociativeProbe, ...] = (
    AssociativeProbe(
        "purpose_to_family_roles", "project.purpose", "identity.family_roles",
        "purpose mother", "purpose",
    ),
    AssociativeProbe(
        "founder_to_public_portfolio", "identity.founder", "founder.public_portfolio",
        "linkedin portfolio", "linkedin",
    ),
    AssociativeProbe(
        "architecture_to_language_model", "architecture.cognitive_cycle",
        "language.model_role", "executive groq", "executive",
    ),
    AssociativeProbe(
        "perception_to_unimplemented_senses", "perception.current_limit",
        "not_implemented.multimodal", "microphone emotion", "microphone",
    ),
    AssociativeProbe(
        "working_memory_to_volatile_deployment", "working_memory.runtime",
        "deployment.volatile", "session render", "session",
    ),
    AssociativeProbe(
        "long_term_memory_to_volatile_deployment", "long_term_memory.governance",
        "deployment.volatile", "consent server", "consent",
    ),
    AssociativeProbe(
        "prenatal_to_unimplemented_capabilities", "prenatal.scaffold",
        "not_implemented.multimodal", "prenatal spiking", "prenatal",
    ),
    AssociativeProbe(
        "learning_to_language_model", "learning.current_limit",
        "language.model_role", "fine-tune groq", "fine-tune",
    ),
    AssociativeProbe(
        "scientific_boundaries_to_purpose", "scientific.boundaries",
        "project.purpose", "consciousness goal", "consciousness",
    ),
)


def _link_key(first: str, second: str) -> tuple[str, str]:
    return (first, second) if first <= second else (second, first)


def _rank_with_links(
    query: str,
    links: Mapping[tuple[str, str], float],
    *,
    limit: int = TOP_K,
) -> tuple[str, ...]:
    """Apply the production network's direct + one-hop activation rule to a graph."""
    direct_facts = retrieve_core_knowledge(query, limit=limit)
    direct = {
        fact.fact_id: max(0.25, 1.0 - (position * 0.12))
        for position, fact in enumerate(direct_facts)
    }
    spread: dict[str, float] = {}
    for (first, second), strength in links.items():
        if first in direct and second not in direct:
            spread[second] = max(spread.get(second, 0.0), direct[first] * strength)
        elif second in direct and first not in direct:
            spread[first] = max(spread.get(first, 0.0), direct[second] * strength)

    activations = {
        fact_id: activation + 0.55 * spread.get(fact_id, 0.0)
        for fact_id, activation in direct.items()
    }
    for fact_id, activation in spread.items():
        activations.setdefault(fact_id, 0.55 * activation)
    return tuple(sorted(activations, key=lambda item: (-activations[item], item))[:limit])


def _query_metrics(
    ranked_ids: tuple[str, ...], probe: AssociativeProbe
) -> QueryMetrics:
    try:
        target_rank: int | None = ranked_ids.index(probe.target_fact_id) + 1
    except ValueError:
        target_rank = None
    false_associations = sum(
        fact_id not in {probe.source_fact_id, probe.target_fact_id}
        for fact_id in ranked_ids[:TOP_K]
    )
    return {
        "target_rank": target_rank,
        "target_recall@5": float(target_rank is not None and target_rank <= TOP_K),
        "target_mrr": 1.0 / target_rank if target_rank is not None else 0.0,
        "false_associations": false_associations,
    }


def _report_metrics(row: QueryMetrics) -> dict[str, Any]:
    """Convert typed internal metrics to the shared JSON-ready report shape."""
    return dict(row)


def _aggregate(rows: Sequence[QueryMetrics]) -> dict[str, float]:
    return {
        "target_recall@5": fmean(float(row["target_recall@5"]) for row in rows),
        "target_mrr": fmean(float(row["target_mrr"]) for row in rows),
        "false_associations_per_query": fmean(
            float(row["false_associations"]) for row in rows
        ),
    }


def run_associative_memory_suite(seed: int = DEFAULT_SEED) -> SuiteResult:
    """Compare learned associations, direct keyword retrieval, and random links."""
    fact_ids = tuple(sorted(fact.fact_id for fact in CORE_KNOWLEDGE))
    candidates = tuple(combinations(fact_ids, 2))
    network = AssociativeKnowledgeNetwork()

    # Validate that the authored probes isolate one source fact at test time and
    # exactly two facts during training. The benchmark must fail closed otherwise.
    for probe in PROBES:
        training_ids = {
            fact.fact_id for fact in retrieve_core_knowledge(probe.training_query, limit=TOP_K)
        }
        cue_ids = tuple(
            fact.fact_id for fact in retrieve_core_knowledge(probe.cue_query, limit=TOP_K)
        )
        if training_ids != {probe.source_fact_id, probe.target_fact_id}:
            raise ValueError(f"Invalid associative training probe: {probe.name}")
        if cue_ids != (probe.source_fact_id,):
            raise ValueError(f"Invalid associative cue probe: {probe.name}")
        network.process(probe.training_query, limit=TOP_K)

    trained_graph = network.snapshot()
    raw_edge_count = trained_graph["association_count"]
    raw_mean_strength = trained_graph["mean_association_strength"]
    if isinstance(raw_edge_count, bool) or not isinstance(raw_edge_count, int):
        raise ValueError("The associative graph edge count is not an integer")
    if isinstance(raw_mean_strength, bool) or not isinstance(raw_mean_strength, (int, float)):
        raise ValueError("The associative graph mean strength is not numeric")
    edge_count = raw_edge_count
    mean_strength = float(raw_mean_strength)
    if edge_count < 1 or edge_count > len(candidates):
        raise ValueError("The associative training graph has an invalid edge count")

    keyword_rows: list[QueryMetrics] = []
    learned_rows: list[QueryMetrics] = []
    by_probe: dict[str, dict[str, dict[str, Any]]] = {}
    for probe in PROBES:
        keyword_ids = tuple(
            fact.fact_id for fact in retrieve_core_knowledge(probe.cue_query, limit=TOP_K)
        )
        learned_ids = tuple(
            item.fact.fact_id for item in network.process(probe.cue_query, limit=TOP_K)
        )
        keyword = _query_metrics(keyword_ids, probe)
        learned = _query_metrics(learned_ids, probe)
        keyword_rows.append(keyword)
        learned_rows.append(learned)
        by_probe[probe.name] = {
            "keyword_only": _report_metrics(keyword),
            "learned_associations": _report_metrics(learned),
        }

    # Random-link control: preserve the trained graph's edge budget and mean
    # strength, but assign edges uniformly over all possible fact pairs. Use a
    # separate stable RNG stream per repetition; no global random state is touched.
    random_rows_by_probe: dict[str, list[QueryMetrics]] = {
        probe.name: [] for probe in PROBES
    }
    for repetition in range(RANDOM_LINK_REPETITIONS):
        generator = rng(seed, "associative-memory", "random-link", str(repetition))
        selected = generator.sample(candidates, k=edge_count)
        random_links = {pair: mean_strength for pair in selected}
        for probe in PROBES:
            random_links = {
                pair: strength * network.decay
                for pair, strength in random_links.items()
                if strength * network.decay >= 0.001
            }
            ranked_ids = _rank_with_links(probe.cue_query, random_links)
            random_rows_by_probe[probe.name].append(_query_metrics(ranked_ids, probe))

    random_rows: list[QueryMetrics] = []
    for probe in PROBES:
        repetitions = random_rows_by_probe[probe.name]
        average: QueryMetrics = {
            "target_rank": None,
            "target_recall@5": fmean(row["target_recall@5"] for row in repetitions),
            "target_mrr": fmean(row["target_mrr"] for row in repetitions),
            "false_associations": fmean(row["false_associations"] for row in repetitions),
        }
        random_rows.append(average)
        by_probe[probe.name]["random_link"] = _report_metrics(average)

    keyword_summary = _aggregate(keyword_rows)
    learned_summary = _aggregate(learned_rows)
    random_summary = _aggregate(random_rows)
    final_snapshot = network.snapshot()

    checks = [
        check(
            "all training probes co-activate exactly their declared source and target",
            True,
            f"{len(PROBES)} fixed source-fact pairs passed their preconditions",
        ),
        check(
            "all cue queries retrieve only their declared source by keyword",
            all(row["target_recall@5"] == 0.0 for row in keyword_rows),
            "Targets are absent from direct keyword results before association learning",
        ),
        check(
            "learned associative recall returns all target facts in top 5",
            learned_summary["target_recall@5"] == 1.0,
            f"Recall@5={learned_summary['target_recall@5']:.3f}",
        ),
        check(
            "learned associations outperform keyword-only and random-link controls",
            learned_summary["target_mrr"] > keyword_summary["target_mrr"]
            and learned_summary["target_mrr"] > random_summary["target_mrr"],
            (
                f"MRR learned={learned_summary['target_mrr']:.3f}, "
                f"keyword={keyword_summary['target_mrr']:.3f}, "
                f"random={random_summary['target_mrr']:.3f}"
            ),
        ),
        check(
            "random-link control matches the trained graph edge budget",
            edge_count <= len(candidates),
            f"{edge_count} edges sampled from {len(candidates)} possible fact pairs",
        ),
        check(
            "associative network remains bounded and does not retain user text",
            isinstance(final_snapshot["association_count"], int)
            and final_snapshot["association_count"] <= network.max_links
            and final_snapshot["retains_user_text"] is False,
            "Only source-fact identifiers and weights are held in volatile process memory",
        ),
    ]

    return SuiteResult(
        suite="associative-memory",
        title="Associative memory: controlled cue completion",
        evidence_category="synthetic_software_benchmark",
        splits={"tuning": 0, "eval": len(PROBES)},
        systems={
            "keyword_only": keyword_summary,
            "learned_associations": learned_summary,
            "random_link": random_summary,
        },
        by_condition=by_probe,
        checks=checks,
        findings=[
            (
                "The learned graph is evaluated on single-fact cues after paired "
                "co-activation training; keyword retrieval alone does not return the targets."
            ),
            (
                f"Learned MRR is {learned_summary['target_mrr']:.3f}; random-link MRR is "
                f"{random_summary['target_mrr']:.3f} over {RANDOM_LINK_REPETITIONS} seeded graphs."
            ),
            (
                "This is a hand-authored source-knowledge probe, not open-ended episodic recall "
                "or evidence of biological hippocampal function."
            ),
        ],
        extra={
            "probe_count": len(PROBES),
            "top_k": TOP_K,
            "random_link_repetitions": RANDOM_LINK_REPETITIONS,
            "training_graph": {
                "source_fact_count": trained_graph["source_fact_count"],
                "edge_count": edge_count,
                "mean_link_strength": mean_strength,
                "max_link_strength": trained_graph["max_association_strength"],
            },
            "random_link_control": {
                "repetitions": RANDOM_LINK_REPETITIONS,
                "edge_count": edge_count,
                "mean_link_strength": mean_strength,
                "sampling": "uniform_without_replacement_over_undirected_fact_pairs",
                "decay_per_query": network.decay,
                "seed": seed,
            },
            "privacy": {
                "state_storage": final_snapshot["state_storage"],
                "retains_user_text": final_snapshot["retains_user_text"],
            },
        },
    )
