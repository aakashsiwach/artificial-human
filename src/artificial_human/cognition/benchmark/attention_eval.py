"""Attention suite: selection among competing candidates.

The system under test is the unmodified ``AttentionSystem.select``. Baselines
rank by relevance only, salience only, or input order. Weight-sensitivity
variants re-apply the documented weighted formula with other weights. They
are ablations used for analysis, not alternative implementations.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from artificial_human.cognition.attention import AttentionCandidate, AttentionSystem
from artificial_human.cognition.benchmark.common import SuiteResult, rng
from artificial_human.cognition.benchmark.datasets import (
    ATTENTION_CONDITIONS,
    AttentionTask,
    generate_attention_tasks,
)
from artificial_human.cognition.benchmark.metrics import (
    mean,
    ndcg_at_k,
    pairwise_accuracy,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

Ranker = Callable[[Sequence[AttentionCandidate]], list[str]]


def _existing(candidates: Sequence[AttentionCandidate]) -> list[str]:
    selected = AttentionSystem().select(list(candidates), limit=len(candidates))
    return [item.reference_id for item in selected]


def _by(key: Callable[[AttentionCandidate], float]) -> Ranker:
    def rank(candidates: Sequence[AttentionCandidate]) -> list[str]:
        return [
            item.reference_id
            for item in sorted(candidates, key=lambda item: (-key(item), item.reference_id))
        ]

    return rank


def _input_order(candidates: Sequence[AttentionCandidate]) -> list[str]:
    return [item.reference_id for item in candidates]


def _weighted(relevance_weight: float) -> Ranker:
    return _by(
        lambda item: round(relevance_weight * item.relevance
                           + (1 - relevance_weight) * item.salience, 4)
    )


SYSTEMS: dict[str, Ranker] = {
    "existing_attention": _existing,
    "relevance_only": _by(lambda item: item.relevance),
    "salience_only": _by(lambda item: item.salience),
    "input_order": _input_order,
}
ABLATIONS: dict[str, Ranker] = {
    "weights_r0.6_s0.4": _weighted(0.6),
    "weights_r0.5_s0.5": _weighted(0.5),
}


def _score(task: AttentionTask, ranking: list[str]) -> dict[str, float]:
    relevant = {key: grade for key, grade in task.grades.items() if grade > 0}
    return {
        "precision@k": precision_at_k(ranking, relevant, task.k),
        "recall@k": recall_at_k(ranking, relevant, task.k),
        "ndcg@k": ndcg_at_k(ranking, relevant, task.k),
        "mrr": reciprocal_rank(ranking, relevant, len(ranking)),
        "pairwise_accuracy": pairwise_accuracy(ranking, task.grades),
    }


def _aggregate(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    keys = ("precision@k", "recall@k", "ndcg@k", "mrr", "pairwise_accuracy")
    return {key: round(mean([row[key] for row in rows]), 4) for key in keys} | {
        "tasks": len(rows)
    }


def _consistency(tasks: Sequence[AttentionTask], seed: int) -> dict[str, float]:
    """Repeat-run identity and invariance of the top-k set to input order."""
    identical = invariant = 0
    for task in tasks:
        first = _existing(task.candidates)
        identical += first == _existing(task.candidates)
        shuffled = list(task.candidates)
        rng(seed, "attention-permutation", task.id).shuffle(shuffled)
        invariant += set(first[: task.k]) == set(_existing(shuffled)[: task.k])
    return {
        "repeat_run_identical": round(identical / len(tasks), 4),
        "topk_permutation_invariance": round(invariant / len(tasks), 4),
    }


def _failure(task: AttentionTask, ranking: list[str]) -> dict[str, Any] | None:
    """A grade-0 candidate selected while a grade-2 candidate was left out."""
    chosen = ranking[: task.k]
    intruders = [item for item in chosen if task.grades[item] == 0]
    missed = [item for item in ranking[task.k:] if task.grades[item] == 2]
    if not intruders or not missed:
        return None
    by_id = {item.reference_id: item for item in task.candidates}
    describe = [
        {"id": key, "grade": task.grades[key], "kind": by_id[key].kind,
         "relevance": by_id[key].relevance, "salience": by_id[key].salience}
        for key in (intruders[0], missed[0])
    ]
    return {"task": task.id, "condition": task.condition, "k": task.k,
            "selected_irrelevant": describe[0], "excluded_highly_relevant": describe[1]}


TRAIN_CONDITIONS: tuple[str, ...] = ("clean_relevance", "salience_informative")
HELD_OUT_CONDITIONS: tuple[str, ...] = ("salient_distractors", "cross_source_normalisation")
WEIGHT_GRID: tuple[float, ...] = tuple(round(0.1 * step, 1) for step in range(10, -1, -1))


def _failure_categories(tasks: Sequence[AttentionTask], ranker: Ranker) -> dict[str, float]:
    """Mean count per task of each failure type in the selected top-k."""
    counts = {"salient_distractor_selected": 0, "weak_learning_example_selected": 0,
              "other_irrelevant_selected": 0, "highly_relevant_missed": 0}
    for task in tasks:
        ranking = ranker(task.candidates)
        chosen = set(ranking[: task.k])
        for candidate in task.candidates:
            grade = task.grades[candidate.reference_id]
            if candidate.reference_id in chosen and grade == 0:
                if candidate.salience >= 0.9:
                    counts["salient_distractor_selected"] += 1
                elif candidate.kind == "learning_example":
                    counts["weak_learning_example_selected"] += 1
                else:
                    counts["other_irrelevant_selected"] += 1
            elif candidate.reference_id not in chosen and grade == 2:
                counts["highly_relevant_missed"] += 1
    return {key: round(value / len(tasks), 4) if tasks else 0.0 for key, value in counts.items()}


def attention_generalisation(tasks: Sequence[AttentionTask]) -> dict[str, Any]:
    """Tune on two conditions (tuning split), test on two unseen conditions (eval split).

    Only the weight of the documented formula is tuned, and only on tuning-split
    tasks of the training conditions. Held-out conditions never influence it.
    """
    tune = [t for t in tasks if t.split == "tuning" and t.condition in TRAIN_CONDITIONS]
    in_distribution = [t for t in tasks if t.split == "eval" and t.condition in TRAIN_CONDITIONS]
    held_out = [t for t in tasks if t.split == "eval" and t.condition in HELD_OUT_CONDITIONS]

    def mean_ndcg(ranker: Ranker, subset: Sequence[AttentionTask]) -> float:
        return mean([_score(task, ranker(task.candidates))["ndcg@k"] for task in subset])

    scores = {weight: mean_ndcg(_weighted(weight), tune) for weight in WEIGHT_GRID}
    tuned_weight = max(WEIGHT_GRID, key=lambda weight: (round(scores[weight], 9), weight))
    systems: dict[str, Ranker] = {
        **SYSTEMS, f"tuned_weight_r{tuned_weight}": _weighted(tuned_weight),
    }
    report: dict[str, Any] = {
        "train_conditions": list(TRAIN_CONDITIONS),
        "held_out_conditions": list(HELD_OUT_CONDITIONS),
        "tuning_tasks": len(tune), "in_distribution_eval_tasks": len(in_distribution),
        "held_out_eval_tasks": len(held_out),
        "tuned_relevance_weight": tuned_weight,
        "tuning_ndcg_by_weight": {str(weight): round(value, 4) for weight, value in scores.items()},
        "systems": {},
    }
    for name, ranker in systems.items():
        in_score = mean_ndcg(ranker, in_distribution)
        by_condition = {
            condition: _aggregate([
                _score(task, ranker(task.candidates))
                for task in held_out if task.condition == condition
            ])
            for condition in HELD_OUT_CONDITIONS
        }
        held_score = mean([values["ndcg@k"] for values in by_condition.values()])
        report["systems"][name] = {
            "in_distribution_ndcg@k": round(in_score, 4),
            "held_out_ndcg@k": round(held_score, 4),
            "generalisation_gap": round(in_score - held_score, 4),
            "held_out_by_condition": by_condition,
            "held_out_failure_categories_per_task": _failure_categories(held_out, ranker),
        }
    return report


def run_attention_suite(seed: int) -> SuiteResult:
    tasks = generate_attention_tasks(seed)
    evaluation = [task for task in tasks if task.split == "eval"]
    result = SuiteResult(
        suite="attention",
        title="Attention: bounded selection among competing candidates",
        evidence_category="computational capability benchmark (synthetic inputs)",
        splits={"tuning": len(tasks) - len(evaluation), "eval": len(evaluation)},
    )
    rankings: dict[str, dict[str, list[str]]] = {}
    for name, ranker in {**SYSTEMS, **ABLATIONS}.items():
        rankings[name] = {task.id: ranker(task.candidates) for task in evaluation}
        rows = [_score(task, rankings[name][task.id]) for task in evaluation]
        target = result.systems if name in SYSTEMS else result.extra.setdefault("ablations", {})
        target[name] = _aggregate(rows)
        for condition in ATTENTION_CONDITIONS:
            condition_rows = [
                _score(task, rankings[name][task.id])
                for task in evaluation if task.condition == condition
            ]
            result.by_condition.setdefault(condition, {})[name] = _aggregate(condition_rows)

    sizes = sorted({len(task.candidates) for task in evaluation})
    result.extra["ndcg_by_set_size"] = {
        name: {
            str(size): round(mean([
                _score(task, rankings[name][task.id])["ndcg@k"]
                for task in evaluation if len(task.candidates) == size
            ]), 4)
            for size in sizes
        }
        for name in SYSTEMS
    }
    cross = [task for task in evaluation if task.condition == "cross_source_normalisation"]
    selected_examples = [
        sum(
            1 for key in rankings["existing_attention"][task.id][: task.k]
            if next(c for c in task.candidates if c.reference_id == key).kind == "learning_example"
        ) / task.k
        for task in cross
    ]
    result.extra["cross_source_selected_learning_example_share"] = round(
        mean(selected_examples), 4
    )
    result.extra["consistency"] = _consistency(evaluation, seed)
    failures = [
        failure for task in evaluation
        if (failure := _failure(task, rankings["existing_attention"][task.id])) is not None
    ]
    result.extra["failure_count"] = len(failures)
    result.extra["failure_count_by_condition"] = {
        condition: sum(1 for item in failures if item["condition"] == condition)
        for condition in ATTENTION_CONDITIONS
    }
    # Up to two examples per condition, so every failure mode is visible.
    result.failures = [
        item for condition in ATTENTION_CONDITIONS
        for item in [f for f in failures if f["condition"] == condition][:2]
    ]

    generalisation = attention_generalisation(tasks)
    result.extra["generalisation"] = generalisation
    existing, relevance = result.systems["existing_attention"], result.systems["relevance_only"]
    gen = generalisation["systems"]
    tuned_name = f"tuned_weight_r{generalisation['tuned_relevance_weight']}"
    result.findings = [
        "Generalisation (tune on clean + informative salience, test on salient distractors "
        f"+ cross-source): existing {gen['existing_attention']['in_distribution_ndcg@k']:.3f} → "
        f"{gen['existing_attention']['held_out_ndcg@k']:.3f}; relevance-only "
        f"{gen['relevance_only']['in_distribution_ndcg@k']:.3f} → "
        f"{gen['relevance_only']['held_out_ndcg@k']:.3f}; weight tuned in-distribution "
        f"(r={generalisation['tuned_relevance_weight']}) "
        f"{gen[tuned_name]['in_distribution_ndcg@k']:.3f} → {gen[tuned_name]['held_out_ndcg@k']:.3f}.",
        f"Overall nDCG@k: existing {existing['ndcg@k']:.3f}, relevance-only "
        f"{relevance['ndcg@k']:.3f}, salience-only {result.systems['salience_only']['ndcg@k']:.3f}, "
        f"input order {result.systems['input_order']['ndcg@k']:.3f}.",
    ] + [
        f"{condition}: existing nDCG@k {result.by_condition[condition]['existing_attention']['ndcg@k']:.3f} "
        f"vs relevance-only {result.by_condition[condition]['relevance_only']['ndcg@k']:.3f}."
        for condition in ATTENTION_CONDITIONS
    ] + [
        f"In the cross-source condition, "
        f"{result.extra['cross_source_selected_learning_example_share']:.1%} of selected "
        "slots went to grade-0 learning examples (25% of candidates), because each source "
        "is max-normalised separately before attention.",
        "Overall averages depend on the author-chosen mix of conditions (salience is "
        "informative in two of five). Read the per-condition results, not the overall "
        "average, to judge the formula.",
        f"{len(failures)} of {len(evaluation)} eval tasks selected an irrelevant candidate "
        "while excluding a highly relevant one.",
    ]
    return result
