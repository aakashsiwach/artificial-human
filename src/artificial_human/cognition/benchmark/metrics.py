"""Evaluation metrics for the cognitive suites.

Ranked-retrieval metrics are reused from the memory benchmark (single source
of truth). This module adds ranking-order, detection and agreement metrics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from artificial_human.memory.benchmark.metrics import (
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "auroc",
    "balanced_accuracy",
    "choose_threshold",
    "detection_rates",
    "kendall_tau",
    "mean",
    "ndcg_at_k",
    "pairwise_accuracy",
    "pearson",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]


def pairwise_accuracy(ranking: Sequence[str], grades: Mapping[str, int]) -> float:
    """Share of item pairs with different grades that the ranking orders correctly.

    Every graded item must appear in ``ranking``. Returns 1.0 when no pair has
    different grades (nothing can be ordered wrongly).
    """
    position = {item: index for index, item in enumerate(ranking)}
    missing = set(grades) - set(position)
    if missing:
        raise ValueError(f"ranking is missing graded items: {sorted(missing)[:3]}")
    items = sorted(grades)
    correct = total = 0
    for i, first in enumerate(items):
        for second in items[i + 1:]:
            if grades[first] == grades[second]:
                continue
            total += 1
            better, worse = (first, second) if grades[first] > grades[second] else (second, first)
            correct += position[better] < position[worse]
    return correct / total if total else 1.0


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the ROC curve (Mann–Whitney U); ties count one half."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    positives = [score for score, label in zip(scores, labels, strict=True) if label == 1]
    negatives = [score for score, label in zip(scores, labels, strict=True) if label == 0]
    if not positives or not negatives:
        raise ValueError("AUROC needs at least one positive and one negative example")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("labels must be 0 or 1")
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positives) * len(negatives))


def detection_rates(
    scores: Sequence[float], labels: Sequence[int], threshold: float
) -> dict[str, float]:
    """Rates for the rule ``score >= threshold`` ⇒ predicted positive (novel)."""
    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels, strict=True):
        predicted = score >= threshold
        if predicted and label == 1:
            tp += 1
        elif predicted:
            fp += 1
        elif label == 1:
            fn += 1
        else:
            tn += 1
    negatives, positives = fp + tn, tp + fn
    return {
        "false_positive_rate": fp / negatives if negatives else 0.0,
        "false_negative_rate": fn / positives if positives else 0.0,
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
        "balanced_accuracy": balanced_accuracy(tp, fp, tn, fn),
    }


def balanced_accuracy(tp: int, fp: int, tn: int, fn: int) -> float:
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return (sensitivity + specificity) / 2


def choose_threshold(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Threshold maximising balanced accuracy; ties keep the lowest threshold.

    Candidates are the midpoints between adjacent distinct scores plus both
    extremes. Use only on a tuning split.
    """
    if not scores:
        raise ValueError("cannot choose a threshold without scores")
    distinct = sorted(set(scores))
    candidates = [distinct[0]] + [
        (low + high) / 2 for low, high in zip(distinct, distinct[1:], strict=False)
    ] + [distinct[-1] + 1e-9]
    best_threshold, best_value = candidates[0], -1.0
    for threshold in candidates:
        value = detection_rates(scores, labels, threshold)["balanced_accuracy"]
        if value > best_value + 1e-12:
            best_threshold, best_value = threshold, value
    return round(best_threshold, 6)


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Pearson correlation; ``None`` when either series is constant."""
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("pearson needs two equal-length series of at least two values")
    mean_left, mean_right = mean(left), mean(right)
    cov = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True))
    var_left = sum((a - mean_left) ** 2 for a in left)
    var_right = sum((b - mean_right) ** 2 for b in right)
    if var_left == 0 or var_right == 0:
        return None
    return cov / math.sqrt(var_left * var_right)


def kendall_tau(observed: Sequence[str], reference: Sequence[str]) -> float:
    """Kendall's tau-a between two orderings of their common items."""
    reference_position = {item: index for index, item in enumerate(reference)}
    common = [item for item in observed if item in reference_position]
    n = len(common)
    if n < 2:
        return 1.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if reference_position[common[i]] < reference_position[common[j]]:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / (n * (n - 1) / 2)
