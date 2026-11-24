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
    "apply_logistic",
    "auroc",
    "balanced_accuracy",
    "brier_score",
    "calibration_slope_intercept",
    "choose_threshold",
    "detection_rates",
    "expected_calibration_error",
    "fit_logistic",
    "reliability_bins",
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


# ------------------------------------------------------------- calibration


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


def fit_logistic(
    features: Sequence[float], labels: Sequence[int], *, ridge: float = 1e-3,
    iterations: int = 100,
) -> tuple[float, float] | None:
    """Fit ``P(y=1) = sigmoid(a + b·x)`` by ridge-regularised Newton–Raphson.

    Returns ``(a, b)``, or ``None`` when the fit does not converge. The small
    ridge keeps estimates finite for (near-)separable data; it shrinks rather
    than eliminates extreme slopes.
    """
    if len(features) != len(labels) or len(features) < 2:
        raise ValueError("logistic fit needs two equal-length series of at least two values")
    a = b = 0.0
    for _ in range(iterations):
        g_a = g_b = 0.0
        h_aa = h_ab = h_bb = 0.0
        for x, y in zip(features, labels, strict=True):
            p = _sigmoid(a + b * x)
            w = p * (1 - p)
            g_a += y - p
            g_b += (y - p) * x
            h_aa += w
            h_ab += w * x
            h_bb += w * x * x
        g_a -= ridge * a
        g_b -= ridge * b
        h_aa += ridge
        h_bb += ridge
        determinant = h_aa * h_bb - h_ab * h_ab
        if determinant <= 1e-12:
            return None
        step_a = (h_bb * g_a - h_ab * g_b) / determinant
        step_b = (h_aa * g_b - h_ab * g_a) / determinant
        a, b = a + step_a, b + step_b
        if not (math.isfinite(a) and math.isfinite(b)):
            return None
        if abs(step_a) < 1e-9 and abs(step_b) < 1e-9:
            return a, b
    return None


def apply_logistic(parameters: tuple[float, float], value: float) -> float:
    return _sigmoid(parameters[0] + parameters[1] * value)


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    if not probabilities or len(probabilities) != len(labels):
        raise ValueError("Brier score needs equal-length, non-empty inputs")
    return mean([(p - y) ** 2 for p, y in zip(probabilities, labels, strict=True)])


def reliability_bins(
    probabilities: Sequence[float], labels: Sequence[int], bins: int = 10
) -> list[dict[str, float | int | None]]:
    """Equal-width bins over [0, 1]: count, mean predicted probability, observed rate."""
    if bins < 1:
        raise ValueError("bins must be positive")
    grouped: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for p, y in zip(probabilities, labels, strict=True):
        if not 0.0 <= p <= 1.0:
            raise ValueError("probabilities must lie in [0, 1]")
        grouped[min(int(p * bins), bins - 1)].append((p, y))
    return [
        {
            "lower": round(index / bins, 4), "upper": round((index + 1) / bins, 4),
            "count": len(members),
            "mean_predicted": round(mean([p for p, _ in members]), 4) if members else None,
            "observed_rate": round(mean([float(y) for _, y in members]), 4) if members else None,
        }
        for index, members in enumerate(grouped)
    ]


def expected_calibration_error(
    probabilities: Sequence[float], labels: Sequence[int], bins: int = 10
) -> float:
    total = len(probabilities)
    if not total:
        raise ValueError("ECE needs at least one prediction")
    error = 0.0
    for item in reliability_bins(probabilities, labels, bins):
        count = int(item["count"] or 0)
        if count:
            error += count / total * abs(
                float(item["mean_predicted"] or 0.0) - float(item["observed_rate"] or 0.0)
            )
    return error


def calibration_slope_intercept(
    probabilities: Sequence[float], labels: Sequence[int]
) -> tuple[float, float] | None:
    """Cox calibration: regress outcomes on logit(p). Ideal slope 1, intercept 0.

    Returns ``(intercept, slope)``, or ``None`` when not estimable (a single
    class, or a fit that does not converge).
    """
    if len(set(labels)) < 2:
        return None
    clipped = [min(max(p, 1e-6), 1 - 1e-6) for p in probabilities]
    logits = [math.log(p / (1 - p)) for p in clipped]
    if len(set(round(value, 9) for value in logits)) < 2:
        return None
    return fit_logistic(logits, labels)
