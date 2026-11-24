"""Prediction/novelty suite.

What ``PredictionSystem`` measures today: ``novelty = 1 − |T(input) ∩ T(context)| /
|T(input)|``, where ``T`` is the lowercase token set (tokens of two or more
alphanumeric characters, no stop-word removal). It does not model the world,
forecast future events, or produce a biological prediction-error signal.

Ground truth is the author-assigned ``unexpected`` label of each item (see
``datasets.PredictionItem``). No language model is used as a judge.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from artificial_human.brain.retrieval import tokenize
from artificial_human.cognition.benchmark.common import SuiteResult
from artificial_human.cognition.benchmark.datasets import PredictionItem, prediction_items
from artificial_human.cognition.benchmark.metrics import (
    apply_logistic,
    auroc,
    brier_score,
    calibration_slope_intercept,
    choose_threshold,
    detection_rates,
    expected_calibration_error,
    fit_logistic,
    mean,
    pearson,
    reliability_bins,
)
from artificial_human.cognition.perception import PerceptionSystem
from artificial_human.cognition.prediction import PredictionSystem
from artificial_human.memory.embeddings import EmbeddingProvider, cosine
from artificial_human.memory.search import lexical_terms

Scorer = Callable[[PredictionItem], float]


def existing_novelty(item: PredictionItem) -> float:
    percept = PerceptionSystem().perceive(item.text)
    return PredictionSystem().assess(percept, list(item.context)).novelty


def content_word_novelty(item: PredictionItem) -> float:
    """Token-overlap baseline on content words (stop words removed, light stemming)."""
    current = set(lexical_terms(item.text))
    context = {term for sentence in item.context for term in lexical_terms(sentence)}
    if not current:
        return 0.0
    return 1.0 - len(current & context) / len(current)


def jaccard_distance(item: PredictionItem) -> float:
    current = set(tokenize(item.text))
    context = {token for sentence in item.context for token in tokenize(sentence)}
    union = current | context
    return 1.0 - len(current & context) / len(union) if union else 0.0


def embedding_novelty(provider: EmbeddingProvider) -> Scorer:
    """Optional reference: 1 − max cosine to any context sentence (local model only)."""
    def score(item: PredictionItem) -> float:
        vectors = provider.embed_documents([item.text, *item.context])
        return 1.0 - max(cosine(vectors[0], vector) for vector in vectors[1:])

    return score


def _system_report(
    scorer: Scorer, tuning: Sequence[PredictionItem], evaluation: Sequence[PredictionItem]
) -> tuple[dict[str, Any], dict[str, float]]:
    tuning_scores = [scorer(item) for item in tuning]
    threshold = choose_threshold(tuning_scores, [item.unexpected for item in tuning])
    scores = {item.id: round(scorer(item), 6) for item in evaluation}
    values = [scores[item.id] for item in evaluation]
    labels = [item.unexpected for item in evaluation]
    rates = detection_rates(values, labels, threshold)
    positives = [scores[item.id] for item in evaluation if item.unexpected]
    negatives = [scores[item.id] for item in evaluation if not item.unexpected]
    correlation = pearson(values, [jaccard_distance(item) for item in evaluation])
    report = {
        "auroc": round(auroc(values, labels), 4),
        "threshold_from_tuning_split": threshold,
        "false_novelty_rate": round(rates["false_positive_rate"], 4),
        "missed_novelty_rate": round(rates["false_negative_rate"], 4),
        "balanced_accuracy": round(rates["balanced_accuracy"], 4),
        "mean_novelty_unexpected": round(mean(positives), 4),
        "mean_novelty_expected": round(mean(negatives), 4),
        "novelty_separation": round(mean(positives) - mean(negatives), 4),
        "correlation_with_lexical_jaccard_distance": (
            None if correlation is None else round(correlation, 4)
        ),
        "repeat_run_identical": [scorer(item) for item in evaluation] == [
            scorer(item) for item in evaluation
        ],
    }
    return report, scores


def _probability_report(probabilities: Sequence[float], labels: Sequence[int]) -> dict[str, Any]:
    rates = detection_rates(probabilities, labels, 0.5)
    slope = calibration_slope_intercept(probabilities, labels)
    return {
        "brier_score": round(brier_score(probabilities, labels), 4),
        "expected_calibration_error": round(expected_calibration_error(probabilities, labels), 4),
        "calibration_intercept": None if slope is None else round(slope[0], 4),
        "calibration_slope": None if slope is None else round(slope[1], 4),
        "auroc": round(auroc(probabilities, labels), 4),
        "false_novelty_rate_at_0.5": round(rates["false_positive_rate"], 4),
        "missed_novelty_rate_at_0.5": round(rates["false_negative_rate"], 4),
        "reliability_bins": reliability_bins(probabilities, labels, bins=5),
    }


def calibration_report(
    scorer: Scorer, tuning: Sequence[PredictionItem], evaluation: Sequence[PredictionItem]
) -> dict[str, Any]:
    """Compare the raw score (read as a probability) with a frozen logistic mapping.

    The logistic mapping is fitted on the tuning split only and then applied,
    unchanged, to the eval split. Calibration changes how well score values
    match observed frequencies; a monotone mapping cannot change the ranking
    (AUROC) and is not evidence of better prediction.
    """
    tuning_scores = [scorer(item) for item in tuning]
    labels = [item.unexpected for item in evaluation]
    scores = [scorer(item) for item in evaluation]
    uncalibrated = [min(max(score, 0.0), 1.0) for score in scores]
    parameters = fit_logistic(tuning_scores, [item.unexpected for item in tuning])
    report: dict[str, Any] = {
        "uncalibrated_score_as_probability": _probability_report(uncalibrated, labels),
        "logistic_mapping_fitted_on": "tuning",
        "logistic_parameters": None if parameters is None else {
            "intercept": round(parameters[0], 4), "slope": round(parameters[1], 4),
        },
    }
    if parameters is not None:
        calibrated = [apply_logistic(parameters, score) for score in scores]
        report["logistic_calibrated"] = _probability_report(calibrated, labels)
    prevalence = mean([float(item.unexpected) for item in tuning])
    report["base_rate_reference"] = {
        "probability": round(prevalence, 4),
        "brier_score": round(brier_score([prevalence] * len(labels), labels), 4),
        "note": "Constant tuning-split prevalence; a score carrying no information.",
    }
    return report


def run_prediction_suite(provider: EmbeddingProvider | None = None) -> SuiteResult:
    items = prediction_items()
    tuning = [item for item in items if item.split == "tuning"]
    evaluation = [item for item in items if item.split == "eval"]
    result = SuiteResult(
        suite="prediction",
        title="Prediction/novelty: lexical novelty against active context",
        evidence_category="computational capability benchmark (authored labels)",
        splits={"tuning": len(tuning), "eval": len(evaluation)},
    )
    scorers: dict[str, Scorer] = {
        "existing_prediction": existing_novelty,
        "content_word_overlap": content_word_novelty,
        "jaccard_distance": jaccard_distance,
    }
    if provider is not None:
        scorers["embedding_distance_reference"] = embedding_novelty(provider)
    all_scores: dict[str, dict[str, float]] = {}
    for name, scorer in scorers.items():
        result.systems[name], all_scores[name] = _system_report(scorer, tuning, evaluation)

    categories = sorted({item.category for item in evaluation})
    for category in categories:
        members = [item for item in evaluation if item.category == category]
        result.by_condition[category] = {
            name: {
                "label": "unexpected" if members[0].unexpected else "expected",
                "mean_novelty": round(mean([all_scores[name][item.id] for item in members]), 4),
            }
            for name in scorers
        }
    result.extra["calibration"] = {
        name: calibration_report(scorer, tuning, evaluation) for name, scorer in scorers.items()
    }
    result.extra["calibration_note"] = (
        "Novelty is a score, not a probability. Calibration maps scores to observed "
        "frequencies of 'unexpected' labels; it does not make the score a better predictor, "
        "and it is unrelated to biological prediction error. With 18 tuning items the "
        "fitted mapping is crude."
    )
    perception = PerceptionSystem()
    predictor = PredictionSystem()
    perturbed_equal = sum(
        predictor.assess(perception.perceive(item.text.upper() + " !!"), list(item.context)).novelty
        == all_scores["existing_prediction"][item.id]
        for item in evaluation
    )
    empty_context = predictor.assess(perception.perceive("Anything at all."), [])
    result.extra["case_and_punctuation_invariance"] = round(perturbed_equal / len(evaluation), 4)
    result.extra["empty_context_comparable"] = empty_context.comparable
    result.extra["empty_context_novelty"] = empty_context.novelty

    threshold = result.systems["existing_prediction"]["threshold_from_tuning_split"]
    for item in evaluation:
        score = all_scores["existing_prediction"][item.id]
        predicted = int(score >= threshold)
        if predicted != item.unexpected:
            result.failures.append({
                "item": item.id, "category": item.category,
                "error": "false novelty" if predicted else "missed novelty",
                "novelty": score, "threshold": threshold,
                "context": " ".join(item.context), "input": item.text,
            })
    existing = result.systems["existing_prediction"]
    by_category = result.by_condition
    result.findings = [
        f"Existing novelty AUROC on the eval split is {existing['auroc']:.3f}. False novelty "
        f"rate {existing['false_novelty_rate']:.3f}, missed novelty rate "
        f"{existing['missed_novelty_rate']:.3f} at the tuning-split threshold {threshold}.",
        "Contradictions look familiar: mean novelty "
        f"{by_category['contradiction']['existing_prediction']['mean_novelty']:.3f}, against "
        f"{by_category['unrelated']['existing_prediction']['mean_novelty']:.3f} for unrelated "
        "input. Negation or a changed value barely changes token overlap.",
        "Paraphrases and semantically related inputs with little word overlap look novel "
        f"({by_category['low_overlap_related']['existing_prediction']['mean_novelty']:.3f}), "
        "because the signal is lexical.",
        "Script violations score lower novelty than typical continuations "
        f"({by_category['contextually_unexpected']['existing_prediction']['mean_novelty']:.3f} vs "
        f"{by_category['contextually_expected']['existing_prediction']['mean_novelty']:.3f}), the "
        "opposite of the intended signal, because violating sentences reuse context words.",
        "The score is a deterministic function of token overlap (correlation with lexical "
        "Jaccard distance "
        f"{existing['correlation_with_lexical_jaccard_distance']}). It does not represent "
        "meaning, expectation, or future events; its above-chance separation comes from "
        "surface word differences.",
    ]
    if "embedding_distance_reference" in result.systems:
        reference = result.systems["embedding_distance_reference"]
        result.findings.append(
            f"The optional local-embedding reference reaches AUROC {reference['auroc']:.3f} "
            "but also scores contradictions as familiar "
            f"({by_category['contradiction']['embedding_distance_reference']['mean_novelty']:.3f}). "
            "Semantic similarity is not contradiction detection."
        )
    return result
