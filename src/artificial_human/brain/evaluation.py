from __future__ import annotations

from collections import Counter
from typing import Any

from artificial_human.brain.retrieval import tokenize


def _lcs_length(left: list[str], right: list[str]) -> int:
    """Compute LCS length with a rolling row to cap memory use."""
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def evaluate_text_similarity(
    reference: str,
    candidate: str,
    *,
    max_tokens: int = 1000,
) -> dict[str, Any]:
    """Return lexical similarity metrics, not a semantic or identity score."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    reference_tokens = tokenize(reference)
    candidate_tokens = tokenize(candidate)
    truncated = len(reference_tokens) > max_tokens or len(candidate_tokens) > max_tokens
    reference_tokens = reference_tokens[:max_tokens]
    candidate_tokens = candidate_tokens[:max_tokens]

    reference_counts = Counter(reference_tokens)
    candidate_counts = Counter(candidate_tokens)
    overlap = sum((reference_counts & candidate_counts).values())
    precision = overlap / len(candidate_tokens) if candidate_tokens else 0.0
    recall = overlap / len(reference_tokens) if reference_tokens else 0.0
    token_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    lcs = _lcs_length(reference_tokens, candidate_tokens)
    rouge_precision = lcs / len(candidate_tokens) if candidate_tokens else 0.0
    rouge_recall = lcs / len(reference_tokens) if reference_tokens else 0.0
    rouge_f1 = (
        2 * rouge_precision * rouge_recall / (rouge_precision + rouge_recall)
        if rouge_precision + rouge_recall
        else 0.0
    )
    normalized_reference = " ".join(reference_tokens)
    normalized_candidate = " ".join(candidate_tokens)

    return {
        "exact_match": bool(reference_tokens) and normalized_reference == normalized_candidate,
        "token_precision": round(precision, 4),
        "token_recall": round(recall, 4),
        "token_f1": round(token_f1, 4),
        "rouge_l_precision": round(rouge_precision, 4),
        "rouge_l_recall": round(rouge_recall, 4),
        "rouge_l_f1": round(rouge_f1, 4),
        "reference_token_count": len(reference_tokens),
        "candidate_token_count": len(candidate_tokens),
        "truncated": truncated,
        "metric_kind": "lexical_similarity",
    }
