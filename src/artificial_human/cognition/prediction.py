from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from artificial_human.brain.retrieval import tokenize
from artificial_human.cognition.perception import Percept


@dataclass(frozen=True, slots=True)
class PredictionAssessment:
    comparable: bool
    novelty: float
    overlapping_terms: tuple[str, ...]
    context_term_count: int


class PredictionSystem:
    """Estimates textual novelty against available context.

    This is a transparent overlap heuristic used as a prediction-error signal.
    It is not biological predictive coding and does not infer hidden beliefs.
    """

    def assess(
        self,
        percept: Percept,
        expected_context: Sequence[str],
    ) -> PredictionAssessment:
        expected_terms = {
            token
            for item in expected_context
            for token in tokenize(item)
        }
        current_terms = set(percept.tokens)
        if not current_terms or not expected_terms:
            return PredictionAssessment(
                comparable=False,
                novelty=0.0,
                overlapping_terms=(),
                context_term_count=len(expected_terms),
            )
        overlap = tuple(sorted(current_terms & expected_terms))
        novelty = 1.0 - (len(overlap) / len(current_terms))
        return PredictionAssessment(
            comparable=True,
            novelty=round(max(0.0, min(1.0, novelty)), 4),
            overlapping_terms=overlap,
            context_term_count=len(expected_terms),
        )
