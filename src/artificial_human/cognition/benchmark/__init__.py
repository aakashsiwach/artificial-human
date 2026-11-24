"""Reproducible evaluation of the current cognitive subsystems.

Suites: attention, working-memory, prediction, memory (wraps the long-term
memory benchmark) and integration (multi-turn scenarios through the cycle).
See ``docs/COGNITIVE_EVALUATION.md``.
"""

from artificial_human.cognition.benchmark.runner import (
    SUITES,
    InvariantViolation,
    deterministic_view,
    run_cognitive_evaluation,
    write_reports,
)

__all__ = [
    "SUITES",
    "InvariantViolation",
    "deterministic_view",
    "run_cognitive_evaluation",
    "write_reports",
]
