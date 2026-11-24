"""Shared types and helpers for the cognitive evaluation suites.

Each suite separates four steps: task generation (``datasets``), execution
against the real implementation, scoring (``metrics``), and reporting
(``report``). Suites never call the network and never use personal data.
"""

from __future__ import annotations

import hashlib
import random
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

Split = Literal["tuning", "eval"]
SPLITS: tuple[Split, ...] = ("tuning", "eval")
DEFAULT_SEED = 20261124


def derive_seed(base: int, *parts: str) -> int:
    """Stable, platform-independent sub-seed (Python's ``hash`` is salted)."""
    digest = hashlib.sha256(":".join([str(base), *parts]).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def rng(base: int, *parts: str) -> random.Random:
    return random.Random(derive_seed(base, *parts))


@dataclass
class SuiteResult:
    """Outcome of one evaluation suite (JSON-serialisable through ``as_dict``)."""

    suite: str
    title: str
    evidence_category: str
    splits: dict[str, int]
    systems: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_condition: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    peak_memory_kib: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "title": self.title,
            "evidence_category": self.evidence_category,
            "splits": self.splits,
            "systems": self.systems,
            "by_condition": self.by_condition,
            "checks": self.checks,
            "failures": self.failures,
            "findings": self.findings,
            "extra": self.extra,
            "runtime_seconds": round(self.runtime_seconds, 4),
            "peak_memory_kib": round(self.peak_memory_kib, 1),
        }


def measured(run: Callable[[], SuiteResult]) -> SuiteResult:
    """Run a suite while recording wall-clock time and peak Python allocation."""
    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    started = time.perf_counter()
    try:
        result = run()
    finally:
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        if not already_tracing:
            tracemalloc.stop()
    result.runtime_seconds = elapsed
    result.peak_memory_kib = peak / 1024
    return result


def check(name: str, passed: bool, detail: str = "", **values: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail, **values}
