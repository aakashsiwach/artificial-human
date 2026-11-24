"""Run selected cognitive evaluation suites and assemble one report."""

from __future__ import annotations

import json
import platform
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from artificial_human.cognition.benchmark.attention_eval import run_attention_suite
from artificial_human.cognition.benchmark.common import DEFAULT_SEED, SuiteResult, measured
from artificial_human.cognition.benchmark.integration_eval import run_integration_suite
from artificial_human.cognition.benchmark.memory_eval import run_memory_suite
from artificial_human.cognition.benchmark.prediction_eval import run_prediction_suite
from artificial_human.cognition.benchmark.working_memory_eval import run_working_memory_suite
from artificial_human.memory.embeddings import EmbeddingProvider

SUITES: tuple[str, ...] = ("attention", "working-memory", "prediction", "memory", "integration")
SCHEMA_VERSION = "1.0"


class InvariantViolation(RuntimeError):
    """A safety invariant (for example owner isolation) failed during evaluation."""


def _suite_runner(
    name: str, seed: int, provider: EmbeddingProvider | None
) -> Callable[[], SuiteResult]:
    runners: dict[str, Callable[[], SuiteResult]] = {
        "attention": lambda: run_attention_suite(seed),
        "working-memory": lambda: run_working_memory_suite(seed),
        "prediction": lambda: run_prediction_suite(provider),
        "memory": lambda: run_memory_suite(provider),
        "integration": lambda: run_integration_suite(provider),
    }
    if name not in runners:
        raise ValueError(f"Unknown suite {name!r}; choose from {', '.join(SUITES)} or all")
    return runners[name]


def resolve_suites(requested: Sequence[str]) -> list[str]:
    if not requested or "all" in requested:
        return list(SUITES)
    unknown = [name for name in requested if name not in SUITES]
    if unknown:
        raise ValueError(f"Unknown suite(s): {', '.join(unknown)}")
    return [name for name in SUITES if name in requested]


def run_cognitive_evaluation(
    suites: Sequence[str] = ("all",),
    *,
    seed: int = DEFAULT_SEED,
    provider: EmbeddingProvider | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    notify = progress or (lambda _message: None)
    selected = resolve_suites(suites)
    model = provider.describe().public() if provider is not None else None
    started = time.perf_counter()
    results: dict[str, dict[str, Any]] = {}
    for name in selected:
        notify(f"running {name}")
        results[name] = measured(_suite_runner(name, seed, provider)).as_dict()
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "suites_run": selected,
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(terse=True),
            "byteorder": sys.byteorder,
        },
        "embedding_model": model,
        "network_used_by_evaluation": False,
        "total_runtime_seconds": round(time.perf_counter() - started, 3),
        "suites": results,
    }
    leaks = [
        item for suite in results.values() for item in suite["checks"]
        if "leak" in item["name"] and not item["passed"]
    ]
    if leaks:
        raise InvariantViolation(f"Owner-isolation invariant failed: {leaks}")
    return report


def deterministic_view(report: dict[str, Any]) -> dict[str, Any]:
    """The report without timing, memory and timestamp fields (for reproducibility checks)."""
    view = json.loads(json.dumps(report))
    for key in ("generated_at", "total_runtime_seconds", "environment"):
        view.pop(key, None)
    for suite in view["suites"].values():
        suite.pop("runtime_seconds", None)
        suite.pop("peak_memory_kib", None)
    return view


def write_reports(
    report: dict[str, Any], output_dir: Path, formats: Sequence[str] = ("json", "markdown")
) -> list[Path]:
    from artificial_human.cognition.benchmark.report import render_markdown

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if "json" in formats:
        path = output_dir / "cognitive-evaluation-results.json"
        path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(path)
    if "markdown" in formats:
        path = output_dir / "cognitive-evaluation-report.md"
        path.write_text(render_markdown(report), encoding="utf-8")
        written.append(path)
    return written
