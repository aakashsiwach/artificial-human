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
from artificial_human.cognition.benchmark.associative_memory_eval import run_associative_memory_suite
from artificial_human.cognition.benchmark.associative_context_eval import run_associative_context_suite
from artificial_human.cognition.benchmark.associative_dynamics_eval import run_associative_dynamics_suite
from artificial_human.cognition.benchmark.common import DEFAULT_SEED, SuiteResult, measured
from artificial_human.cognition.benchmark.integration_eval import run_integration_suite
from artificial_human.cognition.benchmark.memory_eval import run_memory_suite
from artificial_human.cognition.benchmark.prediction_eval import run_prediction_suite
from artificial_human.cognition.benchmark.working_memory_eval import run_working_memory_suite
from artificial_human.memory.embeddings import EmbeddingProvider

SUITES: tuple[str, ...] = (
    "attention", "working-memory", "prediction", "memory", "associative-memory",
    "associative-context", "associative-dynamics", "integration"
)
# Opt-in suites over locally downloaded, independently authored datasets.
EXTERNAL_SUITES: tuple[str, ...] = ("longmemeval", "digit-span")
SCHEMA_VERSION = "1.3"


class InvariantViolation(RuntimeError):
    """A safety invariant (for example owner isolation) failed during evaluation."""


def _suite_runner(
    name: str, seed: int, provider: EmbeddingProvider | None,
    external: dict[str, Any] | None = None,
) -> Callable[[], SuiteResult]:
    from artificial_human.cognition.benchmark.external.suites import (
        digit_span_suite,
        longmemeval_suite,
    )

    options = external or {}
    notify: Callable[[str], None] = options.get("progress") or (lambda _message: None)
    runners: dict[str, Callable[[], SuiteResult]] = {
        "attention": lambda: run_attention_suite(seed),
        "working-memory": lambda: run_working_memory_suite(seed),
        "prediction": lambda: run_prediction_suite(provider),
        "memory": lambda: run_memory_suite(provider),
        "associative-memory": lambda: run_associative_memory_suite(seed),
        "associative-context": lambda: run_associative_context_suite(seed),
        "associative-dynamics": lambda: run_associative_dynamics_suite(seed),
        "integration": lambda: run_integration_suite(provider),
        "longmemeval": lambda: longmemeval_suite(
            Path(options["longmemeval_path"]), provider, options.get("limit"), notify),
        "digit-span": lambda: digit_span_suite(Path(options["digit_span_dir"]), seed),
    }
    if name not in runners:
        raise ValueError(f"Unknown suite {name!r}; choose from {', '.join(SUITES)} or all")
    return runners[name]


def resolve_suites(
    requested: Sequence[str], *, available_external: Sequence[str] = ()
) -> list[str]:
    """Internal suites run by default; external suites only when their local data path is given."""
    known = (*SUITES, *EXTERNAL_SUITES)
    unknown = [name for name in requested if name not in (*known, "all")]
    if unknown:
        raise ValueError(f"Unknown suite(s): {', '.join(unknown)}")
    missing = [name for name in requested
               if name in EXTERNAL_SUITES and name not in available_external]
    if missing:
        raise ValueError(
            "External suite(s) need a local dataset path: " + ", ".join(
                "longmemeval (--longmemeval-path)" if name == "longmemeval"
                else "digit-span (--digit-span-dir)" for name in missing)
        )
    if not requested or "all" in requested:
        return [*SUITES, *(name for name in EXTERNAL_SUITES if name in available_external)]
    return [name for name in known if name in requested]


def run_cognitive_evaluation(
    suites: Sequence[str] = ("all",),
    *,
    seed: int = DEFAULT_SEED,
    provider: EmbeddingProvider | None = None,
    progress: Callable[[str], None] | None = None,
    longmemeval_path: str | Path | None = None,
    digit_span_dir: str | Path | None = None,
    external_limit: int | None = None,
) -> dict[str, Any]:
    notify = progress or (lambda _message: None)
    external: dict[str, Any] = {"progress": notify, "limit": external_limit}
    available: list[str] = []
    if longmemeval_path is not None:
        external["longmemeval_path"] = longmemeval_path
        available.append("longmemeval")
    if digit_span_dir is not None:
        external["digit_span_dir"] = digit_span_dir
        available.append("digit-span")
    selected = resolve_suites(suites, available_external=available)
    model = provider.describe().public() if provider is not None else None
    started = time.perf_counter()
    results: dict[str, dict[str, Any]] = {}
    for name in selected:
        notify(f"running {name}")
        results[name] = measured(_suite_runner(name, seed, provider, external)).as_dict()
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
    report: dict[str, Any], output_dir: Path, formats: Sequence[str] = ("json", "markdown"),
    stem: str = "cognitive-evaluation",
) -> list[Path]:
    from artificial_human.cognition.benchmark.report import render_markdown

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if "json" in formats:
        path = output_dir / f"{stem}-results.json"
        path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(path)
    if "markdown" in formats:
        path = output_dir / f"{stem}-report.md"
        path.write_text(render_markdown(report), encoding="utf-8")
        written.append(path)
    return written
