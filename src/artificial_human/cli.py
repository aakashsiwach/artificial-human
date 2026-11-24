from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from artificial_human.cognition.benchmark.common import DEFAULT_SEED
from artificial_human.cognition.benchmark.runner import EXTERNAL_SUITES as EXTERNAL_COGNITIVE_SUITES
from artificial_human.cognition.benchmark.runner import SUITES as COGNITIVE_SUITES
from artificial_human.core.models import MemoryKind
from artificial_human.memory.embeddings import (
    DEFAULT_FASTEMBED_MODEL,
    EmbeddingProvider,
    FastEmbedProvider,
)
from artificial_human.memory.search import SearchFilters
from artificial_human.organism import ArtificialOrganism


def _simulate() -> None:
    organism = ArtificialOrganism(database_path=":memory:", embedding_provider=None)
    print(json.dumps({"status": organism.status()}, indent=2))
    print(json.dumps({
        "immune_demo": organism.evaluate_health("api_error_rate", 0.12, source="simulation")
    }, indent=2))
    print(json.dumps({"memory_demo": _memory_demo(organism)}, indent=2))


def _memory_demo(organism: ArtificialOrganism) -> dict[str, object]:
    """Synthetic temporal-memory demonstration (lexical + temporal; no model needed)."""
    owner = organism.owner_id
    consent = organism.memory.grant_consent(owner)
    previous, _ = organism.long_term_memory.add(
        owner_id=owner, kind=MemoryKind.SEMANTIC, consent_id=consent.id,
        content="The simulated owner works at Company A.", source="simulation",
        fact_key="owner.employer", valid_from="2025-01-01T00:00:00+00:00",
    )
    organism.long_term_memory.supersede(
        owner_id=owner, memory_id=previous.id, consent_id=consent.id,
        content="The simulated owner moved to Company B and works there.", source="simulation",
        valid_from="2026-03-01T00:00:00+00:00",
    )
    reference = datetime(2026, 6, 1, tzinfo=UTC)
    answers: dict[str, object] = {}
    for query in (
        "Where does the owner work now?",
        "Where did the owner work previously?",
        "Where did the owner work in January?",
    ):
        response = organism.long_term_memory.search(owner, query, reference_time=reference)
        answers[query] = {
            "strategy": response.strategy,
            "temporal_intent": response.temporal_query.intent.value,
            "top": [item.memory.content for item in response.results[:1]],
        }
    return answers


def _emit(text: str) -> None:
    """Write report text to stdout as UTF-8 regardless of the console code page.

    Reports contain non-ASCII characters (for example en dashes and minus signs);
    printing them through a cp1252 console or pipe raised UnicodeEncodeError.
    """
    stream = sys.stdout
    buffer = getattr(stream, "buffer", None)
    line = text + "\n"
    if buffer is None:
        stream.write(line)
        return
    stream.flush()
    buffer.write(line.encode("utf-8"))
    buffer.flush()


def _provider_from_args(args: argparse.Namespace) -> EmbeddingProvider | None:
    if args.backend == "none":
        return None
    return FastEmbedProvider(args.model, cache_dir=args.cache_dir, offline=args.offline)


def _memory_eval(args: argparse.Namespace) -> int:
    from artificial_human.memory.benchmark.runner import (
        render_markdown,
        run_benchmark,
        write_reports,
    )

    report = run_benchmark(
        _provider_from_args(args),
        dataset_path=Path(args.dataset) if args.dataset else None,
        threshold=args.threshold,
        progress=lambda message: print(f"[memory-eval] {message}", file=sys.stderr),
    )
    if args.output_dir:
        json_path, markdown_path = write_reports(report, Path(args.output_dir))
        print(f"[memory-eval] wrote {json_path} and {markdown_path}", file=sys.stderr)
    else:
        _emit(render_markdown(report))
    leaks = sum(item["overall"]["owner_leaks"] for item in report["variants"].values())
    if leaks:
        print(f"[memory-eval] owner isolation violated: {leaks} leaked results", file=sys.stderr)
        return 1
    return 0


def _memory_search(args: argparse.Namespace) -> int:
    organism = ArtificialOrganism()
    response = organism.long_term_memory.search(
        organism.owner_id, args.query, top_k=args.top_k,
        filters=SearchFilters(kinds=tuple(MemoryKind(kind) for kind in args.kind or ())),
    )
    payload = response.explain()
    for item, result in zip(payload["results"], response.results, strict=True):
        if args.show_content:
            item["content"] = result.memory.content
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _memory_index(args: argparse.Namespace) -> int:
    organism = ArtificialOrganism()
    if args.action == "rebuild":
        report = organism.long_term_memory.reindex(organism.owner_id, full=args.full)
        print(json.dumps(report.as_dict(), indent=2))
    print(json.dumps(organism.long_term_memory.status(organism.owner_id), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="artificial-human")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("simulate", help="Run an offline in-memory simulation")
    commands.add_parser("serve", help="Serve the loopback API on 127.0.0.1:8000")
    commands.add_parser("status", help="Print organism status")

    search = commands.add_parser("memory-search", help="Hybrid search of consented memories")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--kind", action="append", choices=[kind.value for kind in MemoryKind])
    search.add_argument(
        "--show-content", action="store_true",
        help="Include memory content in the output (omitted by default)",
    )

    index = commands.add_parser("memory-index", help="Inspect or rebuild the semantic index")
    index.add_argument("action", choices=("status", "rebuild"))
    index.add_argument("--full", action="store_true", help="Re-embed every eligible memory")

    evaluate = commands.add_parser("memory-eval", help="Run the retrieval benchmark")
    evaluate.add_argument("--backend", choices=("fastembed", "none"), default="fastembed")
    evaluate.add_argument("--model", default=DEFAULT_FASTEMBED_MODEL)
    evaluate.add_argument("--cache-dir", default=".artificial-human/models")
    evaluate.add_argument("--offline", action="store_true", help="Use only a cached model")
    evaluate.add_argument("--dataset", help="Alternative dataset JSON path")
    evaluate.add_argument(
        "--threshold", type=float, help="Fixed similarity threshold (skips tuning)"
    )
    evaluate.add_argument("--output-dir", help="Write JSON and Markdown reports here")

    cognition = commands.add_parser(
        "cognition-eval",
        help="Evaluate attention, working memory, prediction, memory and the integrated cycle",
        description=(
            "Runs offline by default. External suites (longmemeval, digit-span) run only when "
            "their local data path is given; nothing is ever downloaded. Exit status: 0 = "
            "evaluation completed (whatever the scores), 1 = execution or dataset error, "
            "2 = invalid arguments, 3 = a safety invariant (owner isolation) failed."
        ),
    )
    cognition.add_argument(
        "--suite", action="append",
        choices=("all", *COGNITIVE_SUITES, *EXTERNAL_COGNITIVE_SUITES),
        help="Suite to run (repeatable; default: all internal suites plus any external "
             "suite whose data path is given)",
    )
    cognition.add_argument(
        "--longmemeval-path",
        help="Local path to longmemeval_oracle.json (opt-in external suite)",
    )
    cognition.add_argument(
        "--digit-span-dir",
        help="Local directory of NEMAR on003838 sub-*_task-memory_beh.tsv files (opt-in)",
    )
    cognition.add_argument(
        "--external-limit", type=int,
        help="Evaluate only the first N LongMemEval questions (quick checks)",
    )
    cognition.add_argument(
        "--report-stem", default="cognitive-evaluation",
        help="File-name stem for written reports (for example external-evaluation)",
    )
    cognition.add_argument("--seed", type=int, default=DEFAULT_SEED)
    cognition.add_argument("--output-dir", help="Write report files here (default: stdout)")
    cognition.add_argument(
        "--format", action="append", choices=("json", "markdown"),
        help="Output format (repeatable; default: both with --output-dir, markdown on stdout)",
    )
    cognition.add_argument(
        "--embedding-backend", choices=("none", "fastembed"), default="none",
        help="Optional local embedding model for semantic variants (default: none, offline)",
    )
    cognition.add_argument("--model", default=DEFAULT_FASTEMBED_MODEL)
    cognition.add_argument("--cache-dir", default=".artificial-human/models")
    cognition.add_argument("--offline", action="store_true", help="Use only a cached model")
    return parser


def _cognition_eval(args: argparse.Namespace) -> int:
    from artificial_human.cognition.benchmark.external import ExternalDatasetError
    from artificial_human.cognition.benchmark.report import render_markdown
    from artificial_human.cognition.benchmark.runner import (
        InvariantViolation,
        resolve_suites,
        run_cognitive_evaluation,
        write_reports,
    )

    available = [name for name, value in (("longmemeval", args.longmemeval_path),
                                          ("digit-span", args.digit_span_dir)) if value]
    try:
        resolve_suites(args.suite or ["all"], available_external=available)
    except ValueError as error:
        print(f"[cognition-eval] invalid arguments: {error}", file=sys.stderr)
        return 2
    provider: EmbeddingProvider | None = None
    if args.embedding_backend == "fastembed":
        provider = FastEmbedProvider(args.model, cache_dir=args.cache_dir, offline=args.offline)
    try:
        report = run_cognitive_evaluation(
            args.suite or ["all"], seed=args.seed, provider=provider,
            progress=lambda message: print(f"[cognition-eval] {message}", file=sys.stderr),
            longmemeval_path=args.longmemeval_path, digit_span_dir=args.digit_span_dir,
            external_limit=args.external_limit,
        )
    except InvariantViolation as error:
        print(f"[cognition-eval] safety invariant failed: {error}", file=sys.stderr)
        return 3
    except ExternalDatasetError as error:
        print(f"[cognition-eval] external dataset error: {error}", file=sys.stderr)
        return 1
    except Exception as error:  # report execution errors without a traceback dump
        print(f"[cognition-eval] execution error: {type(error).__name__}: {error}",
              file=sys.stderr)
        return 1
    formats = args.format or (["json", "markdown"] if args.output_dir else ["markdown"])
    if args.output_dir:
        for path in write_reports(report, Path(args.output_dir), formats, args.report_stem):
            print(f"[cognition-eval] wrote {path}", file=sys.stderr)
    elif "json" in formats:
        _emit(json.dumps(report, indent=1, ensure_ascii=False))
    else:
        _emit(render_markdown(report))
    for name, suite in report["suites"].items():
        print(f"[cognition-eval] {name}: {suite['runtime_seconds']:.2f}s", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "simulate":
        _simulate()
    elif args.command == "status":
        print(json.dumps(ArtificialOrganism().status(), indent=2))
    elif args.command == "serve":
        import uvicorn
        uvicorn.run("artificial_human.api:app", host="127.0.0.1", port=8000, reload=False)
    elif args.command == "memory-search":
        return _memory_search(args)
    elif args.command == "memory-index":
        return _memory_index(args)
    elif args.command == "memory-eval":
        return _memory_eval(args)
    elif args.command == "cognition-eval":
        return _cognition_eval(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
