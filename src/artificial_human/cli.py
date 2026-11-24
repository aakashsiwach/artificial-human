from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

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
        print(render_markdown(report))
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
    return parser


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
