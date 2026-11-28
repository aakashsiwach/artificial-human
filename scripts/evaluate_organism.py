from __future__ import annotations

import argparse
import json
from pathlib import Path

from artificial_human.neural.organism_evaluation import run_organism_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run fixed-seed offline neural and organism evaluation"
    )
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--integration-ticks", type=int, default=60)
    parser.add_argument("--neural-steps", type=int, default=20_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-output/organism-evaluation.json"),
    )
    args = parser.parse_args()

    report = run_organism_evaluation(
        seeds=tuple(args.seeds) if args.seeds else (17, 23, 41),
        integration_ticks=args.integration_ticks,
        neural_benchmark_steps=args.neural_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "evaluation": report["evaluation"],
        "fixed_seeds": report["fixed_seeds"],
        "output": str(args.output),
        "limitations": report["limitations"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
