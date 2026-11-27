from __future__ import annotations

import argparse
import json
from pathlib import Path

from artificial_human.neural.benchmark import benchmark_neural_simulation


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the deterministic LIF microcircuit")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    report = benchmark_neural_simulation(steps=args.steps)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
