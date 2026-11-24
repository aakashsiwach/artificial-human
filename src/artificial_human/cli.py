from __future__ import annotations

import argparse
import json

from artificial_human.organism import ArtificialOrganism


def _simulate() -> None:
    organism = ArtificialOrganism(database_path=":memory:")
    print(json.dumps({"status": organism.status()}, indent=2))
    print(json.dumps({
        "immune_demo": organism.evaluate_health("api_error_rate", 0.12, source="simulation")
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="artificial-human")
    parser.add_argument("command", choices=("simulate", "serve", "status"))
    args = parser.parse_args()

    if args.command == "simulate":
        _simulate()
    elif args.command == "status":
        print(json.dumps(ArtificialOrganism().status(), indent=2))
    elif args.command == "serve":
        import uvicorn
        uvicorn.run("artificial_human.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
