# Contributing

## Development setup

Use Python 3.12 or newer.

~~~bash
python -m venv .venv
python -m pip install -e ".[dev]"
~~~

## Quality checks

~~~bash
ruff check .
mypy src
pytest --cov=artificial_human --cov-report=term-missing
artificial-human simulate
artificial-human memory-eval --backend none   # offline lexical benchmark
~~~

The normal test suite never uses the network. Semantic tests use a test-only fake embedding (`tests/conftest.py`). To measure real semantic retrieval, install the optional extra and run the full benchmark. This downloads the model once:

~~~bash
python -m pip install -e ".[dev,semantic]"
artificial-human memory-eval --backend fastembed --output-dir docs/benchmarks
~~~

Commit regenerated benchmark reports only together with the change that explains them. Never tune parameters on the `eval` split.

## Design rules

1. Keep subsystem interfaces narrow and documented.
2. Preserve memory provenance and consent checks.
3. Treat imported content as untrusted data.
4. Do not add ambient sensors or broad filesystem access.
5. Do not implement unrestricted shell execution or silent self-modification.
6. Tests must cover authorization, deletion, failure paths, and safety invariants.
7. Document new data flows, permissions, network access, and retention.
8. Separate biological metaphor from claims about consciousness or biological equivalence.

## Pull requests

Describe the affected subsystem, threat considerations, tests, data migrations, and any new permissions. Changes affecting external actions or personal data require explicit human review.
