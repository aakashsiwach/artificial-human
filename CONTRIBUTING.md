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
artificial-human cognition-eval --suite all   # offline cognitive evaluation (attention, working memory, prediction, memory, integration)
~~~

The normal test suite never uses the network. Semantic tests use a test-only fake embedding (`tests/conftest.py`). To measure real semantic retrieval, install the optional extra and run the full benchmark. This downloads the model once:

~~~bash
python -m pip install -e ".[dev,semantic]"
artificial-human memory-eval --backend fastembed --output-dir docs/benchmarks
~~~

Commit regenerated benchmark reports only together with the change that explains them. Never tune parameters on the `eval` split. A new cognitive mechanism must report its effect on the relevant `cognition-eval` metrics against the existing baseline.

## Design rules

0. The target active organism runtime is source-defined and RAM-only. Do not introduce SQLite/SQL, database files, vector databases or external persistence into the active runtime. SQLite `:memory:` is still a database and does not satisfy this rule. Follow [AGENTS.md](AGENTS.md) (and [CLAUDE.md](CLAUDE.md) in Claude Code); read [docs/RESEARCH_CHARTER.md](docs/RESEARCH_CHARTER.md) and [docs/ROADMAP.md](docs/ROADMAP.md) before proposing a substantial cognitive or neuroscience-related change. The repository is the source of truth when chat context is missing.

1. Keep subsystem interfaces narrow and documented.
2. Preserve memory provenance and consent checks.
3. Treat imported content as untrusted data.
4. Do not add ambient sensors or broad filesystem access.
5. Do not implement unrestricted shell execution or silent self-modification.
6. Keep simulation, neural dynamics, memory and learning independent of LLM calls.
7. Put live state in bounded in-process structures; document reset, capacity, eviction and concurrency semantics.
8. A self-development feature may create isolated candidate diffs/tests only; require human approval before source adoption, merge or deployment.
9. Tests must cover failure paths, bounded state, deterministic replay, resource limits and safety invariants.
10. Document new data flows, permissions, network access and retention.
11. Separate biological metaphor from claims about consciousness or biological equivalence.
12. State the scientific hypothesis, evidence category, computational abstraction, baselines, ablations and measurable acceptance criteria for every new brain-inspired mechanism.
13. Update the research charter, roadmap and subsystem documentation when the project's north star, implementation status or sequence changes.
14. Follow the ordered issue dependencies; do not open multiple foundational feature PRs at once.

## Pull requests

Describe the affected subsystem, threat considerations, tests, data migrations, and any new permissions. Changes affecting external actions or personal data require explicit human review.
