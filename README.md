# Artificial Human

An experimental, consent-first artificial organism designed to model a person's knowledge, preferences, experiences, and working patterns while maintaining its own digital health.

This project uses the human body as an **engineering metaphor**. Biological systems inspire software boundaries; they are not claims that software is alive, conscious, or biologically equivalent.

## Vision

Build one coherent digital organism rather than a loose collection of chatbots:

- **Brain** — reasoning, goals, and plans.
- **Identity / DNA** — user-authored principles, preferences, and boundaries.
- **Memory / hippocampus** — consent-based episodic, semantic, procedural, preference, decision, and working memory.
- **Nervous system** — typed events and coordination.
- **Sensory system** — explicitly connected, permission-scoped observations.
- **Circulatory system** — prioritised work distribution.
- **Metabolism / oxygen** — bounded compute, time, and action budgets.
- **Immune system** — anomaly detection, containment proposals, investigation, and safety checks.
- **Regeneration** — isolated, test-first repair proposals; never silent self-modification.
- **Sleep** — review and consolidation proposals that preserve provenance and user control.

## Project status

This repository starts with a runnable reference implementation and an intentionally conservative safety model. It is a foundation for iterative research and engineering, not a claim of human-level intelligence or a literal digital person.

## Quick start

Requires Python 3.12 or newer.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m artificial_human simulate
python -m artificial_human serve
```

The API binds to `127.0.0.1:8000` by default. OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

Set `ARTIFICIAL_HUMAN_DB_PATH` to choose the SQLite database location.

## First-run personalisation

Personal memory is **off until the owner explicitly grants consent**.

1. Start the API.
2. Create a `personal_memory` consent record with `POST /consents`.
3. Add memories with `POST /memories`, referencing the returned consent ID.
4. Review, export, correct, or delete saved memories through the API.

Do not store passwords, API keys, authentication tokens, or other secrets in personal memory.

## Safety boundaries

- No ambient microphone, camera, filesystem, browser, or account access is enabled by default.
- Sensors accept explicit observations; connectors must be separately implemented and permission-scoped.
- No generated code or shell command is executed by the organism.
- Regeneration produces proposals for an isolated workspace and human review.
- External actions require an explicit, auditable approval step.
- Memory writes require active, purpose-specific consent and retain provenance.
- The local API is unauthenticated. Keep it on loopback; do not expose it to a network without adding authentication and transport security.

## Documentation

- [Biological blueprint](docs/BIOLOGICAL_BLUEPRINT.md)
- [System architecture](docs/ARCHITECTURE.md)
- [Safety and privacy](docs/SAFETY.md)
- [Development roadmap](docs/ROADMAP.md)
- [Contributing](CONTRIBUTING.md)

## License

Apache-2.0. See [LICENSE](LICENSE).
