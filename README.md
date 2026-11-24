# Artificial Human

An experimental, consent-first artificial organism designed to model a person's knowledge, preferences, experiences, and working patterns while maintaining its own digital health.

The human body is an **engineering metaphor** for modularity and feedback loops. These systems are software, not biological organs; this project does not claim that current AI is conscious or a literal human.

## Vision

Build one coherent digital organism rather than a loose collection of chatbots:

| Biological inspiration | Artificial system | Responsibility |
| --- | --- | --- |
| Brain | Brain | Reasoning, goals, and planning |
| DNA | Identity | Owner-authored principles and personality preferences |
| Hippocampus | Memory | Consent-based experience and knowledge storage |
| Neurons | Nervous system | Typed internal events |
| Eyes and ears | Sensory system | Explicit, permission-scoped observations |
| Heart and vessels | Circulation | Prioritised work distribution |
| Oxygen | Metabolism | Bounded compute and action budgets |
| White blood cells | Immune system | Detect anomalies and recommend defence |
| Stem cells | Regeneration | Propose isolated repairs |
| Sleep | Sleep system | Review and consolidate memory candidates |

## Quick start

Requires Python 3.12 or newer.

~~~bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m artificial_human simulate
python -m artificial_human serve
~~~

The API binds to 127.0.0.1:8000. OpenAPI docs: http://127.0.0.1:8000/docs.

Set ARTIFICIAL_HUMAN_DB_PATH to choose the SQLite database location. The default is .artificial-human/organism.sqlite3.

## Implemented in this foundation

- Modular Python runtime and subsystem boundaries.
- Owner-authored identity profile and explicit observation intake.
- Local SQLite memory for episodic, semantic, procedural, preference, decision, and working data.
- Purpose-specific consent required for personal memory writes; source provenance retained.
- Consent revocation, single-record deletion, owner erasure, and export.
- Internal event bus, priority queue, and declared resource budgets.
- Deterministic goal-plan skeleton.
- Health signal policies and anomaly-to-incident flow.
- Reviewable repair proposals and sleep/consolidation previews.
- FastAPI endpoints, CLI simulation, tests, and GitHub Actions CI.

## First-run personalisation

Personal memory is **off until the owner explicitly grants consent**.

1. Start the API.
2. Create a personal-memory consent grant with POST /consents.
3. Add memories with POST /memories, referencing the returned consent ID.
4. Inspect, export, correct, or delete saved memories through the API.

Do not store passwords, API keys, authentication tokens, or other secrets in personal memory.

### Train a personal clone (owner-supervised dataset)

Memory and training consent are separate. To contribute examples for clone-style learning:

1. Create a second consent grant with purpose `personal_learning`.
2. Submit owner demonstrations to `POST /training/examples` with a task category, input, target response, source, and consent ID.
3. Review each example at `GET /training/examples?status=pending_review`.
4. Approve or reject each example through `POST /training/examples/{id}/review`.
5. Export only reviewed examples from `GET /training/corpus`.

The exported corpus is supervised training data. This release does **not** fine-tune a model, infer your personality automatically, or claim the result is a complete clone. It creates a traceable, owner-curated dataset for later evaluation and model adaptation. Revoking the learning consent hides those examples from normal retrieval and corpus export; deleting an example or using owner erasure removes the stored record.

## Safety boundaries

- No ambient microphone, camera, browser, filesystem crawl, or account access is enabled.
- The baseline does not call hosted model providers.
- No generated code or shell command is executed by the organism.
- Regeneration creates review proposals; it does not modify source code, merge, or deploy.
- External actions require a separate approval mechanism.
- The local API has no authentication. Keep it bound to loopback; do not expose it to a network.

## Documentation

- [Biological blueprint](docs/BIOLOGICAL_BLUEPRINT.md)
- [System architecture](docs/ARCHITECTURE.md)
- [Safety and privacy](docs/SAFETY.md)
- [Development roadmap](docs/ROADMAP.md)
- [Contributing](CONTRIBUTING.md)

## Project status

This is milestone 0: an executable software-organism foundation. It is not yet a trained clone, autonomous self-repair system, artificial consciousness, or complete equivalent of a human body. See the roadmap for staged development.

## License

Apache-2.0. See LICENSE.
