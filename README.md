# Artificial Human

An experimental artificial mind and software organism grounded in human neuroscience and cognitive science. The goal is to study and engineer computational systems for perception, attention, memory, learning, prediction, reasoning, executive control, and adaptation.

The human brain is the **research foundation**, not a decorative metaphor. The current implementation consists of explicit software mechanisms inspired by selected cognitive functions. It is not a biological neural simulation, a complete human brain, a literal human, or evidence of consciousness.

## Vision

Build one coherent artificial organism with interacting cognitive and body-inspired systems:

| Human / biological reference | Artificial Human subsystem | Responsibility |
| --- | --- | --- |
| Sensory pathways | Perception | Transform explicit sensory input into structured percepts |
| Selective attention | Attention | Prioritise relevant information |
| Working-memory networks | Working memory | Hold and manipulate bounded active context |
| Hippocampal memory systems | Memory | Store and retrieve consented episodes and knowledge |
| Predictive processing | Prediction | Compare expectations/context with incoming information |
| Prefrontal executive functions | Executive control | Plan, monitor, and gate decisions |
| Language networks | Language interface | Language comprehension and generation |
| Learning and reinforcement systems | Learning | Adapt from reviewed evidence and evaluated outcomes |
| Sleep-dependent consolidation | Sleep | Review, replay, and consolidate memories |
| Neuronal communication | Nervous system | Typed internal events |
| Cardiovascular circulation | Circulation | Prioritised internal work scheduling |
| Metabolic regulation | Metabolism | Bounded compute and model budgets |
| Immune functions | Immune system | Detect anomalies and recommend defence |
| Tissue repair | Regeneration | Propose isolated repair work |

These are functional research mappings, not one-to-one anatomical equivalents. Brain regions and networks have distributed, overlapping functions.

## Current human-brain-inspired implementation

The Brain now runs an explicit cognitive cycle before calling the language model:

1. Perception normalizes explicit language input.
2. Memory retrieval gathers consent-visible memories and owner-approved learning examples.
3. Attention selects a bounded set of relevant items.
4. Working memory maintains a capacity-limited, volatile workspace.
5. Prediction calculates a lexical novelty signal against active context.
6. Executive control produces an inspectable goal plan with approval gates.
7. The language model acts as a language and deliberation component inside the architecture.
8. A cognitive trace reports which software systems operated.

Implemented modules live under `src/artificial_human/cognition/`. The details, endpoint examples, research limitations, and future mechanisms are documented in [Brain and intelligence](docs/BRAIN.md).

**Important:** the attention and novelty calculations are transparent engineering heuristics. Language perception is currently text-only. This milestone does not implement spiking neurons, realistic neural networks, emotion, homeostatic drives, reinforcement learning, or the full range of human cognitive abilities.

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

Set `ARTIFICIAL_HUMAN_DB_PATH` to choose the SQLite database location. The default is `.artificial-human/organism.sqlite3`.

## Brain provider configuration

The cognitive architecture can use a configurable OpenAI-compatible local or remote model. For local development with an Ollama-compatible endpoint:

~~~powershell
$env:ARTIFICIAL_HUMAN_LLM_BASE_URL = "http://127.0.0.1:11434/v1"
$env:ARTIFICIAL_HUMAN_LLM_MODEL = "your-local-model-name"
python -m artificial_human serve
~~~

See [Brain and intelligence](docs/BRAIN.md) for all configuration and security requirements. Remote providers must use HTTPS.

## Cognitive API

- `POST /brain/respond` — execute the cognitive cycle and generate a response.
- `GET /brain/cognition/status` — inspect the implemented subsystems.
- `GET /brain/status` — inspect model-provider configuration.
- `DELETE /brain/sessions/{session_id}` — clear a volatile working-memory session.

Example:

~~~json
{
  "message": "Explain how episodic memory works.",
  "mode": "assistant",
  "session_id": "optional-session-name"
}
~~~

Without a session ID, working memory is request-scoped. With one, it is retained only in bounded process memory and remains separate from persistent memory. It is not written to SQLite and is lost on process restart.

## Implemented organism foundation

- Modular Python runtime and explicit subsystem boundaries.
- Owner-authored identity profile and explicit observation intake.
- Local SQLite memory for episodic, semantic, procedural, preference, decision, and working data.
- Purpose-specific consent for personal memory writes; source provenance retained.
- Consent revocation, deletion, owner erasure, and export.
- Internal event bus, priority queue, and declared resource budgets.
- Health signal policies and anomaly-to-incident flow.
- Reviewable repair proposals and sleep/consolidation previews.
- FastAPI endpoints, CLI simulation, tests, and GitHub Actions CI.

## First-run personalisation

Personal memory is **off until the owner explicitly grants consent**.

1. Start the API.
2. Create a personal-memory consent grant with `POST /consents`.
3. Add memories with `POST /memories`, referencing the returned consent ID.
4. Inspect, export, or delete saved memories through the API.

Do not store passwords, API keys, authentication tokens, or other secrets in personal memory.

## Safety boundaries

- No ambient microphone, camera, browser, filesystem crawl, or account access is enabled.
- The model provider is configurable; remote endpoints must use HTTPS.
- No generated code or shell command is executed by the cognitive cycle.
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

The project has an executable organism foundation and an integrated, neuroscience-informed cognitive-cycle foundation. It is not yet a complete artificial human, a biologically faithful brain simulation, an autonomous self-repair system, or artificial consciousness. See the roadmap for the research and implementation stages.

## License

Apache-2.0. See LICENSE.
