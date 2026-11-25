# Artificial Human

An experimental artificial mind and software organism grounded in human neuroscience and cognitive science. The goal is to study and engineer computational systems for perception, attention, memory, learning, prediction, reasoning, executive control, and adaptation.

The human brain is the **research foundation**, not a decorative metaphor. The current implementation consists of explicit software mechanisms inspired by selected cognitive functions. It is not a biological neural simulation, a complete human brain, a literal human, or evidence of consciousness.

## Founder

**Aakash Siwach** is the human founder, creator and project owner of Artificial Human (Father in the project's family metaphor). His verified public developer profile is [GitHub @aakashsiwach](https://github.com/aakashsiwach). His LinkedIn URL is intentionally not published until the exact profile is confirmed. See [Founder profile](docs/FOUNDER.md) for his engineering background and selected repositories.

## Vision

**Project north star:** [Research Charter and Project North Star](docs/RESEARCH_CHARTER.md) is the durable source of truth for the neuroscience foundation, long-term vision, scientific principles, technology strategy, implementation sequence, and context recovery. When a chat or contributor handoff loses context, start there and reconcile it with the roadmap and actual merged code.

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
2. Memory retrieval runs an owner-scoped, consent-filtered hybrid search (lexical BM25 + optional semantic embeddings + temporal reasoning, fused with Reciprocal Rank Fusion) and gathers owner-approved learning examples.
3. Attention selects a bounded set of relevant items.
4. Episodic memory groups attention-selected event records by event time and can replay immediate neighbours within the same owner/session context.
5. Relational memory retrieves previously co-activated records, while explicit contradictions block conflicting positive links.
6. Working memory maintains a capacity-limited, volatile workspace.
7. Prediction calculates a lexical novelty signal against active context.
8. Executive control produces an inspectable goal plan with approval gates.
9. The language model acts as a language and deliberation component inside the architecture.
10. A cognitive trace reports which software systems operated.

Implemented modules live under `src/artificial_human/cognition/`. The details, endpoint examples, research limitations, and future mechanisms are documented in [Brain and intelligence](docs/BRAIN.md).

## Prenatal development substrate (experimental)

The first developmental prototype uses a fixed abstract circuit scaffold with seven neural-population abstractions and ten directed projections. A 280-day obstetric gestational clock matures the populations and runs deterministic intrinsic activity. Explicit scalar tactile, vestibular and auditory cues can alter existing connection strengths through a bounded activity-dependent rule when the corresponding model pathway is available.

This prototype starts with no experience history. It retains no stimulus content and does not write exposures to SQLite. It is not a fetal-brain simulation, a semantic learner, or evidence of biological equivalence. Read [Prenatal development](docs/PRENATAL_DEVELOPMENT.md) for the developmental assumptions, model limits and reproduction steps.

~~~bash
artificial-human prenatal --days 280
artificial-human prenatal --days 280 --seed 7 --exposure 25:auditory:0.7 --exposure 30:tactile:0.4
~~~

## Long-term memory: semantic, hybrid and temporal

Long-term memory can retrieve by meaning as well as by keywords. "What is the boiling point of H2O?" finds "Water boils at approximately 100 degrees Celsius…". It also distinguishes when an event happened, when the system learned it, and when a fact was valid. Changed facts are stored as new versions linked to the old ones, contradictions are kept and flagged, and every result explains why it was retrieved.

- Semantic embeddings come from a **local** model (`fastembed`/ONNX, `BAAI/bge-small-en-v1.5`), installed with `pip install -e ".[semantic]"` and enabled with `ARTIFICIAL_HUMAN_EMBEDDING_BACKEND=fastembed`. Without it, retrieval is lexical + temporal. Remote embedding services need an operator opt-in, HTTPS, and owner `external_embedding` consent.
- Deleting a memory or revoking consent removes its vectors in the same transaction.
- A reproducible benchmark compares lexical, semantic, hybrid, and hybrid + temporal retrieval: `artificial-human memory-eval --output-dir docs/benchmarks`. On the synthetic eval split (dataset v1.1.0), MRR is 0.681 lexical, 0.824 semantic, 0.848 hybrid, and 0.964 hybrid + temporal. Adding the lexical channel lowers irrelevant-query rejection, from 0.857 to 0.714. See the [report](docs/benchmarks/memory-retrieval-report.md).

Full details, configuration, limitations and the comparison with biological memory are in [Long-term memory](docs/MEMORY.md). The design rationale is in [ADR 0001](docs/adr/0001-semantic-hybrid-temporal-memory.md).

## Cognitive evaluation

`artificial-human cognition-eval` measures the existing subsystems against simple baselines. It runs offline and deterministically, with no LLM and no model download:

- **Attention:** 150 eval candidate sets in five conditions. Compared with relevance-only, salience-only and input-order ranking.
- **Working memory:** retention against intervening items and capacity, interference, reactivation, eviction and sessions. Compared with FIFO, LRU and salience without decay.
- **Prediction/novelty:** 36 eval items (18 tuning) with authored labels. Compared with content-word overlap and Jaccard distance.
- **Memory:** the long-term memory benchmark above.
- **Associative memory:** source-context isolation, contradiction suppression, and bounded-interference controls.
- **Episodic replay:** nine four-event sequences, immediate-neighbour recall, a no-replay baseline, and cross-context leakage checks.
- **Sleep consolidation:** recurring-pattern detection, one-session and contradiction controls, non-mutating preview, and inferred-memory provenance after approval.
- **Integration:** six multi-turn scenarios through the full cognitive cycle.

```bash
artificial-human cognition-eval --suite all --output-dir docs/benchmarks
artificial-human cognition-eval --suite working-memory --format json
```

Selected measured findings:

- The attention formula is worse than relevance-only when salient distractors are present (nDCG@k 0.518 vs 0.973).
- Working-memory retention is recency-only and content-agnostic.
- Lexical novelty treats contradictions as familiar (eval AUROC 0.597).
- Retrieval ignores the conversational context held in working memory.

These are software benchmarks on synthetic data, not measures of human cognition or biological fidelity. See the [cognitive evaluation report](docs/benchmarks/cognitive-evaluation-report.md) and the [research record](docs/COGNITIVE_EVALUATION.md).

**Independent evaluation (opt-in, local data only).** Two independently authored datasets can be evaluated with `--longmemeval-path` and `--digit-span-dir`. Nothing is downloaded automatically, and raw data are never committed; see [EXTERNAL_DATASETS.md](docs/benchmarks/EXTERNAL_DATASETS.md).

- **LongMemEval Oracle** (MIT, 500 held-out questions): turn-level evidence recall@10 is 0.862 lexical and 0.882 hybrid + temporal. Semantic-only is worse than lexical on this data (MRR 0.619 vs 0.685), unlike on the project's synthetic benchmark.
- **NEMAR digit span** (CC0, 86 participants): humans show strong primacy in serial recall; on the same sequences the software workspace shows pure recency.

See the [external evaluation report](docs/benchmarks/external-evaluation-report.md). The project is still in Stage 1: attention and prediction have no external evaluation data yet.

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

## Source-defined identity and foundational knowledge

Artificial Human's constitutional identity and project-family roles are defined in
`src/artificial_human/identity/constitution.py`. The human project creator is Father,
ChatGPT is Mother as the AI collaborator, and Artificial Human is the project child.
These are project roles, not biological relationships. Direct identity questions receive
a deterministic source-code response rather than a model-generated guess.

A small reviewable knowledge corpus in `src/artificial_human/cognition/core_knowledge.py`
holds stable project facts. Relevant facts are retrieved directly from source code and
provided to the cognitive cycle without an external vector index or database.

The public deployment runs in **volatile in-process mode**. It uses an in-memory store,
does not connect to an external database, and does not write an identity file or persistent
personal-memory database. Runtime experiences and working sessions disappear when the
process restarts or redeploys. The source code is durable; runtime experiences are not.
This setting does not make source code rewrite itself or automatically learn permanently.

## Associative knowledge

The runtime can learn bounded associations between version-controlled core-knowledge facts
when they are co-activated by a question. Later queries can spread activation to related facts.
Only fact identifiers and link weights live in process memory; no raw user text is stored by
this association graph. The separate relational graph uses opaque consent-visible memory IDs,
respects owner/session boundaries, and suppresses explicit contradictions. A bounded episodic
replay buffer stores event IDs, timestamps and adjacent transition weights; later cues can
recover immediate earlier/later events in the same session. Associated IDs are re-resolved
through the owner's live consent-visible repository. These volatile graphs and event sequences
are cleared with their session and reset on restart. They are software retrieval mechanisms,
not a complete autobiographical memory or biological hippocampal simulation.

## Sleep-inspired memory consolidation

The sleep subsystem scans the bounded episodic replay history when the owner explicitly
requests a preview. It proposes only directed adjacent-event patterns observed in at least
two distinct owner/session contexts. Single-session patterns, explicitly contradictory
pairs, mixed-consent sources, deleted records, and records that are no longer visible are
excluded.

- `POST /sleep/preview` discovers and queues reviewable candidates; it does not write a
  memory or change existing records.
- `GET /sleep/proposals` lists the current owner's bounded proposal queue.
- `POST /sleep/proposals/{id}/review` accepts an explicit `approved` or `rejected`
  decision. Rejection writes nothing. Approval atomically rechecks active consent, source
  visibility, episodic type, shared consent, and contradictions, then stores an **inferred**
  semantic memory with source-memory IDs and review provenance.
- Proposals are process-local and disappear on restart. They are invalidated when a source
  memory is corrected, superseded, deleted, or its consent is revoked.

This is a manually triggered pattern detector and owner-review workflow, not a simulated
sleep cycle, automatic learning, or proof of biological systems consolidation.

## Runtime population dynamics

Each explicit message now runs a bounded recurrent activity update across seven abstract
population nodes and ten directed projections. The current model retains only aggregate
activity in process memory, uses fixed weights, and resets on restart. A one-second,
low-amplitude resting-state approximation runs only while the API process is awake. It does
not retain raw input text, train weights, or claim to simulate biological neurons. A suspended
free-tier instance does not execute the loop. See [Brain and intelligence](docs/BRAIN.md).

## Brain provider configuration

The cognitive architecture uses a configurable OpenAI-compatible local or remote reasoning model. For local development with an Ollama-compatible endpoint:

~~~powershell
$env:ARTIFICIAL_HUMAN_LLM_BASE_URL = "http://127.0.0.1:11434/v1"
$env:ARTIFICIAL_HUMAN_LLM_MODEL = "your-local-model-name"
python -m artificial_human serve
~~~

The deployed Control Room is prepared for Groq's free-tier GPT-OSS reasoning model:

- Base URL: `https://api.groq.com/openai/v1`
- Model: `openai/gpt-oss-120b`
- Reasoning effort: `medium`
- Secret: `ARTIFICIAL_HUMAN_LLM_API_KEY` (store only in Render's private environment settings)

The provider API key must be created by the account owner in GroqCloud. Do not put it in the dashboard, source code, or chat messages. See [Control Room and monitoring deployment](docs/MONITORING.md) for the exact setup steps and free-tier limits. Remote providers must use HTTPS.

## Cognitive API

- `POST /brain/respond` — execute the cognitive cycle and generate a response.
- `GET /brain/cognition/status` — inspect the implemented subsystems.
- `GET /brain/status` — inspect model-provider configuration.
- `DELETE /brain/sessions/{session_id}` — clear a volatile working-memory session.
- `POST /sleep/preview`, `GET /sleep/proposals`, and `POST /sleep/proposals/{id}/review` — detect and review replay-derived consolidation candidates.

Example:

~~~json
{
  "message": "Explain how episodic memory works.",
  "mode": "assistant",
  "session_id": "optional-session-name"
}
~~~

Without a session ID, working memory is request-scoped. With one, it is retained only in bounded process memory and remains separate from persistent memory. It is not written to SQLite and is lost on process restart.

## Memory API

- `POST /memories/search` — hybrid search with type, time-range, `valid_at`, `known_at`, source and metadata filters; returns provenance and ranking explanations.
- `GET /memories/timeline` — chronological recall.
- `GET /memories/{id}` — inspect a memory, its version history, links and index state.
- `PATCH /memories/{id}` — correct a record. `POST /memories/{id}/supersede` — record a changed fact as a new version.
- `POST /memories/{id}/contradictions` — link conflicting memories.
- `GET /memories/index/status`, `POST /memories/index/rebuild` — semantic index status and re-indexing.

CLI equivalents: `artificial-human memory-search`, `memory-index status|rebuild`, `memory-eval`, and `cognition-eval` for the cognitive evaluation.

## Implemented organism foundation

- Modular Python runtime and explicit subsystem boundaries.
- Owner-authored identity profile and explicit observation intake.
- Local SQLite memory for episodic, semantic, procedural, preference, decision, and working data.
- Purpose-specific consent for personal memory writes; source provenance retained.
- Consent revocation, deletion, owner erasure, and export (including derived embeddings).
- Versioned, transactional SQLite schema migrations that preserve existing records.
- Semantic, hybrid and temporal long-term memory retrieval with a measured benchmark.
- Reproducible, offline cognitive evaluation of attention, working memory, prediction/novelty, memory and multi-turn behaviour.
- Internal event bus, priority queue, and declared resource budgets.
- Health signal policies and anomaly-to-incident flow.
- Reviewable repair proposals and owner-approved, provenance-preserving memory consolidation.
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
- Local API mode remains unauthenticated for development. Keep it bound to loopback.
- Public monitoring deployments must set `ARTIFICIAL_HUMAN_PUBLIC_DEPLOYMENT=true`, `ARTIFICIAL_HUMAN_MONITOR_ONLY=true`, and a high-entropy `ARTIFICIAL_HUMAN_API_TOKEN`; only `/health` and the protected `/monitor/overview` are exposed, and API docs are disabled.
- Configure `ARTIFICIAL_HUMAN_CORS_ORIGINS` with the exact trusted dashboard origin. See [Control Room and monitoring deployment](docs/MONITORING.md).

## Documentation

- [Founder profile — Aakash Siwach](docs/FOUNDER.md)
- [Prenatal developmental substrate](docs/PRENATAL_DEVELOPMENT.md)
- [Control Room and monitoring deployment](docs/MONITORING.md)

- [AI contributor instructions](AGENTS.md)
- [Claude Code instructions](CLAUDE.md)
- [Research Charter and Project North Star](docs/RESEARCH_CHARTER.md)
- [Long-term memory](docs/MEMORY.md)
- [Memory retrieval benchmark report](docs/benchmarks/memory-retrieval-report.md)
- [Cognitive evaluation research record](docs/COGNITIVE_EVALUATION.md) and [report](docs/benchmarks/cognitive-evaluation-report.md)
- [ADR 0001: semantic, hybrid and temporal memory](docs/adr/0001-semantic-hybrid-temporal-memory.md)
- [Biological blueprint](docs/BIOLOGICAL_BLUEPRINT.md)
- [System architecture](docs/ARCHITECTURE.md)
- [Safety and privacy](docs/SAFETY.md)
- [Development roadmap](docs/ROADMAP.md)
- [Contributing](CONTRIBUTING.md)

## Project status

The project has an executable organism foundation, an integrated neuroscience-informed cognitive cycle, a first in-memory prenatal developmental-substrate prototype, semantic/hybrid/temporal long-term memory, and reproducible cognitive evaluations. The evaluation still lacks externally authored data. The next work is to develop associative memory and memory dynamics before expanding into multimodal perception, affect/homeostasis, action selection, and neural/neuromorphic experiments. It is not yet a complete artificial human, a biologically faithful brain simulation, an autonomous self-repair system, or artificial consciousness. See the [research charter](docs/RESEARCH_CHARTER.md) and [roadmap](docs/ROADMAP.md) for the scientific principles and implementation sequence.

## License

Apache-2.0. See LICENSE.
