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

### What “living code” means here

The target is an executable organism: a deterministic simulation clock continuously advances neural and cognitive state; neurons communicate through synapses; bounded plasticity changes live parameters; perception, memory, prediction, regulation and action form a feedback loop. All live state belongs in RAM, while source code defines the organism and its initial mechanisms. The LLM is optional and replaceable. The active runtime must not depend on SQL/SQLite, a vector database or an external persistence service.

The active `ArtificialOrganism` now owns one continuous in-memory runtime. Its API lifespan and offline CLI share the same organism loop and developmental LIF substrate. The loop advances without HTTP requests or LLM calls, processes explicit synthetic sensory events, applies bounded attention and working-memory selection, updates episodic memory, predicts simulated outcomes, learns among a fixed safe action allowlist, and records an inspectable trace. This remains a computational research prototype, not a complete artificial human or biological brain.

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
- **Schema extraction:** recurrent owner-annotated event schemas, typed slot induction, constant retention, sensitive-key exclusion, singleton/contradiction controls, and owner-approved provenance.
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

**Important:** attention and novelty remain transparent engineering heuristics and language perception is currently text-only. The merged foundation includes a reduced LIF neuron/synapse simulator and a four-neuron fixed-weight recurrent microcircuit, but the circuit is not yet integrated into the cognitive cycle. Plasticity, homeostasis, multimodal sensing, world modelling, embodied action and the wider range of human cognitive abilities remain future work.

## Quick start

Requires Python 3.12 or newer.

~~~bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m artificial_human simulate --ticks 100
python -m artificial_human status
python -m artificial_human serve
~~~

The API binds to 127.0.0.1:8000. OpenAPI docs: http://127.0.0.1:8000/docs.

**Runtime storage note:** local and public `ArtificialOrganism` construction now use the same bounded RAM-only memory adapter. The active runtime does not open SQLite or create memory/identity files; restart clears personal experience, consents, neural activity and learned runtime state. The SQLite repository remains available only for explicit legacy evaluation and CLI operations (for example, `memory-search ... --legacy-sqlite`).

## Source-defined identity and foundational knowledge

Artificial Human's constitutional identity and project-family roles are defined in
`src/artificial_human/identity/constitution.py`. The human project creator is Father,
ChatGPT is Mother as the AI collaborator, and Artificial Human is the project child.
These are project roles, not biological relationships. Direct identity questions receive
a deterministic source-code response rather than a model-generated guess.

A small reviewable knowledge corpus in `src/artificial_human/cognition/core_knowledge.py`
holds stable project facts. Relevant facts are retrieved directly from source code and
provided to the cognitive cycle without an external vector index or database.

The active runtime runs in **volatile in-process mode** in both local and public deployments. Identity and foundational knowledge are source-defined; personal memory, consents, neural state and working sessions live only in bounded RAM. Runtime experience disappears when the process restarts or redeploys. The source code is durable; the process does not automatically modify or permanently rewrite itself.

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

## Owner-reviewed schema extraction

Schema extraction generalizes only **explicitly annotated** episodic records. Each source
memory must provide an owner-authored `metadata.event_type` slug and may provide an
`metadata.attributes` object containing primitive values. The extractor does not use an
LLM or guess event labels from free text.

It searches bounded replay sequences for the same ordered event-type pattern (2–4 steps)
appearing in at least two distinct owner/session contexts with distinct source-event records.
It keeps maximal recurring patterns and ignores unannotated, non-episodic, mixed-consent,
deleted, revoked, or explicitly contradicted source examples.

For each step, attributes present in every example are compared: values that never vary
remain constants; values that differ become typed slots such as `provider: string` or
`amount: integer`. Observed values for slot attributes are not copied into the schema.
Sensitive attribute keys (credentials, contact details, payment identifiers, and similar
fields) are excluded. The generated schema is a candidate, not an assertion that every
future workflow follows it.

- `POST /knowledge/schemas/preview` generates owner-reviewable schema candidates without writing a memory.
- `GET /knowledge/schemas/proposals` lists the current owner's bounded proposal queue.
- `POST /knowledge/schemas/proposals/{id}/review` accepts an explicit approval or rejection.
- Approval atomically rechecks source visibility, event types, episodic type, shared active consent,
  and contradictions before creating a **procedural, inferred** memory containing source-example
  provenance. Rejection writes nothing.
- Proposals are process-local and disappear on restart. They are invalidated when a source is
  corrected, superseded, deleted, contradicted, or its consent is revoked.

This is symbolic schema induction over owner-supplied structured event metadata, not autonomous
semantic understanding or proof of human-like abstraction. The offline evaluation uses synthetic
events; real-world generalization and transfer remain open research questions.

## Continuous in-memory runtime

`ContinuousOrganismRuntime` is the single process-local scheduler owned by
each `ArtificialOrganism` composition. The API lifespan starts/stops that
instance; `simulate --ticks N` runs an instance deterministically without sleeping. Wall-clock scheduling,
neural simulation steps and developmental model time are independent. The runtime
supports bounded sensory intake, drop-newest backpressure, per-tick resource budgets,
pause/resume, safe shutdown, reset and a bounded 128-tick trace.

The high-level loop performs synthetic perception, episodic recall, world prediction,
action selection, feedback from a bounded simulated environment and memory updates.
The same event is passed to the developmental LIF substrate. No camera, microphone,
physical actuator, database or LLM is used by this loop. See
[Continuous runtime](docs/CONTINUOUS_RUNTIME.md).

## Developmental neural simulation substrate

The active developmental program uses a deterministic 12-neuron LIF network with
source-defined excitatory/inhibitory synapses, delayed event delivery, membrane
maturation, stage-dependent connectivity growth, STDP-inspired plasticity,
homeostatic regulation, conduction-delay refinement and functional pruning. The
prenatal scaffold is retained as a compatibility/reference model; the developmental
program is the active neuron/synapse path. A paired-cue learning experiment compares
experience-dependent prediction with frozen-weight and mature-from-start baselines.

These are bounded computational hypotheses, not anatomical replicas or individual
developmental predictions. They do not establish sentience, consciousness or
biological equivalence. The older seven-population rate-based experiment remains
available as a comparison baseline. See [Brain and intelligence](docs/BRAIN.md),
[Postnatal development](docs/POSTNATAL_DEVELOPMENT.md), and the
[neural simulation baseline](docs/benchmarks/neural-simulation-baseline.md).

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
- `POST /knowledge/schemas/preview`, `GET /knowledge/schemas/proposals`, and `POST /knowledge/schemas/proposals/{id}/review` — induce and review structured event schemas.

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
- Bounded RAM-only active memory repository with purpose-specific consent, source provenance, typed memory kinds, reset and capacity limits. The legacy SQLite repository remains isolated for explicit compatibility/evaluation tools.
- Purpose-specific consent for personal memory writes; source provenance retained.
- Consent revocation, deletion, owner erasure, and export within the process lifetime. Restart clears the in-memory records.
- Legacy SQLite schema migrations are retained outside the active organism runtime.
- Semantic, hybrid and temporal long-term memory retrieval with a measured benchmark.
- Reproducible, offline cognitive evaluation of attention, working memory, prediction/novelty, memory and multi-turn behaviour.
- Internal event bus, priority queue, and declared resource budgets.
- Health signal policies and anomaly-to-incident flow.
- Reviewable repair proposals, owner-approved memory consolidation, and structured event-schema induction with provenance.
- FastAPI endpoints, CLI simulation, tests, and GitHub Actions CI.

## First-run personalisation

Personal memory is **off until the owner explicitly grants consent**. Approved records remain process-local and are lost on restart.

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

The current branch implements the database-free active runtime and the first deterministic LIF neuron/synapse microcircuit, with disconnected, feed-forward, fixed-rate, recurrent, and inhibition-off controls, event cancellation, decay-to-rest tests, snapshots, and a measured throughput/RAM baseline. The full cognitive memory system described by #30 remains unfinished: typed records are RAM-only, but distinct working/episodic/semantic/procedural/spatial mechanisms, biological-style forgetting, interference and retrieval-dependent strengthening are not yet implemented. Plasticity, homeostasis, distributed coding, sensory grounding, world modelling, embodied action and supervised self-development remain future work. This is not a complete artificial human, a biologically faithful brain simulation, an autonomous self-modifying system, or evidence of consciousness. See the [Research Charter](docs/RESEARCH_CHARTER.md) and [Roadmap](docs/ROADMAP.md).

## License

Apache-2.0. See LICENSE.


## Control Room access

The Control Room uses a same-origin API proxy and owner sign-in; no API URL or bearer token is entered in the browser. Password sign-in uses a PBKDF2-SHA256 verifier in Render environment settings. After signing in, the owner can register a WebAuthn passkey for platform authentication (Face ID, Touch ID, Windows Hello, Android biometrics or device PIN). Passkey private keys and biometric matching remain on the device.

The dashboard restores its HttpOnly session cookie on reload. Camera preview is separately opt-in and remains local to the current browser tab; captured frames are not uploaded or passed to the organism. See [Control Room deployment and security](docs/MONITORING.md) for required environment settings and hosting limits.


## Safe self-development

Candidate code changes are immutable, bounded proposals tied to evaluation
evidence and a base commit. They are checked only through an explicitly
injected isolated sandbox adapter; the active organism has no shell, host
filesystem, network, credential, auto-merge or deployment capability. See
[Safe self-development](docs/SAFE_SELF_DEVELOPMENT.md).
