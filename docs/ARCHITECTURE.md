# Architecture

## Runtime

The [Research Charter](RESEARCH_CHARTER.md) defines the project north star, scientific method, technology-selection criteria, staged roadmap, and context-recovery process. It is the required starting point for substantial architecture changes.

The project is a Python 3.12+ modular monolith with a local SQLite store and a loopback-only FastAPI interface. Modules communicate through explicit calls and a small synchronous event bus. The human brain is the research reference; software modules are transparent computational approximations, not literal biological organs.

## Cognitive module boundaries

- **cognition.perception:** normalize explicit language input and attach source/time provenance.
- **cognition.attention:** select a limited set of candidate memories and approved demonstrations.
- **cognition.working_memory:** maintain a bounded volatile workspace, salience decay, and reactivation counts.
- **cognition.prediction:** compute a lexical novelty / overlap signal against active workspace context.
- **cognition.executive:** decompose a goal into inspectable steps with an approval boundary.
- **cognition.architecture:** orchestrate the cognitive cycle and create an inspectable trace.
- **brain.service:** pass selected context to a configurable language model and return its answer with the cognitive trace.
- **cognition.benchmark:** reproducible evaluation of the subsystems above (seeded synthetic datasets → execution against the real classes → shared metrics → JSON/Markdown reports). It reuses `memory.benchmark` for long-term recall, makes no network calls by default, and never writes to the organism's database. CLI: `artificial-human cognition-eval`.

The model is one component of the cognitive architecture; it does not replace the surrounding systems.

## Integrated cognitive data flow

1. The perception system converts explicitly supplied language into a `Percept`.
2. Memory retrieval runs `HybridMemorySearch` for the requesting owner over memories whose `personal_memory` consent remains active. Lexical, semantic and temporal channels are fused by Reciprocal Rank Fusion. Twice the memory budget is returned so attention can choose among them, and each candidate carries its channels and explanation.
3. The learning system returns only owner-approved demonstrations under active `personal_learning` consent.
4. The attention system ranks candidate references and selects a bounded set.
5. The working-memory system activates selected content in a capacity-limited workspace. No session ID means request-scoped memory; a supplied session ID uses a bounded, owner-scoped process-memory session.
6. The prediction system estimates lexical novelty relative to previous workspace context. This signal prompts assumption-checking; it is not a truth score.
7. Executive control creates a goal plan and marks steps that need approval.
8. The language model receives the current percept, selected evidence, bounded working-memory context, prediction signal, and executive plan.
9. The response includes an evidence list and cognitive trace. Hidden chain-of-thought is not exposed.
10. The owner can clear a volatile session. Process restart also clears all volatile sessions.

## Remaining organism modules

## Target cognitive architecture

The current runtime is a text-driven cycle. The long-term target is a closed-loop system that connects sensory and interoceptive input, perceptual state estimation, associative memory, attention, predictive world modelling, affective appraisal, homeostatic state, goal/action selection, environment feedback, learning, and consolidation. Language models remain replaceable components inside the architecture, not the architecture itself.

These are research directions, not implemented runtime modules. Every future component must have a defined hypothesis, input/output contract, measurable task, baseline, ablation, limitations, privacy scope, and safety boundary. See [Research Charter](RESEARCH_CHARTER.md).

- **identity:** owner-authored profile and boundaries.
- **memory:** SQLite repository enforcing active consent and source provenance, with versioned migrations.
- **memory.temporal / memory.search / memory.index / memory.embeddings:** temporal model and intent rules, hybrid search with explanations, consent-aware vector indexing, and replaceable embedding backends (local by default; see [ADR 0001](adr/0001-semantic-hybrid-temporal-memory.md)).
- **memory.benchmark:** synthetic dataset, IR metrics and the reproducible evaluation runner.
- **sensory:** manual observations only; no ambient device access.
- **nervous system:** in-process event dispatch.
- **circulation:** bounded priority queue for internal work.
- **metabolism:** declared per-run operation, token, and elapsed-time budgets.
- **immune:** health-signal thresholds and incident generation.
- **regeneration:** repair proposals with approval state; no shell or code execution.
- **sleep:** consolidation review proposals; no automatic rewriting or forgetting.

## Storage and privacy

SQLite is the persistent store. Memory rows retain owner ID, type, content, source, consent grant, recorded time, metadata, and (schema v2) optional event time, validity interval, fact key, epistemic status and correction time. Revoking consent blocks future writes, excludes those records from normal retrieval, and deletes their embeddings. Explicit deletion endpoints support removal.

Schema v2 tables:

- `memory_links` — `supersedes` and `contradicts` relations between an owner's memories, with effective time and note.
- `memory_embeddings` — one vector per memory and embedding space: model id, version, dimensions, content hash, float32 vector. Deleted with the memory, on consent revocation, or when the model changes and the index is rebuilt.
- `memory_index_failures` — memory id, space and error class only. Never content.

Migrations are tracked with `PRAGMA user_version`, run in per-version transactions, and never modify existing rows. Semantic indexing is a derived, rebuildable layer. The memory record is always written first and remains the source of truth.

Working memory is separate from persistent memory. Optional working-memory sessions live in process memory, are bounded, owner-scoped, and explicitly clearable. They are never implicitly copied to SQLite.

## API

- `POST /brain/respond` runs the cognitive cycle and generation.
- `GET /brain/cognition/status` reports the active cognitive subsystem implementations.
- `DELETE /brain/sessions/{session_id}` clears volatile working memory.
- `GET /brain/status` reports model-provider configuration.
- `POST /memories/search`, `GET /memories/timeline`, `GET|PATCH /memories/{id}`, `POST /memories/{id}/supersede`, `POST /memories/{id}/contradictions`, `GET /memories/index/status` and `POST /memories/index/rebuild` expose long-term memory search, temporal recall, versioning and indexing (see [MEMORY.md](MEMORY.md#8-api-and-cli)).
- Existing identity, consent, memory, learning, immune, regeneration, and sleep endpoints remain available. Existing memory responses gain fields only.

The API is unauthenticated in this milestone and must remain loopback-only. Do not expose it to a LAN, reverse proxy, or public internet until authentication, authorization, TLS, abuse controls, and audit logging are implemented.

## Extension points

Embedding backends already implement the narrow `EmbeddingProvider` interface (`describe`, `embed_documents`, `embed_query`) and declare their locality. Future research adapters should implement similar interfaces for approximate-nearest-neighbour indexing, multimodal sensory processing, affective appraisal, homeostatic drives, action selection, reinforcement learning, relational/associative memory, neural or spiking-network experiments, host telemetry, test runners, and isolated repair workspaces. Each connector must declare data scopes, network destinations, retention, approval requirements, hypotheses, and evaluation measures.
