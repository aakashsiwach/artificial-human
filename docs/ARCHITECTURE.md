# Architecture

## Runtime

The project is a Python 3.12+ modular monolith with a local SQLite store and a loopback-only FastAPI interface. Modules communicate through explicit calls and a small synchronous event bus. The human brain is the research reference; software modules are transparent computational approximations, not literal biological organs.

## Cognitive module boundaries

- **cognition.perception:** normalize explicit language input and attach source/time provenance.
- **cognition.attention:** select a limited set of candidate memories and approved demonstrations.
- **cognition.working_memory:** maintain a bounded volatile workspace, salience decay, and reactivation counts.
- **cognition.prediction:** compute a lexical novelty / overlap signal against active workspace context.
- **cognition.executive:** decompose a goal into inspectable steps with an approval boundary.
- **cognition.architecture:** orchestrate the cognitive cycle and create an inspectable trace.
- **brain.service:** pass selected context to a configurable language model and return its answer with the cognitive trace.

The model is one component of the cognitive architecture; it does not replace the surrounding systems.

## Integrated cognitive data flow

1. The perception system converts explicitly supplied language into a `Percept`.
2. Memory retrieval returns only memories whose `personal_memory` consent remains active.
3. The learning system returns only owner-approved demonstrations under active `personal_learning` consent.
4. The attention system ranks candidate references and selects a bounded set.
5. The working-memory system activates selected content in a capacity-limited workspace. No session ID means request-scoped memory; a supplied session ID uses a bounded, owner-scoped process-memory session.
6. The prediction system estimates lexical novelty relative to previous workspace context. This signal prompts assumption-checking; it is not a truth score.
7. Executive control creates a goal plan and marks steps that need approval.
8. The language model receives the current percept, selected evidence, bounded working-memory context, prediction signal, and executive plan.
9. The response includes an evidence list and cognitive trace. Hidden chain-of-thought is not exposed.
10. The owner can clear a volatile session. Process restart also clears all volatile sessions.

## Remaining organism modules

- **identity:** owner-authored profile and boundaries.
- **memory:** SQLite repository enforcing active consent and source provenance.
- **sensory:** manual observations only; no ambient device access.
- **nervous system:** in-process event dispatch.
- **circulation:** bounded priority queue for internal work.
- **metabolism:** declared per-run operation, token, and elapsed-time budgets.
- **immune:** health-signal thresholds and incident generation.
- **regeneration:** repair proposals with approval state; no shell or code execution.
- **sleep:** consolidation review proposals; no automatic rewriting or forgetting.

## Storage and privacy

SQLite is the persistent store. Memory rows retain owner ID, type, content, source, consent grant, timestamp, and metadata. Revoking consent blocks future writes and excludes those records from normal retrieval. Explicit deletion endpoints support removal.

Working memory is separate from persistent memory. Optional working-memory sessions live in process memory, are bounded, owner-scoped, and explicitly clearable. They are never implicitly copied to SQLite.

## API

- `POST /brain/respond` runs the cognitive cycle and generation.
- `GET /brain/cognition/status` reports the active cognitive subsystem implementations.
- `DELETE /brain/sessions/{session_id}` clears volatile working memory.
- `GET /brain/status` reports model-provider configuration.
- Existing identity, consent, memory, learning, immune, regeneration, and sleep endpoints remain available.

The API is unauthenticated in this milestone and must remain loopback-only. Do not expose it to a LAN, reverse proxy, or public internet until authentication, authorization, TLS, abuse controls, and audit logging are implemented.

## Extension points

Future research adapters should implement narrow interfaces for semantic/hybrid retrieval, multimodal sensory processing, affective appraisal, homeostatic drives, action selection, reinforcement learning, temporal/relational memory, neural or spiking-network experiments, host telemetry, test runners, and isolated repair workspaces. Each connector must declare data scopes, network destinations, retention, approval requirements, hypotheses, and evaluation measures.
