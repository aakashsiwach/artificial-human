# Architecture

## Runtime

The first version is a Python 3.12+ modular monolith with a local SQLite store and a loopback-only FastAPI interface. Modules communicate through explicit calls and a small synchronous event bus. This keeps the baseline inspectable before adding distributed infrastructure or model providers.

## Module boundaries

- brain: deterministic goal-plan skeleton; no external action executor.
- identity: owner-authored profile and boundaries.
- memory: SQLite repository enforcing active consent and source provenance.
- sensory: manual observations only; no ambient device access.
- nervous system: in-process event dispatch.
- circulation: bounded priority queue for internal work.
- metabolism: declared per-run operation, token, and elapsed-time budgets.
- immune: health-signal thresholds and incident generation.
- regeneration: repair proposals with approval state; no shell or code execution.
- sleep: consolidation review proposals; no automatic rewriting or forgetting.

## Data flow

1. An explicitly connected source submits an observation.
2. The sensory boundary validates source, subject, and scope.
3. Internal events communicate the observation to interested subsystems.
4. Memory persists a record only with active owner consent for personal_memory.
5. Health signals become incidents when a configured threshold is crossed.
6. Regeneration creates a reviewable proposal with tests and safety checks.
7. Sleep prepares a candidate review from existing consented memories.
8. The owner inspects, corrects, approves, or deletes data and proposals.
9. Owner demonstrations enter a separate personal_learning consent scope.
10. Training examples remain pending until the owner approves or rejects them.
11. Only approved examples under active learning consent can enter a supervised corpus export.

## Storage

SQLite is the local first-milestone store. Memory rows retain owner ID, type, content, source, consent grant, timestamp, and metadata. Revoking consent blocks future writes and excludes those records from normal retrieval. Explicit deletion endpoints support removal.

## API

The API is intended for loopback-only local use and has no authentication in this milestone. Do not expose it to a LAN, reverse proxy, or public internet until authentication, authorization, TLS, abuse controls, and audit logging are implemented.

## Extension points

Future adapters should implement narrow interfaces for reasoning providers, vector retrieval, external sensors, host telemetry, test runners, and isolated repair workspaces. Each connector must declare data scopes, network destinations, retention, and approval requirements.
