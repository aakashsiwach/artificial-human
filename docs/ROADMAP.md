# Roadmap

The project is intentionally incremental. “Artificial human” describes the long-term research direction, not a claim that today's code is conscious or a complete human clone.

## Milestone 0 — organism foundation

- [x] Modular biological subsystem boundaries.
- [x] Consent-gated SQLite memory with provenance and deletion.
- [x] Explicit manual observation intake.
- [x] Typed internal event bus.
- [x] Resource budget and priority queue.
- [x] Deterministic goal planning skeleton.
- [x] Health signal detection and incident creation.
- [x] Reviewable regeneration proposals.
- [x] Sleep/consolidation review proposals.
- [x] Local API, CLI, tests, and CI.
- [x] Separate consent scope for owner-supervised learning examples.
- [x] Pending-review, approve/reject, delete, and approved-corpus export workflow.

## Milestone 1 — model-backed brain and identity

- [x] Persist owner-authored identity profiles and boundaries.
- [x] Add an OpenAI-compatible provider adapter for local and remote models.
- [x] Add consent-filtered lexical memory retrieval.
- [x] Supply owner-approved demonstrations in style-clone drafting mode.
- [x] Expose brain, identity, and provider status APIs.
- [x] Return evidence references and keep conversations non-persistent by default.
- [x] Include a visible task workflow in Brain responses (without exposing hidden chain-of-thought).
- [x] Add held-out style response evaluation with token F1 and ROUGE-L surface metrics.
- [ ] Add semantic/vector and hybrid retrieval with measured retrieval evaluations.
- [ ] Add structured conversation sessions with explicit opt-in persistence.

## Milestone 2 — deeper cognition and clone evaluation

- Add multi-step reasoning, reflection, and plan critique.
- Add a bounded tool registry with per-tool scopes and approval policies.
- Build clone-fidelity benchmarks from owner-approved held-out examples.
- Measure voice/style similarity, factual recall, decision consistency, and uncertainty calibration.
- Distinguish remembered facts, inferences, predictions, and generated content.
- Evaluate local fine-tuning or other adaptation methods against a held-out baseline.

## Milestone 3 — body telemetry and immune system

- Add read-only, permission-scoped host and application monitoring.
- Add event correlation, incident timelines, and severity calibration.
- Add safe containment proposals and operator escalation.
- Evaluate false positives, missed incidents, and recovery time.

## Milestone 4 — regeneration

- Add ephemeral sandbox/worktree workers.
- Reproduce incidents with generated regression tests.
- Prepare immutable diffs and evidence bundles.
- Require review and separate deployment approval.
- Add rollback plans and post-repair verification.

## Milestone 5 — continuous learning

- Add scheduled sleep cycles and owner-reviewed consolidation.
- Evaluate memory usefulness, staleness, and contradiction.
- Add budget-aware resource scheduling and provider cost controls.
- Add self-maintenance proposals for this project, with independent tests and review.
