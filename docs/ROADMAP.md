# Roadmap

The long-term goal is a computational artificial mind whose architecture is grounded in human neuroscience and cognitive science. The project proceeds through explicit research hypotheses and measurable implementations. It does not claim that present-day code is a complete human brain or conscious.

**Source of truth:** [Research Charter](RESEARCH_CHARTER.md). It records the north star, scientific evidence standards, current baseline, technology strategy, staged research plan, and a reusable context-recovery prompt. The roadmap records delivery status; the charter records why and how we choose the work. Always verify merged code and CI before changing checkboxes.

## Strategic execution order

1. **Close the measurement gap (next):** build held-out, multi-turn benchmarks for attention, working-memory retention, prediction, and memory recall. Expand beyond the current small, project-authored English memory dataset and record baselines, ablations, failure cases, and resource use.
2. **Deepen memory and learning:** add associative/relational memory, temporal context and contiguity, retrieval-dependent strengthening, forgetting/interference, source monitoring, and owner-reviewed consolidation. Compare each mechanism with the current static memory baseline.
3. **Ground the mind in the environment:** add permission-scoped multimodal perception, spatial/world-state representations, and uncertainty-aware multisensory integration.
4. **Add affect and regulation:** research bounded appraisal, interoception, homeostasis, and motivation, with explicit state and no unsupported claims about feelings.
5. **Develop adaptive action:** compare safe, simulation-first reinforcement and action-selection approaches under independent approval boundaries.
6. **Research neural substrates and alternative architectures:** compare recurrent, state-space, local-plasticity, spiking, and neuromorphic approaches only on tasks where their proposed mechanisms can be measured.
7. **Integrate and compare cognitive architectures:** evaluate closed-loop behaviour, generalisation, adaptation, calibration, robustness, and resource efficiency. Keep consciousness research separate from capability claims.

Do not skip the evaluation stage merely to add a visually impressive subsystem. Follow the charter's scientific protocol and minimum acceptance criteria.

## Milestone 0 — organism foundation

- [x] Modular organism subsystem boundaries.
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

## Milestone 1 — integrated cognitive-cycle foundation

- [x] Explicit language perception with source provenance.
- [x] Bounded attention over retrieved memory and approved learning candidates.
- [x] Capacity-limited volatile working memory with salience decay and reactivation.
- [x] Owner-scoped, bounded, explicitly clearable in-process working-memory sessions.
- [x] Contextual lexical novelty / prediction-error heuristic.
- [x] Executive goal decomposition with an explicit approval boundary.
- [x] Orchestrate the systems through a cognitive architecture before model generation.
- [x] Return inspectable cognitive traces and subsystem status.
- [x] Keep long-term memory and learning consent-gated.
- [x] Add semantic/vector/hybrid and temporal memory with measured retrieval evaluations ([docs](MEMORY.md), [ADR 0001](adr/0001-semantic-hybrid-temporal-memory.md), [report](benchmarks/memory-retrieval-report.md)).
  - [x] Replaceable embedding interface with a local in-process backend (fastembed/ONNX); remote backends need operator opt-in, HTTPS and owner consent.
  - [x] Consent-aware incremental indexing, re-indexing on model change, failure isolation, and deletion/revocation of vectors.
  - [x] Unified hybrid search (BM25 + embeddings + temporal channel, Reciprocal Rank Fusion) with filters and per-result explanations.
  - [x] Bitemporal records (event, recorded, validity times), supersession links, `known_at`/`valid_at` queries, contradiction preservation.
  - [x] Integration before attention in the cognitive cycle, with retrieval channels in the trace.
  - [x] Reproducible synthetic benchmark with tuning/eval split, baselines and ablations (Recall@K, Precision@K, MRR, nDCG@K).
- [ ] Build held-out benchmarks for attention, working-memory retention, recall, and prediction. (A synthetic long-term recall benchmark exists; it is not externally authored and does not cover attention, working memory, or prediction.)
- [ ] Add multimodal perception behind permission-scoped sensory interfaces.

## Milestone 2 — deeper biological and cognitive mechanisms

- [ ] Research and specify affective appraisal and salience without unsupported emotion inference.
- [ ] Implement homeostatic drives and motivation with bounded, inspectable state.
- [ ] Implement action selection and reinforcement learning with approval constraints.
- [ ] Add associative and relational memory, source reconciliation, and contradiction handling. (Contradictions are now preserved, linked and surfaced; reconciliation, source credibility and associative graphs are not implemented.)
- [ ] Add memory dynamics: retrieval-dependent strengthening, decay/interference, and contiguity-based temporal cueing, each evaluated against the current static retrieval.
- [ ] Replace English keyword temporal-intent rules with an evaluated parser; extend the memory benchmark with externally authored, larger held-out data.
- [ ] Investigate sleep-stage-inspired replay and owner-reviewed memory consolidation.
- [ ] Compare symbolic, predictive-processing, global-workspace, and neural-network approaches.
- [ ] Evaluate each mechanism against baselines and held-out cognitive tasks.
- [ ] Experiment with neural/spiking models only where they produce testable benefits.

## Milestone 3 — adaptive intelligence and learning

- [ ] Add continual learning with replay, forgetting controls, and catastrophic-forgetting evaluations.
- [ ] Evaluate model adaptation separately from memory retrieval and prompt conditioning.
- [ ] Measure factual recall, generalization, decision consistency, calibration, and transfer.
- [ ] Maintain a benchmark suite that catches regressions across cognitive functions.
- [ ] Add multimodal perception and grounded prediction as separately evaluated capabilities.

## Milestone 4 — organism telemetry and immune system

- [ ] Add read-only, permission-scoped host and application monitoring.
- [ ] Add event correlation, incident timelines, and severity calibration.
- [ ] Add safe containment proposals and operator escalation.
- [ ] Evaluate false positives, missed incidents, and recovery time.

## Milestone 5 — regeneration and self-maintenance

- [ ] Add ephemeral sandbox/worktree workers.
- [ ] Reproduce incidents with generated regression tests.
- [ ] Prepare immutable diffs and evidence bundles.
- [ ] Require review and separate deployment approval.
- [ ] Add rollback plans and post-repair verification.
- [ ] Add owner-reviewed self-maintenance proposals with independent tests and audit trails.

## Research discipline

For every brain-inspired capability, record:
1. The biological or cognitive-science evidence and competing explanations.
2. The computational abstraction and what it deliberately leaves out.
3. Falsifiable expected behavior and benchmark tasks.
4. Baseline comparisons, failure cases, and ablations.
5. Safety, consent, privacy, and human oversight requirements.

Do not mark a biological function complete because a module has a brain-region name. Completion requires tests and evidence that the intended computational behavior works.
