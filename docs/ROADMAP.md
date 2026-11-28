# Roadmap — the code-first living organism

The north star is one executable computational organism grounded in human neuroscience and cognitive science. Its brain must be implemented as explicit, continuously advancing code with bounded RAM state. This is not an LLM wrapped around a memory database.

**Source of truth:** [Research Charter](RESEARCH_CHARTER.md). This roadmap tracks implementation and evidence. The charter records the scientific principles and research rationale. Verify merged code and CI before changing any status.

## Non-negotiable runtime contract

- Source code defines the organism's architecture, neural mechanisms, initial parameters, developmental programme, and stable constitution.
- The active brain runtime uses bounded RAM state only: no SQLite/SQL, database, vector database, external memory service, or disk-backed runtime state.
- Neuron activity, synapses, weights, working state, episodic/semantic/procedural memory, goals, and internal drives live in process memory for this phase.
- A deterministic simulation clock advances the organism independently of HTTP requests and language-model calls.
- Core neural activity, adaptation, memory dynamics, prediction, and simulated action must run with the LLM disabled.
- Restart resets runtime experience and learned parameters for now. Durable learning may later be exported as reviewed, version-controlled source artefacts.
- Self-development creates isolated candidate patches, tests, immutable diffs, and evidence. It never silently mutates the running executable, merges, deploys, expands permissions, or accesses secrets.
- Biological labels identify research questions, not one-to-one anatomical replicas. No claims of consciousness, sentience, or human-level intelligence.

## Verified current baseline

The merged main branch and the current issue #36 implementation contain the
following foundation. Issue #36 is complete only when its PR is merged with
green post-merge CI.

- The active `ArtificialOrganism` uses source-defined identity and bounded
  process-local memory in local and public modes. Default construction does not
  open SQLite or create persistence files. The SQLite repository is lazy-loaded
  only for explicit legacy operations.
- `neural/simulation.py` provides deterministic LIF membrane integration,
  excitatory/inhibitory synapses, delayed events, stable ordering, bounded
  queues/currents, cancellation, pause/resume, single-step, reset and
  in-memory snapshots.
- `development/program.py` is the active developmental LIF program. It uses
  12 neurons and stage-dependent synapses, membrane/refractory maturation,
  conduction-delay refinement, bounded STDP-inspired plasticity, homeostatic
  regulation and functional pruning. Prenatal and postnatal model clocks are
  explicit. These are engineering hypotheses, not anatomical replicas.
- `neural/circuits.py` retains a four-neuron recurrent circuit as a controlled
  baseline; `cognition/dynamics.py` retains the seven-population rate model as
  a request-path comparison component. Neither is advanced as a second
  autonomous brain by the continuous runtime.
- PR #55 adds a bounded LIF episodic engram circuit with sparse keyed feature
  codes, context binding, partial-cue completion, inhibitory competition,
  interference/pattern-separation measures and retrieval-dependent
  strengthening. Owner-scoped memory records remain the source of event
  content; raw text is not retained in the neural engram index.
- The high-level `OrganismLoop` processes explicit synthetic sensory events,
  bounded episodic memory, world-state prediction, a simple action learner,
  homeostatic state and a deterministic simulated environment. The same event
  is delivered to the developmental LIF program. Action choice is not yet
  decoded from LIF population activity; issues #32–#34 are implemented as separate bounded slices, while #38 remains open for integrated circuit-to-organism coupling.
- Issue #36's `ContinuousOrganismRuntime` implementation gives
  `ArtificialOrganism` one composition-owned runtime. The API lifespan starts
  and stops it; the CLI supports deterministic accelerated ticks. It adds a
  bounded sensory queue, drop-newest backpressure, per-tick resource budgets,
  pause/resume, shutdown, reset, snapshots and a bounded recent trace. Neural,
  developmental and wall-clock time remain independent.
- Synthetic sensory events exercise vision, audition, touch, proprioception
  and interoception interfaces. No camera, microphone, physical actuator or
  real-world action is enabled.
- The consent-aware RAM repository, specialized cognitive-memory stores and
  neural episodic circuit remain in process memory. The continuous loop's
  synthetic episodes are distinct from owner-controlled personal memories.
- The optional LLM is a language/request adapter. The continuous neural,
  memory, prediction, regulation and simulated-action loop does not require
  it.
- Code repair currently creates proposals only; there is no autonomous
  self-authored code/test lifecycle or automatic merge/deployment.

These are engineering mechanisms, not biological equivalence or claims of
consciousness. See [Brain](BRAIN.md), [Architecture](ARCHITECTURE.md),
[Continuous runtime](CONTINUOUS_RUNTIME.md), [Memory](MEMORY.md), and
[Safety](SAFETY.md).

## Ordered implementation programme

### Phase 0 — align direction and establish the RAM-only runtime contract

- [x] [#17 — Align architecture and project docs with the code-first living-organism direction](https://github.com/aakashsiwach/artificial-human/issues/17). Closed.
- [x] [#18 — Specify the database-free organism runtime and migration boundary](https://github.com/aakashsiwach/artificial-human/issues/18). Merged and closed after PR #43.

**Exit gate:** architecture docs agree; tests prove default startup creates no database and all live state is bounded RAM state.

### Phase 1 — neural simulation foundation

- [x] [#19 — Deterministic neural simulation kernel and biological timebase](https://github.com/aakashsiwach/artificial-human/issues/19). Merged and closed after PR #43, including a reproducible performance/RAM baseline.
- [x] [#20 — Neuron models and inspectable membrane dynamics](https://github.com/aakashsiwach/artificial-human/issues/20). Merged and closed after PR #43 with LIF state, spike timestamps, leak, threshold, reset and refractory tests.
- [x] [#21 — Excitatory and inhibitory synapse dynamics](https://github.com/aakashsiwach/artificial-human/issues/21). Merged and closed after PR #43 with signed, delayed transmission, bounded current and inhibition ablation.
- [x] [#22 — Event-driven spike propagation and synaptic delays](https://github.com/aakashsiwach/artificial-human/issues/22). Merged and closed after PR #43 with stable event ordering, queue limits and pending-event cancellation.
- [x] [#23 — First recurrent neural microcircuit](https://github.com/aakashsiwach/artificial-human/issues/23). Merged and closed after PR #43 with disconnected, feed-forward, recurrent, fixed-rate and inhibition-off controls; full biological fidelity is not claimed.

**Exit gate:** a deterministic runnable experiment shows individual neurons communicating through delayed excitatory/inhibitory synapses, recurrent activity, and stable decay—not just named population counters.

### Phase 2 — learning, regulation, representation, and memory

- [x] [#24 — Local synaptic plasticity and learning-rule comparisons](https://github.com/aakashsiwach/artificial-human/issues/24). Merged and closed.
- [x] [#25 — Neural homeostasis and bounded internal regulation](https://github.com/aakashsiwach/artificial-human/issues/25). Merged and closed.
- [x] [#26 — Population coding and distributed neural representations](https://github.com/aakashsiwach/artificial-human/issues/26). Merged and closed.
- [x] [#30 — Replace database-backed memory with bounded in-memory cognitive memory](https://github.com/aakashsiwach/artificial-human/issues/30). Merged and closed; the active runtime uses bounded in-memory cognitive stores.

**Exit gate:** independent, bounded RAM mechanisms learn, retrieve, associate, forget, and replay without an LLM or database; learning is compared against fixed-weight and simpler baselines.

### Phase 3 — perception and world modelling

- [x] [#27 — Permission-scoped sensory encoders](https://github.com/aakashsiwach/artificial-human/issues/27). Merged and closed; the current runtime uses synthetic events only.
- [x] [#28 — Multimodal integration and perceptual state estimation](https://github.com/aakashsiwach/artificial-human/issues/28). Merged and closed for the current synthetic task.
- [x] [#29 — Predictive world-state model and prediction-error loop](https://github.com/aakashsiwach/artificial-human/issues/29). Merged and closed with a bounded synthetic world model.

**Exit gate:** explicit, permission-scoped sensory streams produce uncertainty-aware neural state; predicted environment changes are compared with observed outcomes.

### Phase 4 — connected brain circuits, body, and action

- [x] [#31 — Episodic binding, replay, and consolidation circuits](https://github.com/aakashsiwach/artificial-human/issues/31). Merged and closed after PR #55.
- [x] [#32 — Action selection and reward learning](https://github.com/aakashsiwach/artificial-human/issues/32). Merged and closed after PR #56; this is a bounded tabular computational model, not a basal-ganglia replica.
- [x] [#33 — Forward prediction and adaptive control](https://github.com/aakashsiwach/artificial-human/issues/33). Merged and closed after PR #57; this is a bounded adaptive controller, not a cerebellar anatomical model.
- [x] [#34 — Simulated body and environment](https://github.com/aakashsiwach/artificial-human/issues/34). Merged and closed after PR #58; this is a deterministic scalar simulation, not a physical body.

**Exit gate:** met by PR #58's reproducible closed-loop simulated task. Integrated decoding of LIF population activity into action selection remains open under #38.

### Phase 5 — development and continuous organism runtime

- [x] [#35 — Unify prenatal and runtime substrate; extend postnatal development](https://github.com/aakashsiwach/artificial-human/issues/35). Merged and closed after PR #60.
- [x] [#36 — Run the organism as a continuous in-memory process](https://github.com/aakashsiwach/artificial-human/issues/36). Implemented in PR #61; closes on merge and green CI.

**Exit gate:** one coherent substrate follows a developmental clock and continues bounded internal neural, regulatory, memory, and predictive processes without incoming requests or LLM calls.

### Phase 6 — safe self-development and integration

- [ ] [#37 — Safe self-development through isolated code proposals, tests, and verified diffs](https://github.com/aakashsiwach/artificial-human/issues/37).
- [ ] [#38 — Integrate brain circuits into one executable artificial organism](https://github.com/aakashsiwach/artificial-human/issues/38).

**Exit gate:** the organism can identify a measured failure, propose a candidate code/test change in isolation, and provide reproducible evidence; human approval remains required before source changes are adopted.

### Phase 7 — evaluation and completion

- [ ] [#39 — Behavioural, neural, and organism-level evaluation gates](https://github.com/aakashsiwach/artificial-human/issues/39).
- [ ] [#40 — Verify the living-organism completion gate](https://github.com/aakashsiwach/artificial-human/issues/40).

**Exit gate:** each mechanism has fixed-seed tests, baselines, ablations, failure cases, resource bounds, limitations, and green CI. Issue #40 validates the complete system; merged PRs alone do not constitute completion.

## Historical foundations (already merged)

The earlier work remains useful as research baseline: modular organism composition, consent-aware SQLite memory, source-defined identity/knowledge, explicit text perception, attention and working memory, lexical prediction, executive planning, associative/relational software graphs, replay/consolidation proposals, owner-reviewed schema extraction, the prenatal scaffold, benchmark infrastructure, event bus, resource governor, immune/incident, and regeneration proposal modules.

These are not evidence of a neural brain or a no-database runtime. Reuse them where tests and interfaces fit the new architecture; migrate or retire them when they conflict with the runtime contract.

## Delivery rules

1. Work in dependency order. Complete one foundational issue and one substantial PR at a time.
2. Do not start the next branch while the current foundational PR is under review.
3. Make coherent commits, run local targeted/full checks before pushing, and avoid one-file commits that trigger repeated CI runs.
4. Do not mark work complete because it compiles, has a biological name, or passes only synthetic unit tests. Require a runnable demonstration and baseline comparison.
5. No dashboard, dataset adapter, external service, database feature, or unrelated agent framework may displace the neural substrate.
6. Do not merge a PR without the repository owner's explicit instruction.

## Scientific completion standard

For every mechanism, document the empirical constraint, competing explanations, computational hypothesis, abstraction level, omitted biology, discriminating task, baseline, ablation, failure cases, resource bounds, privacy/safety controls, and reproducible result. The project does not claim a literal biological brain, consciousness, sentience, or human-level general intelligence.
