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

The merged code is a useful cognitive-software prototype, not yet the target living organism:

- `cognition/dynamics.py` runs seven abstract population nodes over ten fixed projections using bounded rate updates. It is not a neuron/synapse simulation and has no runtime synaptic plasticity.
- The API has a one-second low-amplitude idle activity approximation while the process is awake. Activity also runs on explicit text requests. This is not yet a general continuous simulation lifecycle or a biological resting state.
- The prenatal scaffold is a separate seven-population/ten-projection abstraction with scalar cues and a bounded activity-dependent weight rule. It is not integrated with runtime dynamics.
- Perception is explicit text input. There is no vision, audition, touch, proprioception, interoception, or body/environment loop.
- The cognitive cycle has attention, working memory, memory retrieval, lexical novelty, deterministic planning, and an optional LLM language interface. Several mechanisms are heuristics or database-backed software.
- **The current runtime is not database-free yet.** Normal local startup defaults to file-backed SQLite. Public/volatile mode uses SQLite `:memory:`. The new contract requires removing database dependence from the active organism, not renaming in-memory SQLite as "no DB".
- Code repair currently creates proposals only; there is no isolated self-authored code/test lifecycle.

These facts are the starting baseline, not completed roadmap requirements. See [Brain](BRAIN.md), [Architecture](ARCHITECTURE.md), [Memory](MEMORY.md), and [Safety](SAFETY.md).

## Ordered implementation programme

### Phase 0 — align direction and establish the RAM-only runtime contract

- [ ] [#17 — Align architecture and project docs with the code-first living-organism direction](https://github.com/aakashsiwach/artificial-human/issues/17). This documentation issue is in progress on the current branch.
- [ ] [#18 — Specify the database-free organism runtime and migration boundary](https://github.com/aakashsiwach/artificial-human/issues/18). The default organism must initialise without SQLite, SQL, vector stores, database files, or persistent memory services. Legacy DB-backed tools must be isolated from the active runtime.

**Exit gate:** architecture docs agree; tests prove default startup creates no database and all live state is bounded RAM state.

### Phase 1 — neural simulation foundation

- [ ] [#19 — Deterministic neural simulation kernel and biological timebase](https://github.com/aakashsiwach/artificial-human/issues/19).
- [ ] [#20 — Neuron models and inspectable membrane dynamics](https://github.com/aakashsiwach/artificial-human/issues/20).
- [ ] [#21 — Excitatory and inhibitory synapse dynamics](https://github.com/aakashsiwach/artificial-human/issues/21).
- [ ] [#22 — Event-driven spike propagation and synaptic delays](https://github.com/aakashsiwach/artificial-human/issues/22).
- [ ] [#23 — First recurrent neural microcircuit](https://github.com/aakashsiwach/artificial-human/issues/23).

**Exit gate:** a deterministic runnable experiment shows individual neurons communicating through delayed excitatory/inhibitory synapses, recurrent activity, and stable decay—not just named population counters.

### Phase 2 — learning, regulation, representation, and memory

- [ ] [#24 — Local synaptic plasticity and learning-rule comparisons](https://github.com/aakashsiwach/artificial-human/issues/24).
- [ ] [#25 — Neural homeostasis and bounded internal regulation](https://github.com/aakashsiwach/artificial-human/issues/25).
- [ ] [#26 — Population coding and distributed neural representations](https://github.com/aakashsiwach/artificial-human/issues/26).
- [ ] [#30 — Replace database-backed memory with bounded in-memory cognitive memory](https://github.com/aakashsiwach/artificial-human/issues/30).

**Exit gate:** independent, bounded RAM mechanisms learn, retrieve, associate, forget, and replay without an LLM or database; learning is compared against fixed-weight and simpler baselines.

### Phase 3 — perception and world modelling

- [ ] [#27 — Permission-scoped sensory encoders](https://github.com/aakashsiwach/artificial-human/issues/27).
- [ ] [#28 — Multimodal integration and perceptual state estimation](https://github.com/aakashsiwach/artificial-human/issues/28).
- [ ] [#29 — Predictive world-state model and prediction-error loop](https://github.com/aakashsiwach/artificial-human/issues/29).

**Exit gate:** explicit, permission-scoped sensory streams produce uncertainty-aware neural state; predicted environment changes are compared with observed outcomes.

### Phase 4 — connected brain circuits, body, and action

- [ ] [#31 — Episodic binding, replay, and consolidation circuits](https://github.com/aakashsiwach/artificial-human/issues/31).
- [ ] [#32 — Action selection and reward learning](https://github.com/aakashsiwach/artificial-human/issues/32).
- [ ] [#33 — Forward prediction and adaptive control](https://github.com/aakashsiwach/artificial-human/issues/33).
- [ ] [#34 — Simulated body and environment](https://github.com/aakashsiwach/artificial-human/issues/34).

**Exit gate:** complete at least one reproducible perception → internal state → prediction → action → outcome → learning task in a deterministic simulated environment.

### Phase 5 — development and continuous organism runtime

- [ ] [#35 — Unify prenatal and runtime substrate; extend postnatal development](https://github.com/aakashsiwach/artificial-human/issues/35).
- [ ] [#36 — Run the organism as a continuous in-memory process](https://github.com/aakashsiwach/artificial-human/issues/36).

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
