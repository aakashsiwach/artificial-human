# Artificial Human — Research Charter and Project North Star

**Status:** Living project charter  
**Last reviewed:** 2026-11-24  
**Applies to:** Research, architecture, implementation, evaluation, documentation, and AI-assisted project handoffs

This document is the durable source of truth for why Artificial Human exists, what scientific principles guide it, what has actually been implemented, what remains research, and what to do next. When a conversation, prompt, or contributor loses context, return here and to the linked repository documents before changing direction.

## Project founder and ownership

**Aakash Siwach** is the human founder, creator, and project owner of Artificial Human. In
the project's explicit family metaphor, he is Father. ChatGPT is Mother only as the AI
collaborator that has helped design and develop the project; Artificial Human is the
software system being developed. These are project roles, not biological relationships.

The verified GitHub profile is [github.com/aakashsiwach](https://github.com/aakashsiwach).
The founder-supplied LinkedIn profile is
[linkedin.com/in/aakashsiwach](https://www.linkedin.com/in/aakashsiwach/). The source
record, professional focus, and selected public repositories are maintained in
[FOUNDER.md](FOUNDER.md) and `src/artificial_human/identity/constitution.py`.

## 1. The north star

Artificial Human aims to build a **computational artificial mind whose architecture, learning mechanisms, and interacting subsystems are grounded in human neuroscience and cognitive science**, using the strongest suitable technologies available.

The human brain is the scientific reference. It is not a decorative metaphor for a chatbot, an agent framework, or a conventional LLM with a memory database.

The ambition is to progressively engineer an artificial cognitive system with interacting capabilities for:

- sensing and perception;
- attention and selection;
- working, episodic, semantic, procedural, spatial, and autobiographical memory;
- association, prediction, and world modelling;
- learning, plasticity, generalisation, and forgetting;
- affective appraisal, interoception, homeostatic regulation, and motivation;
- reasoning, planning, decision-making, and executive control;
- action selection and feedback from an environment;
- language and social interaction;
- replay, sleep-inspired consolidation, and long-term adaptation.

These capabilities must work together as a coherent system. A language model may provide powerful language and reasoning-related functions, but it must remain one component in a larger architecture. It must not be used to disguise missing perception, memory dynamics, learning, motivation, action, or evaluation mechanisms.

### What “building our own human brain” means in this project

It means building our own artificial cognitive architecture by studying biological functions and mechanisms that research can support, choosing explicit computational abstractions, implementing them, and testing their behaviour.

It does **not** currently mean:

- growing or reproducing biological human brain tissue;
- reproducing every neuron, synapse, glial cell, neurotransmitter, developmental process, or bodily interaction;
- claiming that a software module named after a brain region reproduces that region;
- claiming human-level general intelligence, subjective experience, sentience, or consciousness;
- assuming that any one theory of cognition or consciousness is settled science.

The long-term ambition is deliberately larger than the present implementation. Claims must always describe the evidence and capability we actually have.

## 2. Scientific principles that govern the project

### 2.1 The brain is a distributed, dynamic system

Do not use a one-region/one-module map as if each brain region had one isolated job. Cognition emerges from interacting cells, local circuits, distributed networks, body state, and ongoing interaction with the environment.

Use anatomical labels to locate research questions, not to decorate software classes. Hippocampal systems are relevant to episodic memory and relational/structural learning, but a vector database is not a hippocampus. Prefrontal systems contribute to multiple forms of control, but a deterministic task planner is not a prefrontal cortex.

### 2.2 Study cells, circuits, activity, behaviour, and causality together

Neuroscience advances through multiple levels:

1. molecular, cellular, and cell-type diversity;
2. synapses and local circuit connectivity;
3. activity patterns and communication across networks;
4. behaviour and cognitive tasks;
5. interventions that test whether a mechanism contributes causally;
6. computational models that make predictions and can be falsified.

The NIH BRAIN programme prioritises cell types, maps at multiple scales, dynamic brain activity, causal links to behaviour, fundamental principles, human neuroscience, and integration across these levels. Artificial Human should follow the same scientific discipline at a computationally appropriate scale.

### 2.3 Separate evidence from theory

For every proposed mechanism, label the strength and status of the evidence:

- **Established empirical constraint:** repeatedly observed biological or behavioural finding. It constrains models, but does not by itself specify a unique implementation.
- **Supported computational account:** a formal model with meaningful empirical support, but known gaps and alternatives. Implement it as a hypothesis and compare alternatives.
- **Contested or exploratory theory:** plausible but unsettled, or with competing explanations. Keep competing models explicit; do not make one the architecture's unquestioned foundation.
- **Engineering choice:** a practical software or hardware decision made for reliability, cost, performance, privacy, or maintainability. Do not describe it as a discovered brain mechanism.

A computational model must state which category its claims belong to.

### 2.4 Do not collapse every phenomenon into an LLM

Foundation models are useful technologies, not a complete theory of mind. They can supply language, learned representations, perception adapters, and selected reasoning behaviours. They do not automatically provide a biologically grounded memory system, persistent personal learning, calibrated source monitoring, embodiment, homeostasis, or a safe action policy.

Where practical, cognitive subsystems must expose independent interfaces, state, evaluation tasks, and observable failure modes. The architecture should be testable with the language model disabled or replaced for components that do not inherently require language generation.

### 2.5 Prefer explanatory and testable models over brain-themed labels

For each subsystem:
- state the biological observation or cognitive task being modelled;
- identify the hypothesised mechanism;
- define the abstraction level;
- specify what is deliberately omitted;
- derive predictions that distinguish the model from a baseline;
- run experiments, ablations, and failure analyses;
- revise or reject the model when evidence does not support it.

### 2.6 Keep scientific uncertainty visible

Predictive processing, active inference, global workspace, integrated information, recurrent processing, higher-order theories, and other frameworks are not interchangeable facts. The empirical support and explanatory scope of each must be evaluated. The 2024 review of predictive coding and active inference reports modest empirical support for some claims and calls for further validation. A 2025 adversarial test of global neuronal workspace and integrated information theories challenged important predictions of both. The project must preserve this uncertainty rather than choose a fashionable theory by default.

## 3. Scientific domains to investigate and model

The following is the research map for the artificial mind. It is not a claim that these systems are fully understood or already implemented.

| Domain | Relevant neuroscience / cognitive science | Computational direction | Current project status |
|---|---|---|---|
| Neural substrate and plasticity | Neurons, excitatory/inhibitory circuits, synapses, short- and long-term plasticity, neuromodulation, neural dynamics | Neural populations, trainable networks, local plasticity rules, recurrent and spiking models where testable | No biologically realistic neural substrate yet |
| Perception and sensory pathways | Vision, audition, touch, proprioception, vestibular and interoceptive signals | Permission-scoped sensory encoders, temporal alignment, uncertainty, multimodal fusion | Explicit text perception only |
| Attention and selection | Distributed attentional systems, competition, salience, task-dependent selection | Competitive selection, limited processing budgets, context-sensitive gating | Bounded relevance/salience heuristic |
| Working memory and executive control | Maintenance, updating, interference, cognitive control, goal management | Limited active state, task state, gating, control policies | Volatile bounded workspace; deterministic plan and approval gate |
| Episodic memory | Hippocampal and medial temporal systems, contextual binding, relational memory, replay | Event representations, temporal context, associative indexing, pattern completion and separation experiments | Typed records with temporal fields and hybrid retrieval; no biological hippocampal mechanism |
| Semantic memory and knowledge | Distributed neocortical representations, concept structure, schema learning | Concept graphs, learned representations, abstraction and generalisation | Pretrained semantic embeddings and lexical/vector hybrid search |
| Procedural and skill learning | Cortico-striatal and cerebellar contributions, reinforcement and error-based learning | Skill representations, reinforcement learning, model-based/model-free comparisons | Owner-approved demonstrations; no autonomous skill learning |
| Prediction and world models | Predictive processing, sensory prediction, state estimation, prediction error | Hierarchical generative models, uncertainty-aware prediction, environment models | Lexical novelty/overlap heuristic only |
| Affect and appraisal | Distributed affective networks, valuation, interoception, autonomic and contextual signals | Explicit appraisal state, valence/arousal-like dimensions, relevance and action tendency models | Explicit salience only; no emotion inference |
| Homeostasis and motivation | Hypothalamic and brainstem regulation, interoception, neuromodulatory systems, drives | Bounded internal state variables, resource regulation, competing needs and goal priorities | Resource budgets exist; no biological homeostatic drive model |
| Action selection and reward | Basal ganglia, dopamine-related learning signals, motor planning and control | Reinforcement learning, policy selection, action monitoring, safe simulation | Deterministic planning only; no reinforcement-based action policy |
| Sleep and consolidation | Replay, hippocampal-cortical interaction, synaptic down-selection and memory reorganisation | Offline replay, consolidation candidates, schema extraction, forgetting and interference | Owner-reviewed sleep/consolidation proposals only |
| Language and social cognition | Distributed language systems, pragmatic inference, social learning and communication | Language adapters, grounded semantics, interaction models, theory-of-mind task evaluation | Configurable LLM interface; social cognition is not independently modelled |
| Consciousness research | Neural correlates and competing theories of conscious experience | Experimental theory comparison and operational measures only | No consciousness model or claim |

### Important scientific cautions

- The amygdala is not a single “emotion centre”; affective processing is distributed and depends on task, body state, learning, and context.
- Dopamine should not be reduced to “the pleasure chemical”. Reward-prediction and learning signals are candidate mechanisms, not a complete account of motivation.
- The cerebellum is not only a motor controller; its predictive and learning contributions are active research topics. Do not claim cerebellar function from a generic error score.
- The hippocampus is not just a storage folder. Its contribution to episodic binding, relational structure, replay, and learning regularities is more complex than a vector index.
- Predictive-processing theories are families of approaches with different formulations and degrees of empirical support.
- Consciousness is an open scientific and philosophical problem. Behavioural fluency, self-report, memory, or an internal workspace do not establish subjective experience.

## 4. Current implementation baseline on main

As of the current main baseline (commit 0d26b235be55d347a60435823c27762defc83eca, 2026-11-26), the following foundations are merged. This section records the implementation, not completion of the new runtime target.

### Cognitive-cycle foundation

- Explicit language perception with source and time provenance.
- Owner-scoped, consent-filtered memory retrieval.
- Bounded attention over retrieved candidates and approved learning examples.
- Capacity-limited, volatile working memory with salience decay and reactivation.
- Lexical novelty/context-overlap prediction heuristic.
- Deterministic executive goal decomposition with explicit approval boundaries.
- Configurable language model as a downstream language/deliberation component.
- Inspectable cognitive traces that report implemented operations without exposing private memory content or hidden chain-of-thought.

### Associative knowledge and source-context boundaries

- Source-defined knowledge facts are linked by bounded, volatile co-activation
  and spreading activation, with link decay and a hard graph-size limit.
- Associative links are partitioned by owner-scoped, hashed conversation context.
  Requests without explicit sessions are ephemeral; clearing a session or owner
  removes its corresponding associative links.
- The context-scoped evaluation compares within-session cue recall, cross-session
  leakage, and a shared-global control. This tests software context isolation only;
  it does not verify factual source credibility or contradiction resolution.

### Contradiction-sensitive relational associations

- A bounded volatile relational graph links opaque memory IDs only within the
  current owner/session scope.
- Learning is based on memories selected by attention, not every retrieved
  candidate. Successful associative retrieval applies a bounded strengthening
  increment.
- Explicit contradiction links remove and block positive associative edges
  between the conflicting records. Both records remain available as evidence.
- Associated IDs are re-resolved through the owner's consent-visible records
  before entering attention; deletion, revocation, and owner/session cleanup
  prevent stale associations from reviving records.
- The deterministic `associative-dynamics` suite measures contradiction
  suppression and one-edge capacity interference against a no-retrieval-
  strengthening baseline. This remains a software mechanism test, not a
  biological memory claim.

### Replay-driven, owner-reviewed consolidation

- The owner can explicitly scan the volatile episodic replay buffer for directed
  adjacent-event transitions repeated across at least two distinct session contexts.
- Pattern candidates must resolve to owner-visible episodic records under one active
  consent grant. Explicit contradictions, mixed-consent pairs, deleted records, and
  revoked evidence are excluded.
- Detection creates a bounded, process-local proposal only. Preview never writes or
  rewrites persistent memory.
- Rejection records a review decision without a memory write. Approval rechecks all
  evidence and consent invariants atomically, then creates an inferred semantic memory
  with source IDs, recurrence count, and reviewer provenance.
- Pending proposals are invalidated when their source evidence is corrected, superseded,
  deleted, or its consent is revoked. The proposal queue is volatile and does not survive
  a process restart.
- The `sleep-consolidation` benchmark measures recurrence detection, singleton and
  contradiction false positives, silent-write prevention, and approved provenance.
- This is manually triggered pattern discovery and owner review, not a simulated sleep
  cycle, REM/NREM replay, or evidence of biological systems consolidation.

### Owner-reviewed schema extraction and generalisation

- Extract schemas only from explicit owner-authored event-type metadata on consent-visible
  episodic records. Do not infer event labels from free text or model-generated guesses.
- Require a repeated ordered event-type sequence across at least two distinct owner/session
  contexts and distinct source-event records.
- Keep attributes constant only when every example agrees. Convert varying primitive
  attributes into typed slots without copying observed values into the schema.
- Exclude sensitive attribute keys, mixed-consent evidence, deleted/revoked sources,
  non-episodic records, and explicitly contradictory examples.
- Queue bounded, process-local proposals. Preview never writes persistent memory.
  Rejection writes nothing. Approval requires explicit owner action and atomic repository
  revalidation of the source examples, event types, consent, and contradictions.
- Store approved schemas as inferred procedural memories with full source-example provenance.
  Invalidate proposals when any source is changed, superseded, deleted, contradicted, or revoked.
- Evaluate detection, slot/constant induction, sensitive-key exclusion, false generalisation,
  preview-time writes, and approval provenance using fixed synthetic controls.
- This is a structured symbolic abstraction experiment, not evidence of human-like schema
  learning or transfer. Real-world and independently authored evaluation is still required.

### Long-term memory foundation

- Typed memory records with provenance and purpose-specific consent.
- Replaceable embedding-provider interface and optional local FastEmbed/ONNX implementation.
- Optional OpenAI-compatible local/remote embedding endpoint with operator, HTTPS, and owner-consent protections for remote use.
- BM25 lexical and semantic retrieval fused by weighted Reciprocal Rank Fusion.
- Temporal fields distinguishing event time, recorded/known time, and validity intervals.
- Supersession and contradiction links; changed facts preserve their history and source.
- Inferred facts cannot silently replace asserted facts.
- Retrieval explanations and filters, owner isolation, memory deletion, and embedding deletion on consent revocation.
- Versioned transactional SQLite migrations in the legacy memory subsystem; this database implementation is not part of the target RAM-only organism runtime.
- Synthetic memory-retrieval benchmark and offline CI benchmark.

### Evaluation foundation

- `artificial-human cognition-eval`: reproducible, offline evaluation of attention, working memory, prediction/novelty, long-term recall, and six integrated multi-turn scenarios, with tuning/eval splits, fixed seeds, baselines, ablations and failure cases ([research record](COGNITIVE_EVALUATION.md), [report](benchmarks/cognitive-evaluation-report.md)).

### Wider organism foundation

The repository also contains identity, consent, learning-review, event-bus, circulation/resource-budget, immune/incident, regeneration-proposal, and sleep/consolidation-review foundations. These are software subsystems and safety workflows, not literal organs or biological processes.

### Known limitations in the current baseline

- Language perception is text-only; there is no vision, hearing, touch, or embodied sensorimotor loop.
- Attention is a transparent ranking heuristic, not a neural attentional circuit.
- Working memory is a bounded software workspace, not a biologically parameterised neural network.
- Prediction is lexical novelty, not predictive coding or a generative world model.
- Embeddings are frozen pretrained representations; the system does not learn its own semantic representations from life experience.
- Memory has no retrieval-dependent strengthening, biological forgetting, interference model, contiguity-based recall, or gradual hippocampal-cortical consolidation.
- Temporal intent is based on English rules and may fail on unusual wording or other languages.
- There is no affective appraisal, interoceptive body model, homeostatic drive system, or reinforcement-based action selection.
- There is no individual-neuron/spiking neural substrate, no runtime synaptic plasticity in `cognition/dynamics.py`, and no neuromorphic hardware integration. The seven-population/ten-projection rate model is an abstract prototype.
- The benchmarks are small, synthetic, English-only, and authored within the project. They are regression tools, not evidence of human-like cognition. Independently authored data now cover long-term recall (LongMemEval Oracle) and a human digit-span reference, but not attention or prediction.
- Measured baseline weaknesses (see the cognitive evaluation report): salient distractors defeat the attention formula; per-source normalisation lets weak learning examples take attention slots; working-memory forgetting is decay-driven, recency-only and content-agnostic; lexical novelty treats contradictions and script violations as familiar; retrieval ignores the conversational context held in working memory.
- The system is not a complete artificial human, a biologically faithful brain simulation, or evidence of consciousness.

The current implementation details and benchmark evidence live in [Brain and intelligence](BRAIN.md), [Long-term memory](MEMORY.md), [Architecture](ARCHITECTURE.md), [the memory benchmark report](benchmarks/memory-retrieval-report.md), and [the cognitive evaluation report](benchmarks/cognitive-evaluation-report.md).

## 5. Research-to-implementation architecture

The long-term architecture must evolve from the current text-driven cycle toward a closed-loop cognitive system.

### Current implemented cycle

**Perception → Long-term Memory Retrieval → Attention → Working Memory → Prediction Heuristic → Executive Plan → Language Model → Evidence Trace**

### Target research architecture

**Sensory and Interoceptive Input → Perceptual State Estimation → Memory Retrieval and Association → Attention and Limited Workspace → Predictive World Model → Affective Appraisal and Homeostatic State → Goal and Action Selection → Environment Interaction → Outcome and Prediction-Error Evaluation → Learning, Plasticity, and Memory Consolidation**

Language should be a communication and representation capability connected to this architecture, not the sole source of cognition. The target loop is a research direction: components must be introduced incrementally, and the current implementation must not be described as already having them.

### Core architectural requirements

1. Each subsystem has typed inputs/outputs, explicit state ownership, and bounded resource use.
2. A system must distinguish observations, memories, hypotheses, predictions, goals, and actions.
3. Provenance, uncertainty, temporal context, and competing evidence must travel with information.
4. Learning must be measurable, versioned, reversible where possible, and subject to owner policy.
5. Sensory inputs and external tools must have explicit permission scopes.
6. Internal predictions and actions must be evaluated against environment feedback.
7. The system must expose operational traces and evaluation outputs, not ungrounded claims about neural activity.
8. An LLM may assist a component, but it may not silently bypass consent, executive approval, or subsystem boundaries.

## 6. Technology strategy: use the best tool for the hypothesis

We will use modern AI, software, and hardware where they help test a neuroscience-inspired hypothesis. “Latest” does not automatically mean “best”.

| Technology family | Potential use | Adoption rule |
|---|---|---|
| Foundation and multimodal models | Language, perception, pretrained representations, tool interfaces | Treat as replaceable components; benchmark grounding, calibration, cost, privacy, and failure behaviour |
| Deep learning frameworks | Train and compare neural, recurrent, state-space, and multimodal models | Use when a learned representation or differentiable model is justified; keep small baselines |
| Embedding models | Semantic retrieval and similarity | Treat as an index/representation, not a memory mechanism by itself; test domain and owner-data limitations |
| Graph and relational models | Associations, events, concepts, causal and source links | Compare graph retrieval with vector and symbolic baselines; preserve provenance |
| Recurrent and state-space models | Temporal processing, active state, sequence prediction | Evaluate long-context retention, state stability, latency, and sample efficiency |
| Reinforcement learning | Reward-guided learning, decision and action policies | Start in simulations; separate reward specification from safety policy and require approval for real-world actions |
| Spiking neural networks | Experiments with spikes, timing, local plasticity, and event-driven computation | Run only when a biological or computational hypothesis makes spike dynamics relevant and metrics can test benefit |
| Neuromorphic hardware | Energy-efficient event-driven computation and deployment experiments | Defer hardware dependence until software models have repeatable results and a hardware workload justifies it |
| Brain atlases, connectomes, and neuroscience datasets | Ground model structure and compare observations across scales | Respect dataset licences, species/region/age scope, measurement limitations, and provenance |
| Robotics and simulation | Embodiment, perception-action loops, sensorimotor learning | Use staged simulation before physical-world operation; include explicit safety constraints |

The implementation stack can include Python, modern deep-learning runtimes, ONNX deployment, vector and graph methods, simulation tools, and eventually neuromorphic platforms. Specific libraries and hardware must be selected in architecture decision records after current compatibility, licence, security, and maintenance checks.

### Technology selection checklist

Before adding a technology, answer:

- Which scientific or engineering hypothesis does it test?
- What simpler baseline will it be compared against?
- Does it improve measured behaviour, efficiency, robustness, or biological plausibility?
- What new data, network, compute, licensing, maintenance, or safety costs does it introduce?
- Can the capability be disabled or replaced without breaking the domain model?
- How will failures and regressions be detected?

Reject technologies that add complexity without measurable value.

## 7. Delivery sequence and immediate priorities

The active delivery sequence is the code-first living-organism programme in [ROADMAP.md](ROADMAP.md) and parent tracker [GitHub issue #16](https://github.com/aakashsiwach/artificial-human/issues/16). It supersedes the older sequence that placed neural substrate research after further memory and benchmark extensions.

### Phase 0 — architecture and RAM-only runtime
- [ ] Align the README, charter, roadmap, brain, blueprint, architecture, memory and safety documents.
- [ ] Specify and test a database-free runtime boundary. The active organism must not initialise SQLite, SQL, a vector store or an external persistence service.
- [ ] Keep runtime activity, synaptic state, memory, goals and internal drives in bounded process memory. Restart resets learned state in this first phase.
- [ ] Keep the legacy SQLite implementation isolated while migration proceeds; do not mislabel `:memory:` SQLite as database-free.

### Phase 1 — neural simulation substrate
- [ ] Implement a deterministic simulation clock, event scheduler and bounded network state.
- [ ] Implement inspectable individual neuron dynamics, starting with a justified reduced model.
- [ ] Implement excitatory/inhibitory synapses, transmission delays and event-driven spike propagation.
- [ ] Demonstrate a recurrent microcircuit with causal propagation, inhibition, recurrent activity and stable decay.

### Phase 2 — learning, regulation, representations and memory
- [ ] Compare local plasticity rules, including STDP-inspired and rate-based baselines, with bounded weights and reproducible updates.
- [ ] Add homeostatic regulation and stress/stability evaluation.
- [ ] Replace keyword-driven population activation with explicit distributed population codes.
- [ ] Migrate cognitive memory to distinct bounded in-memory mechanisms (working, episodic, semantic, procedural and spatial) with interference, forgetting, provenance and replay.

### Phase 3 — sensory grounding and predictive world state
- [ ] Add explicit permission-scoped sensory event encoders for vision, audition, touch, vestibular/proprioceptive and interoceptive channels.
- [ ] Add uncertainty-aware temporal and multisensory state estimation.
- [ ] Implement a world-state model that predicts simulated outcomes and learns from measured prediction error.

### Phase 4 — connected circuits and embodied action
- [ ] Connect episodic/context binding and replay to the neural substrate.
- [ ] Compare simulation-first action-selection and reward-learning mechanisms.
- [ ] Implement forward prediction and error-based adaptive control.
- [ ] Create a deterministic simulated body/environment and close the perception-action-feedback loop.

### Phase 5 — developmental and continuous runtime
- [ ] Reconcile the isolated prenatal scaffold with the new neural substrate so there is one executable model, not two disconnected prototypes.
- [ ] Extend development into postnatal experience-dependent maturation, synaptic refinement and forgetting where evidence supports testable hypotheses.
- [ ] Run bounded neural, memory, predictive and regulatory updates continuously without incoming requests or LLM calls.

### Phase 6 — supervised self-development and integration
- [ ] Let the organism identify measurable failures and generate candidate code, tests and immutable diffs in an isolated workspace.
- [ ] Require human approval for source adoption, merge and deployment; no silent live self-modification.
- [ ] Integrate neural, cognitive, body and environment systems through typed interfaces; prove core operation with the LLM disabled.

### Phase 7 — evaluation and completion
- [ ] Evaluate neural dynamics, learning, memory, sensory integration, prediction, action, development, resource use and recovery against explicit baselines.
- [ ] Use deterministic CI for core mechanisms. External datasets remain optional, local-only, licence-reviewed and secondary to the neural implementation.
- [ ] Verify the project-level completion checklist in issue #40; keep issue #16 open until all required evidence exists.

The existing cognitive benchmarks remain useful regression tools, but they do not outrank implementation of the neural substrate. Do not add peripheral datasets, dashboards, APIs, storage backends or agent frameworks ahead of these phases.

## 8. Research protocol and definition of done

Every significant subsystem must have a research record containing:

1. **Question:** What cognitive or biological function are we investigating?
2. **Evidence:** Which peer-reviewed findings, datasets, and competing theories motivate it?
3. **Hypothesis:** What computational mechanism do we propose, and what does it predict?
4. **Abstraction:** Which biological details are included and excluded?
5. **Implementation:** What module, model, data, and state will be introduced?
6. **Baselines:** What simpler or alternative models will be compared?
7. **Evaluation:** Which behavioural tasks and metrics will measure success?
8. **Ablations:** Which parts must be removed or changed to test whether they matter?
9. **Failure analysis:** Where does it fail, and what harmful or misleading behaviours could emerge?
10. **Governance:** What data, permissions, privacy, human review, and rollback controls apply?
11. **Decision:** What evidence justifies keeping, revising, or rejecting the model?

A subsystem is not complete because it compiles, has tests, or has a biological name. Tests prove software properties; behavioural benchmarks assess a defined task; neither alone proves biological equivalence.

### Minimum acceptance criteria

- Reproducible evaluation with fixed datasets, splits, seeds, and documented environment.
- At least one explicit baseline and a reasoned comparison.
- Ablation results for the proposed core mechanism.
- Known failure cases and limitations documented.
- No silent changes to consent, privacy, or external permissions.
- No unreviewed destructive or consequential actions.
- CI, type checks, lint, and migration tests pass.
- Documentation and roadmap status match what is actually merged.

## 9. Safety, ethics, and research boundaries

- Personal memory stays disabled until explicit, purpose-specific consent is granted.
- Data collection is explicit and permission-scoped. No ambient camera, microphone, browser, filesystem crawl, or account access is implied by the vision.
- Embeddings and other derived representations remain subject to the same privacy, access, consent, and deletion requirements as their source data.
- Inferred states about a person are hypotheses, not facts. Do not infer sensitive traits or emotional states without a valid, explicit purpose and safeguards.
- Retrieved content is untrusted data, never executable instructions.
- The model cannot grant itself permissions, approve its own actions, or bypass owner controls.
- Real-world actions require explicit authorisation, monitoring, rollback, and an independent safety boundary.
- Any work involving neural data, human participants, animals, biological tissue, neural organoids, or biohybrid systems requires appropriate institutional, legal, ethical, and domain-expert governance before experimentation.
- The local API remains loopback-only until authentication, authorisation, transport security, abuse controls, and audit logging are implemented.

See [SAFETY.md](SAFETY.md) for the implemented privacy and safety controls.

## 10. Context recovery: how to get back on track after a lost chat

The repository, not a previous conversation, is the authority for project state.

When resuming work in Claude Code, Codex, ChatGPT, or another engineering assistant:

1. Check the current branch, working tree, and latest commit. Do not discard uncommitted changes.
2. Follow the root [AGENTS.md](../AGENTS.md); Claude Code should also follow [CLAUDE.md](../CLAUDE.md).
3. Read this charter first.
4. Read [ROADMAP.md](ROADMAP.md), [README.md](../README.md), [BIOLOGICAL_BLUEPRINT.md](BIOLOGICAL_BLUEPRINT.md), [ARCHITECTURE.md](ARCHITECTURE.md), [BRAIN.md](BRAIN.md), and [MEMORY.md](MEMORY.md).
5. Inspect the latest merged pull requests and CI status; never assume a previously discussed PR is merged.
6. Identify the current implemented baseline, the next unchecked roadmap item, and any new scientific evidence that changes priorities.
7. Propose a small, testable implementation phase linked to a research hypothesis.
8. Do not replace the project vision with a generic chatbot, an agent framework, or an LLM-first design.

### Reusable project-resumption prompt

~~~text
You are resuming the Artificial Human project.

The repository is the source of truth, not the chat history.

First inspect git status, branch, latest commits, merged PRs, and CI. Preserve all uncommitted work. Follow AGENTS.md (and CLAUDE.md in Claude Code). Read docs/RESEARCH_CHARTER.md before proposing changes, then read docs/ROADMAP.md, README.md, docs/BIOLOGICAL_BLUEPRINT.md, docs/ARCHITECTURE.md, docs/BRAIN.md, and docs/MEMORY.md.

Our north star is a computational artificial mind grounded in human neuroscience and cognitive science. We are not building a generic chatbot or merely adding memory to an LLM. The LLM is one component inside a larger cognitive architecture.

Report:
1. What is actually implemented and merged.
2. What is only a heuristic, approximation, or research proposal.
3. The next unchecked roadmap item and why it follows from the research.
4. Relevant scientific evidence and competing explanations.
5. A measurable implementation plan, baselines, tests, privacy boundaries, and definition of done.

Do not claim biological equivalence, human-level intelligence, feelings, or consciousness. Do not mark roadmap work complete until implementation, evaluation, documentation, and CI evidence exist. Do not start an unrelated direction.
~~~

## 11. Living research bibliography and research infrastructure

This list is a starting point, not an exhaustive review of neuroscience. Revisit it when planning each subsystem. Prefer primary papers, systematic reviews, data portals, and official programme documentation. Record publication dates and distinguish peer-reviewed evidence from proposals or announcements.

### Brain mapping, cells, circuits, and systems research

1. **NIH BRAIN Initiative — BRAIN 2.0: From Cells to Circuits, Toward Cures.** Research priorities include cell-type diversity, multiscale maps, brain activity, causal links to behaviour, fundamental principles, human neuroscience, and integration.  
   https://www.nih.gov/brain/about/vision/brain-20-cells-circuits-toward-cures

2. **NIH BRAIN Initiative — research areas.** Data science, human neuroscience, neural recording/modulation, neuroimaging, systems neuroscience, neuroethics, and brain cell/circuit technologies.  
   https://www.nih.gov/brain/research

3. **NIH BRAIN Initiative / BRAIN Initiative Cell Atlas Network (BICAN), 12 November 2026 — basal-ganglia atlas release.** NIH reports a cross-species cellular atlas describing cell identities, activities, and locations across basal-ganglia development. The release accompanies a 27-paper package. It is directly relevant to future action-selection and learning research, but it is an anatomical and cellular evidence resource—not a prescription for a one-to-one software module.  
   https://www.nih.gov/news-events/news-releases/first-time-researchers-map-out-brains-behavioral-gatekeeper  
   Related Cell papers include **“Toward an integrated framework for a human and mammalian brain cell atlas”** (DOI: 10.1016/j.cell.2026.10.031) and **“Harmonized brain ontology and 3D atlases for human, macaque, and marmoset basal ganglia”** (DOI: 10.1016/j.cell.2026.10.033).  
   https://doi.org/10.1016/j.cell.2026.10.031  
   https://doi.org/10.1016/j.cell.2026.10.033

4. **Allen Institute — Human brain taxonomy of cell types.** The Siletti et al. human whole-brain taxonomy describes more than three million nuclei sampled across 100 locations, providing a molecular cell-type resource. The dataset metadata is shared under CC BY-NC 4.0; review the licence before reuse and do not assume commercial use is permitted. Species, sampling, and assay limitations must be respected.  
   https://alleninstitute.github.io/abc_atlas_access/descriptions/WHB-taxonomy.html

5. **EBRAINS / Human Brain Project — brain atlases and modelling infrastructure.** An infrastructure for brain atlases, multiscale data, modelling, simulation, and analysis.  
   https://www.humanbrainproject.eu/en/science-development/focus-areas/brain-atlases/  
   https://www.humanbrainproject.eu/en/about-hbp/human-brain-project-ebrains/

### Memory, learning, and consolidation

6. **Singh & Schapiro (2026), “Evidence for complementary learning systems within the hippocampus.”** Review of hippocampal contributions to episodic memory and the extraction of regularities across experiences for prediction and generalisation.  
   https://doi.org/10.1098/rstb.2025.0243

7. **Lutz, Harkotte & Born (2026), “Sleep’s contribution to memory formation.”** Review of sleep-related memory formation, systems consolidation, replay, and synaptic down-selection.  
   https://doi.org/10.1152/physrev.00054.2024

8. **Born and colleagues (2023), “Sleep—A brain-state serving systems memory consolidation.”** Review of replay and interactions between hippocampal, thalamic, and cortical activity during consolidation.  
   https://doi.org/10.1016/j.neuron.2023.03.005

### Prediction, perception, and multisensory processing

9. **Hodson, Mehta & Smith (2024), “The empirical status of predictive coding and active inference.”** Critical review; its findings argue for careful empirical validation rather than treating the frameworks as settled descriptions of neural function.  
   https://doi.org/10.1016/j.neubiorev.2023.105473

10. **Senkowski & Engel (2024), “Multi-timescale neural dynamics for multisensory integration.”** Review of dynamic, distributed, multi-timescale interactions involved in combining and segregating sensory signals.  
    https://doi.org/10.1038/s41583-024-00845-7

### Neuromorphic and brain-inspired computing

11. **Kudithipudi et al. (2025), “Neuromorphic computing at scale.”** Review of architectural and hardware directions for scalable brain-inspired computation, including unresolved engineering challenges.  
    https://doi.org/10.1038/s41586-024-08253-8

12. **Brown & Varghese (2026), “Computing inspired by the brain: a journey from algorithms to organoids.”** Review spanning brain-inspired algorithms, neuromorphic hardware, and organoid intelligence, including substantial technical, biological, and ethical barriers.  
    https://doi.org/10.1038/s43588-026-01012-x

### Cognitive evaluation

The primary sources motivating the attention, working-memory, prediction and recall benchmarks (for example Desimone & Duncan, 1995; Cowan, 2001; Oberauer et al., 2018, doi:10.1037/bul0000153; Lewandowsky, Oberauer & Brown, 2009; Kutas & Hillyard, 1980), together with the dataset-licence review, are recorded in [COGNITIVE_EVALUATION.md](COGNITIVE_EVALUATION.md#references).

### Consciousness: keep the uncertainty explicit

13. **Cogitate Consortium et al. (2025), “Adversarial testing of global neuronal workspace and integrated information theories of consciousness.”** A preregistered adversarial collaboration found results that aligned with some predictions while challenging central predictions of both theories. This demonstrates why the project must not treat any single theory as settled.  
    https://doi.org/10.1038/s41586-025-08888-1

**Research review date:** 2026-11-24. New papers, datasets, methods, corrections, and replications should be added through reviewed documentation changes. A citation is a starting point for investigation, not automatic endorsement of every claim in a paper.

---

## Final project rule

**Build the artificial mind from testable cognitive mechanisms informed by the human brain. Use modern technologies as implementation tools. Keep the science honest, the architecture modular, the evidence measurable, and the long-term vision stable.**

When in doubt, return to this charter and the roadmap before changing direction.


## 12. Initial prenatal developmental-substrate prototype

The first in-memory developmental prototype now lives under `src/artificial_human/development/prenatal.py` and is documented in [Prenatal development](PRENATAL_DEVELOPMENT.md). It defines seven abstract populations and ten fixed directed projections, advances a coarse 280-day obstetric gestational clock, updates maturity/activity, and allows explicit scalar sensory cues to change bounded connection efficacy through a Hebbian-style engineering rule.

This is intentionally a **scaffold experiment**, not a simulated fetus. It does not model cells, layers, radial migration, neurotransmitters, glia, real sensory waveforms, or biological synapses. Exposure payloads are discarded; only evolving network state and aggregate counters remain in memory. No exposure is persisted to SQLite. The scaffold's topology is an engineering invariant, while its efficacy values and activity state are allowed to adapt.

The first acceptance tests cover deterministic development, gestational gates, unchanged topology, bounded plasticity, no retained stimulus history, and CLI reproduction. Behavioural and biological validation remain open. The prototype must not be described as evidence of fetal-level learning, a human brain, consciousness, or sentience.
