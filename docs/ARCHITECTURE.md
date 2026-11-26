# Architecture — code-first living organism

## Project target

Artificial Human is being developed as a continuously advancing computational organism. Its brain is executable source-defined code with explicit state transitions and bounded RAM state. It is not an LLM wrapped around a database.

The [Research Charter](RESEARCH_CHARTER.md) defines evidence standards; the [Roadmap](ROADMAP.md) defines ordered delivery; parent issue [#16](https://github.com/aakashsiwach/artificial-human/issues/16) tracks completion.

## Current baseline — honest status

The merged repository is a Python modular monolith with FastAPI, a cognitive software pipeline and a legacy SQLite memory subsystem.

- Normal local `ArtificialOrganism` construction defaults to file-backed SQLite.
- Public/volatile mode uses SQLite `:memory:`; this is process-local but remains a database.
- `cognition/dynamics.py` has seven abstract populations and ten fixed projections, performs bounded rate updates, and does not model individual neurons, spike timing, synaptic delays or runtime plasticity.
- The API lifespan runs a low-amplitude idle activity approximation once per second while the process is awake. It is not a complete simulation clock or evidence of biological resting-state activity.
- The prenatal scaffold is a separate experimental abstraction and is not integrated with runtime dynamics.
- Perception is explicit text; the current world model, sensory body and simulated action loop do not exist.
- The cognitive cycle has attention, working memory, memory retrieval, lexical novelty, deterministic planning and an optional LLM language interface. Several mechanisms are heuristics or database-backed software.

The current implementation is **not yet database-free**. Phase 0 issue #18 must establish and test that boundary before active runtime migration.

## Target runtime layers

1. **Simulation kernel:** deterministic simulation time, scheduler, event ordering, pause/resume/reset, bounded work and inspectable snapshots.
2. **Neural substrate:** neuron models, membrane state, excitatory/inhibitory synapses, delays, recurrent connectivity, spike propagation and local plasticity.
3. **Network/circuit models:** distributed population coding, recurrent microcircuits, context/episodic binding, predictive/error-correction and action-selection mechanisms. These are computational hypotheses, not anatomical replicas.
4. **Cognitive processes:** perception, attention, working memory, episodic/semantic/procedural/spatial memory, prediction, appraisal, homeostasis, motivation, planning and learning.
5. **Body and environment:** explicit sensory events, internal state, simulated body/action space, environmental transitions, feedback and outcome evaluation.
6. **Organism lifecycle:** continuous ticks, developmental time, resource governance, stability monitoring, bounded scheduling, snapshots and shutdown.
7. **Optional language interface:** replaceable LLM adapter. Core neural, memory, learning and simulated action mechanisms must work when it is disabled.
8. **Supervised self-development:** isolated code/test proposals and immutable diffs, with human approval before source adoption or deployment.

## Runtime state and persistence contract

During this programme:
- Source code defines architecture, stable identity, initial parameters, model rules and reviewed foundational knowledge.
- Live neural state, synaptic weights, activity traces, memory representations, goals and internal drives are bounded in-process data structures.
- There is no active SQL/SQLite, database, external vector store, remote persistence service or disk-backed runtime state.
- Restart resets experience and learned parameters. Durable updates may later be exported as reviewed source artefacts; automatic persistence is not part of this phase.
- State ownership, bounds, forgetting/eviction, concurrency, snapshot and reset semantics are explicit.
- Legacy SQLite code stays isolated until migration and compatibility tests prove it can be retired. SQLite `:memory:` does not satisfy the no-database requirement.

## Information flow

```text
explicit sensory/event input
    -> modality encoder and uncertainty
    -> neural activity and recurrent circuit processing
    -> attention / working state / memory association
    -> world-state prediction and internal appraisal
    -> bounded goal and action selection
    -> simulated body and environment
    -> observed outcome and prediction error
    -> local plasticity, memory update and consolidation
```

The LLM may assist language comprehension/generation at an adapter boundary. It must not replace neural dynamics, learn synaptic parameters, bypass executive/safety policy, or be required for the organism to advance its internal state.

## Scientific and engineering requirements

Every major mechanism must specify empirical motivation, competing explanations, computational hypothesis, abstraction level, omitted biology, typed interface, baseline, ablation, task, failure cases, resource limits and safety boundary.

Brain-region names are research pointers, not module design instructions. A module is not a hippocampus, cortex or basal ganglia simply because it has that name.

## Existing modules and migration

- `cognition/dynamics.py`: legacy abstract rate-based population experiment. Replace or adapt only after the neuron/synapse kernel demonstrates equivalent or improved behaviours.
- `development/prenatal.py`: separate seven-population/ten-projection scaffold. It must ultimately share the neural substrate rather than remain a competing engine.
- `memory/*`: current SQLite repository, hybrid retrieval, temporal fields and consent-aware index. This is legacy infrastructure to migrate or isolate; do not expand its database dependence.
- `cognition/*`: existing attention, working memory, prediction and executive software are useful baselines, not proof of neural implementation.
- `brain/*`: optional language and planning adapters. The core organism must work without a configured provider.
- `circulation/*`, `metabolism/*`, `immune/*`, `regeneration/*`, `sleep/*`: bounded software infrastructure, not literal organs.

## API and tools

CLI and API are control surfaces over one organism runtime; they must not create independent brains or own simulation state. The kernel is testable without either interface. The API remains loopback-only until authentication, authorisation, TLS, abuse controls and audit logging are implemented.

## Self-development boundary

The organism may identify a measured failure and prepare a candidate patch/test package in an isolated workspace. It cannot directly edit the running process, overwrite protected source, execute unrestricted commands, access secrets, grant permissions, merge or deploy. Candidate diffs, test evidence, resource use, expiry and rollback information require human review.