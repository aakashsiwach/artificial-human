# Architecture — code-first living organism

## Project target

Artificial Human is being developed as a continuously advancing computational organism. Its brain is executable source-defined code with explicit state transitions and bounded RAM state. It is not an LLM wrapped around a database.

The [Research Charter](RESEARCH_CHARTER.md) defines evidence standards; the [Roadmap](ROADMAP.md) defines ordered delivery; parent issue [#16](https://github.com/aakashsiwach/artificial-human/issues/16) tracks completion.

## Current baseline — honest status

The repository is a Python modular monolith with FastAPI, a cognitive software pipeline, a bounded RAM-only active memory adapter, a legacy SQLite repository, and a first individual-neuron simulation package.

- Local and public `ArtificialOrganism` construction now use the same RAM-only runtime contract. The active path creates no database file and does not call SQLite; the legacy SQLite implementation is lazy-loaded for explicit legacy operations only.
- `neural/simulation.py` implements deterministic discrete simulation time, LIF membrane state, refractory periods and spike timestamps, excitatory/inhibitory synapses, decaying postsynaptic currents, delayed event ordering, cancellation, bounded queues/currents, pause/resume/single-step/reset and in-memory snapshot/restore.
- `neural/circuits.py` defines a small recurrent four-neuron circuit. It is an engineering experiment, not an anatomical model; weights are fixed.
- The API idle task advances the LIF circuit while the process is awake. The existing seven-population rate engine remains a separate comparison baseline.
- Perception is explicit text. Vision, audition, touch, proprioception, interoception, predictive world state and a body/environment loop do not yet exist.
- The cognitive cycle still uses attention, working memory, lexical novelty, deterministic planning and an optional LLM language interface. The RAM memory repository keeps consented typed records, but full distinct cognitive-memory dynamics remain #30.

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
`memory/volatile_repository.py`: bounded, consent-aware RAM storage used by the active organism. `memory/repository.py` retains the legacy SQLite implementation and is lazy-loaded for explicit compatibility/evaluation tools only. The active RAM adapter does not yet satisfy #30's distinct memory-system, interference, forgetting and strengthening requirements.
- `cognition/*`: existing attention, working memory, prediction and executive software are useful baselines, not proof of neural implementation.
- `brain/*`: optional language and planning adapters. The core organism must work without a configured provider.
- `circulation/*`, `metabolism/*`, `immune/*`, `regeneration/*`, `sleep/*`: bounded software infrastructure, not literal organs.

## API and tools

CLI and API are control surfaces over one organism runtime; they must not create independent brains or own simulation state. The kernel is testable without either interface. The API remains loopback-only until authentication, authorisation, TLS, abuse controls and audit logging are implemented.

## Self-development boundary

The organism may identify a measured failure and prepare a candidate patch/test package in an isolated workspace. It cannot directly edit the running process, overwrite protected source, execute unrestricted commands, access secrets, grant permissions, merge or deploy. Candidate diffs, test evidence, resource use, expiry and rollback information require human review.

## Control Room connectivity and owner access

The Vercel Control Room proxies `/api/*` requests to the Render API on the same origin. The browser no longer requires a manually entered API URL or bearer token. Owner access is a single-account password fallback plus optional WebAuthn passkeys, with signed HttpOnly cookies and no database. Passkey private keys and biometric matching remain local to the authenticator. Camera preview is a separate opt-in browser feature and is not yet connected to a sensory encoder.

A browser session can restore automatically while its cookie is valid, but this does not keep a free Render instance awake. Free-tier suspension pauses the process-local neural loop; always-on execution requires paid compute.
