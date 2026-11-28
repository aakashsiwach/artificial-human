# Architecture — code-first living organism

## Project target

Artificial Human is a research software prototype for a continuously advancing,
source-defined computational organism. The active runtime uses explicit neural,
memory, prediction and simulated-action mechanisms with bounded RAM state. It is
not an LLM wrapped around a database.

The [Research Charter](RESEARCH_CHARTER.md) defines evidence standards; the
[Roadmap](ROADMAP.md) defines ordered delivery; parent
[issue #16](https://github.com/aakashsiwach/artificial-human/issues/16) tracks
the complete programme.

## Active composition root

`ArtificialOrganism` is the process-local composition root. It owns one
`OrganismLoop` and one `ContinuousOrganismRuntime`. The public
`neural_simulation` reference aliases the developmental LIF simulation owned
by that loop; it is not a second network.

The active continuous path is:

```text
ArtificialOrganism
  └── ContinuousOrganismRuntime
        └── OrganismLoop
              ├── synthetic sensory event / population encoder
              ├── bounded episodic memory
              ├── world-state predictor and error feedback
              ├── bounded action learner
              ├── simulated environment
              ├── homeostatic and forward-control state
              └── DevelopmentalProgram
                    └── deterministic LIF neurons and synapses
```

The request-driven `CognitiveArchitecture` remains a separate interface for
owner-scoped personal memory retrieval, attention, working memory, planning and
optional language generation. Its older seven-population
`cognition/dynamics.py` engine remains a request-path comparison component;
the continuous runtime does not advance it as a second autonomous brain.
The episodic neural engram circuit is connected to the consent-scoped cognitive
memory path. These are current software layers, not claims of one-to-one
biological regions.

## Neural and developmental substrate

`development/program.py` provides the active 12-neuron LIF developmental
network. It includes:

- inspectable membrane potential, threshold, reset, refractory state and spikes;
- excitatory/inhibitory synapses, delayed event propagation and bounded queues;
- stage-dependent synaptogenesis and bounded connectivity;
- membrane/refractory maturation and conduction-delay refinement as explicit
  computational proxies;
- bounded STDP-inspired plasticity, homeostatic regulation and functional
  pruning;
- prenatal-day and postnatal-model-unit clocks with inspectable snapshots;
- a paired-cue learning task compared with frozen-weight and mature-from-start
  baselines.

The default continuous loop starts from a prenatal-280/postnatal-500 synthetic
fixture so its test modalities are available. This is an engineering fixture,
not a clinical prediction or a person's age. The earlier
`development/prenatal.py` seven-population scaffold and
`neural/circuits.py` four-neuron microcircuit remain reference/baseline
experiments. The active developmental program is not an anatomical replica.

`neural/episodic.py` adds sparse, process-keyed engram features, context
binding, partial-cue completion, inhibitory competition, interference and
retrieval-dependent strengthening. Event text remains in the owner-scoped,
consent-governed in-memory repository; the neural engram index does not retain
raw text.

## Continuous runtime and clocks

`ContinuousOrganismRuntime` owns the deterministic tick function and an
optional asynchronous wall-clock scheduler. The API lifespan starts and stops
the composition-owned runtime. The CLI uses the same runtime interface for
accelerated manual simulation.

Three clocks remain independent:

| Clock | Advances when | Unit |
| --- | --- | --- |
| Neural simulation | LIF experiences and simulation steps execute | Configured neural simulation step |
| Developmental model | An explicit development command is issued | Prenatal days or postnatal model units |
| Wall clock | The asynchronous scheduler waits between ticks | Seconds |

Wall-clock uptime does not silently age the developmental model. Manual
`run_ticks(n)` executes deterministic accelerated ticks without sleeping.
When no external event is queued, the organism loop generates an internal
proprioceptive event from its simulated environment and continues prediction,
action, feedback and memory updates without an HTTP request or LLM call.

### Lifecycle and bounds

Default runtime limits:

| Resource | Bound |
| --- | ---: |
| Pending sensory events | 128 |
| Events processed per tick | 8 |
| Declared operations per tick | 8 |
| Per-tick wall-time budget | 0.25 seconds |
| Spikes inspected per tick | 256 |
| Recent lifecycle trace | 128 ticks |
| Synthetic episodic memory | 128 episodes |
| Simulated environment episode | 100 actions |
| Neural pending-event queue | 8,192 events |
| Manual accelerated batch | 10,000 ticks |

The event queue uses drop-newest backpressure. Queue rejection, tick-budget
exhaustion, excessive spike activity or an over-budget tick marks health as
degraded and is visible in snapshots. The per-tick `ResourceGovernor` limits
declared operations and elapsed time; it is not an operating-system CPU or
memory monitor. Reset requires the background scheduler to be stopped and
clears the runtime's queue, synthetic episodic state, neural state, counters
and bounded trace.

See [Continuous runtime](CONTINUOUS_RUNTIME.md) for the exact API, lifecycle
transitions, default fixture and verification protocol.

## Runtime state and persistence contract

During this programme:

- Source code defines architecture, stable identity, initial parameters,
  developmental rules and reviewed foundational knowledge.
- Live neural activity, synaptic weights, traces, synthetic episodic memory,
  working state, goals and internal regulation remain bounded in process RAM.
- The active organism runtime does not require SQL/SQLite, a database, external
  vector store, remote persistence service or disk-backed runtime state.
- Restart resets runtime experience and learned parameters. Durable learning
  may later be exported as reviewed source artefacts; automatic persistence is
  not part of this phase.
- Personal memory remains consent-scoped and owner-controlled in its separate
  in-memory repository. Synthetic loop memory is not a substitute for personal
  memory.
- Legacy SQLite code remains isolated for explicit compatibility and evaluation
  tools. SQLite `:memory:` is not considered a database-free architecture.

## Information flow and current integration boundary

The current synthetic continuous loop follows:

```text
synthetic sensory event
    -> population encoding
    -> developmental LIF experience
    -> memory recall / update
    -> world-state prediction
    -> bounded high-level action selection
    -> simulated environment transition
    -> observed outcome and prediction error
    -> bounded learning and regulation
```

The same event is delivered to the high-level loop and the LIF substrate, but
the current action policy is still the high-level `ActionLearner`; it is not
decoded from LIF population activity. Basal-ganglia-inspired reward learning,
cerebellar forward prediction, full body/environment modelling and integrated
circuit-to-action decoding remain later work packages. The runtime's toy
environment is not a robotics interface and cannot perform real-world actions.

The request-driven cognitive cycle is separate:

```text
explicit language request
    -> consent-scoped memory retrieval
    -> attention and working memory
    -> lexical novelty / overlap heuristic
    -> deterministic plan and approval boundary
    -> optional language provider
    -> structured evidence trace
```

The LLM may assist language comprehension/generation at an adapter boundary.
It is not required for the continuous runtime, neural simulation, synthetic
memory updates, regulation or simulated action.

## Scientific and engineering requirements

Every major mechanism must specify empirical motivation, competing
explanations, computational hypothesis, abstraction level, omitted biology,
typed interface, baseline, ablation, task, failure cases, resource limits and
safety boundary.

Brain-region names are research pointers, not module design instructions. A
module is not a hippocampus, cortex or basal ganglia simply because it has that
name. Passing a synthetic test demonstrates only the tested software property;
it does not establish biological equivalence, human-like cognition,
consciousness or sentience.

## Existing modules and migration

- `development/program.py`: active developmental LIF network, local learning,
  regulation and explicit developmental clock.
- `development/prenatal.py`: earlier seven-population prenatal scaffold,
  retained as a reference/compatibility experiment.
- `neural/simulation.py`: deterministic LIF simulation kernel, delayed
  excitatory/inhibitory events, bounded queues and inspectable state.
- `neural/circuits.py`: four-neuron recurrent microcircuit baseline.
- `neural/episodic.py`: sparse engram binding/completion experiment integrated
  with consent-scoped episodic memory.
- `neural/organism_loop.py`: synthetic perception, episodic loop memory,
  prediction, action selection, regulation and simulated environment.
- `runtime.py`: bounded scheduler, event queue, lifecycle, resource budgets,
  status and trace.
- `cognition/dynamics.py`: legacy abstract rate-based request-path baseline;
  not advanced by the continuous runtime.
- `cognition/*`: attention, working memory, prediction and executive software;
  useful baselines, not proof of neural implementation.
- `memory/volatile_repository.py` and `memory/cognitive.py`: consent-aware
  RAM storage and distinct cognitive-memory mechanisms.
- `memory/repository.py`: legacy SQLite implementation, lazy-loaded for
  explicit compatibility/evaluation paths only.
- `brain/*`: optional language and planning adapters; the continuous organism
  must work with the provider disabled.
- `circulation/*`, `metabolism/*`, `immune/*`, `regeneration/*`,
  `sleep/*`: bounded software infrastructure and proposals, not literal
  organs.

## API and tools

The API and CLI are control surfaces over the composition-owned runtime. They
do not create independent background neural simulations.

- `GET /runtime/status` returns lifecycle state, health, counters and bounded
  recent trace.
- `POST /runtime/events` accepts explicit synthetic sensory events.
- `POST /runtime/pause` and `POST /runtime/resume` control the scheduler.
- `POST /runtime/development` advances postnatal model units explicitly.
- `artificial-human simulate --ticks N` runs deterministic accelerated ticks.
- `artificial-human serve` starts the API and its runtime lifespan.

Public runtime-control routes remain subject to the API's owner authentication
and deployment policy. The kernel and deterministic manual simulation are
testable without the API or an external language provider.

## Self-development boundary

The organism may identify a measured failure and prepare a candidate patch/test
package in an isolated workspace. It cannot directly edit the running process,
overwrite protected source, execute unrestricted commands, access secrets,
grant permissions, merge or deploy. Candidate diffs, test evidence, resource
use, expiry and rollback information require human review.

## Control Room connectivity and owner access

The Vercel Control Room proxies `/api/*` requests to the Render API on the same
origin. The browser no longer requires a manually entered API URL or bearer
token. Owner access is a single-account password fallback plus optional
WebAuthn passkeys, with signed HttpOnly cookies and no database. Passkey private
keys and biometric matching remain local to the authenticator. Camera preview
is a separate opt-in browser feature and is not connected to a sensory encoder.

A browser session can restore automatically while its cookie is valid, but
this does not keep a free Render instance awake. Free-tier suspension pauses
the process-local neural loop; always-on execution requires paid compute.
