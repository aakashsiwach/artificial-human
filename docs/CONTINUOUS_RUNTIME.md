# Continuous in-memory organism runtime

## Composition and ownership

`ArtificialOrganism` owns one `OrganismLoop` and one
`ContinuousOrganismRuntime`. Its `neural_simulation` attribute is an alias to
the same developmental LIF simulation held by `OrganismLoop.neural_development`.
The API lifespan and offline CLI simulation both operate on this composition;
they do not construct separate background neural simulations.

The high-level `OrganismLoop` performs deterministic synthetic perception,
bounded attention selection, seven-item working-memory maintenance, episodic
recall, world-state prediction, action selection, simulated-environment
feedback, and bounded memory updates. The same sensory event is encoded and
delivered to the developmental LIF substrate, which performs neuron/synapse
updates, plasticity and homeostatic regulation. Action selection uses the
opponent-channel `ActionSelectionCircuit` over a fixed three-action simulator
allowlist. Learned values cannot expand that allowlist or trigger real-world
actions. Actions are not decoded directly from LIF population activity; this
remains an explicit integration boundary. The old
population-dynamics engine remains available to the request-driven assistant
path for compatibility, but it is not advanced as a second autonomous brain by
the continuous runtime.

The default loop starts from a source-defined prenatal-280/postnatal-500
developmental fixture so all synthetic modalities used by the loop are
available. This is an engineering starting state, not a person's age or a
biological prediction. The older stage-only `DevelopmentProgram` in
`neural.organism_loop` is retained as a compatibility baseline and is not
used by the active runtime.

## Three independent clocks

| Clock | Advances when | Unit |
| --- | --- | --- |
| Neural simulation | A developmental LIF experience or simulation step runs | Neural simulation step (configured in milliseconds) |
| Developmental | An explicit `advance_developmental_time` command is issued | Prenatal days or postnatal model units |
| Wall clock | The optional asynchronous scheduler waits between ticks | Seconds |

Wall-clock uptime never silently advances developmental age. Manual
`run_ticks(n)` executes deterministic accelerated ticks without sleeping.
The API's background scheduler uses a configurable wall-clock interval while
calling the same synchronous tick function.

## Lifecycle

- **Start:** API lifespan calls `await runtime.start()`. Start is idempotent.
- **Tick:** consume at most the configured number of queued sensory events; if
  no event is queued, generate one internal proprioceptive event from the
  simulated environment. Each organism step updates perception, memory,
  prediction, action, environment feedback and the LIF substrate.
- **Pause/resume:** pause stops state advancement but leaves the bounded event
  queue available for later processing. Resume continues the same runtime.
- **Stop:** cancel and await the scheduler task; no background task is left
  running. A stopped runtime remains available for deterministic manual ticks.
- **Reset:** requires the background task to be stopped. It clears the event
  queue, recent trace, synthetic episodic state, simulated environment,
  developmental LIF state and runtime counters, then returns to the documented
  default developmental fixture.
- **Snapshot:** reports lifecycle state, health, counters, queue depth,
  developmental stage, neural simulation state and a bounded recent trace.

## Bounds and overload policy

Default `RuntimeConfig`:

| Bound | Default |
| --- | ---: |
| Sensory queue capacity | 128 events |
| Events per runtime tick | 8 |
| Declared operations per tick | 8 |
| Tick wall-time budget | 0.25 seconds |
| Maximum spikes inspected per tick | 256 |
| Recent lifecycle trace | 128 ticks |
| Manual accelerated batch | 10,000 ticks |
| Organism episodic memory | 128 episodes |
| Simulated environment episode | 100 actions before reset |
| Neural event queue | 8,192 pending events |

The event queue uses **drop newest** when full. The rejected event increments
`dropped_events` and marks runtime health as degraded. The per-tick
`ResourceGovernor` bounds declared operations and elapsed wall time; it is not
an operating-system CPU or memory monitor. Exceeding a spike or tick budget
marks health degraded and is visible in snapshots. Reset clears the degraded
state and counters.

No queue, trace, episodic store or simulated episode grows without a configured
bound. The runtime does not silently evict user-owned personal memory; that
remains governed by the separate consent-scoped in-memory repository.

## API and CLI

The API lifespan starts and stops the same runtime owned by its
`ArtificialOrganism` instance.

- `GET /runtime/status` — lifecycle, health, counters and bounded trace
- `POST /runtime/events` — submit an explicit synthetic sensory event
- `POST /runtime/pause` and `POST /runtime/resume` — control the scheduler
- `POST /runtime/development` — advance postnatal model units explicitly. The default fixture is already postnatal; a custom runtime injected with a prenatal-stage program can use the Python runtime API's `prenatal_days` argument.

The active loop exposes explicit ablation switches for attention, working
memory, episodic memory, the LIF substrate and learned action selection.
`run_integration_ablation_comparison(ticks=N)` runs the same seeded synthetic
task across the full model and five single-subsystem ablations. It reports
cumulative reward, mean absolute prediction error, neural spike count, memory
occupancy and action-learning updates. These are task-specific software
measurements, not neuroscience validation.

The offline CLI runs deterministic accelerated simulation:

```bash
artificial-human simulate --ticks 100
artificial-human status
artificial-human serve
```

No real camera, microphone, sensor, network-connected actuator, physical
environment, database or LLM is used by the continuous loop. The environment's
actions are simulated and constrained to `{-1, 0, 1}`.

## Verification

```bash
pytest -q tests/test_organism_loop.py tests/test_continuous_runtime.py tests/test_runtime_api.py tests/test_brain_api.py
```

The tests cover deterministic accelerated execution, autonomous idle
proprioception, bounded memory and queues, drop-newest backpressure, per-tick
budgets, independent developmental and neural clocks, pause/resume/stop/reset,
API lifecycle ownership, explicit sensory events and bounded trace retention.

Continuous execution is a software lifecycle property. It is not evidence of
biological life, sentience, subjective experience, human-level cognition or
consciousness.
