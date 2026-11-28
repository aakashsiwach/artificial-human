# Artificial Human

**Artificial Human is an open-source computational artificial-mind research project grounded in human neuroscience and cognitive science.** It explores how explicit neural, memory, learning, prediction, regulation, and action mechanisms can work together as one continuously advancing software organism.

The goal is not to wrap a language model around a database and call it a brain. Source code defines the architecture and initial mechanisms; the active organism keeps bounded live state in process memory. Language models are optional adapters, not the source of the organism's core neural activity.

> **Scientific boundary:** this is an experimental computational model. It is not biological tissue, a complete human brain, a literal human, or evidence of intelligence, sentience, subjective experience, or consciousness. Biological labels describe research questions and hypotheses, not one-to-one anatomical replicas.

<p align="center">
  <a href="https://aakashsiwach.github.io/artificial-human/"><strong>Project website</strong></a> ·
  <a href="https://github.com/aakashsiwach/artificial-human"><strong>Source repository</strong></a> ·
  <a href="https://artificial-human-control-room.vercel.app"><strong>Control Room</strong></a> ·
  <a href="https://artificial-human-monitor-api.onrender.com"><strong>Render API</strong></a>
</p>

## Live deployments

| Surface | URL | What it is |
| --- | --- | --- |
| Project website | [aakashsiwach.github.io/artificial-human](https://aakashsiwach.github.io/artificial-human/) | Public project overview, architecture, implementation status, deployment links, and documentation index |
| Control Room (Vercel) | [artificial-human-control-room.vercel.app](https://artificial-human-control-room.vercel.app) | Owner-authenticated dashboard for developmental snapshots, system status, and the optional interactive cognitive interface |
| Backend API (Render) | [artificial-human-monitor-api.onrender.com](https://artificial-human-monitor-api.onrender.com) | FastAPI service used by the Control Room |
| API health | [artificial-human-monitor-api.onrender.com/health](https://artificial-human-monitor-api.onrender.com/health) | Render health-check endpoint |

The Control Room calls the Render service through a same-origin Vercel /api proxy. The browser does not need a manually entered API URL or bearer token. Owner sign-in is required for private Control Room features; do not publish passwords, session secrets, provider keys, or API tokens.

**Hosting note:** the current Render backend uses the free service plan. Free services can spin down after inactivity, and the process-local organism state is lost on a restart, sleep-related restart, or redeploy. This deployment is not durable storage or a guarantee of always-on neural execution. The Control Room's camera preview is opt-in and local to the browser; captured frames are not uploaded to the API or fed into the organism.

Deployment configuration and owner setup are documented in [Control Room and monitoring deployment](docs/MONITORING.md).

## Project at a glance

| Area | Current implementation |
| --- | --- |
| Runtime | One composition-owned continuous in-memory organism runtime with independent neural, developmental, and wall-clock time |
| Neural substrate | Deterministic leaky-integrate-and-fire (LIF) neurons, excitatory/inhibitory synapses, delayed events, bounded queues, and inspectable spike traces |
| Development | A 12-neuron developmental LIF program with stage-dependent connectivity, membrane maturation, bounded STDP-inspired plasticity, homeostasis, conduction-delay refinement, and functional pruning |
| Integration | A continuous organism loop connecting synthetic sensory input, attention, working memory, episodic recall, neural activity, world prediction, action selection, and a simulated body |
| Memory | Consent-aware, owner-scoped memory interfaces and bounded in-memory stores; optional local semantic embeddings and temporal/hybrid retrieval in the request-oriented cognitive system |
| Evaluation | Deterministic cognitive, neural, developmental, action-selection, adaptive-control, and organism-level benchmarks with fixed seeds and ablations |
| Self-development | Immutable proposal and evidence contracts with strict patch validation and fail-closed sandbox integration; no built-in production sandbox executor or autonomous merge |
| Language model | Optional OpenAI-compatible provider adapter; core continuous neural/runtime functions do not require an LLM call |

## How the organism loop works

The continuous loop runs as an in-process software lifecycle, not only when an HTTP request arrives. A tick connects the main mechanisms:

~~~text
synthetic sensory event
        |
        v
sensory encoding and bounded attention
        |
        v
working memory <----> episodic recall
        |
        v
developmental LIF neural substrate
        |
        v
bounded neural-activity summary + body state
        |
        v
world prediction and learned action selection
        |
        v
bounded simulated body/environment
        |
        v
outcome, prediction error, reward and memory update
        |
        +------> bounded trace / next tick
~~~

The same composition-owned runtime is used by the API lifespan and deterministic CLI simulation. Neural simulation time, developmental model time, and wall-clock scheduling are separate clocks. The runtime supports bounded sensory intake, drop-newest backpressure, per-tick resource budgets, pause/resume, safe shutdown, reset, snapshots, and a bounded recent trace.

**Important integration detail:** the LIF substrate advances through a fixed propagation window so delayed downstream activity can be observed. A coarse population-state summary and working-memory focus condition the high-level tabular action selector. Individual LIF spikes are **not** decoded directly into motor commands. Actions remain in a small allowlisted simulated action space; no physical actuator or real-world action is connected.

See [Integrated organism loop](docs/INTEGRATED_ORGANISM_LOOP.md) and [Continuous runtime](docs/CONTINUOUS_RUNTIME.md).

## Architecture and subsystems

### Neural substrate and development

- Deterministic LIF membrane integration with explicit neuron state, excitatory/inhibitory synapses, delays, stable event ordering, bounded event/current resources, and snapshots.
- A small recurrent microcircuit is retained as a controlled baseline alongside feed-forward, disconnected, inhibition-off, and fixed-rate comparisons.
- The active developmental program uses a 12-neuron substrate. Its bounded maturation and learning rules are engineering hypotheses tested against frozen-plasticity and mature-from-start controls.
- A prenatal scaffold remains available as a reference/compatibility model; it is not a biological fetal-brain simulation.

Read [Brain architecture](docs/BRAIN.md), [Prenatal development](docs/PRENATAL_DEVELOPMENT.md), [Postnatal development](docs/POSTNATAL_DEVELOPMENT.md), [Neural plasticity](docs/NEURAL_PLASTICITY.md), and [Neural homeostasis](docs/NEURAL_HOMEOSTASIS.md).

### Cognition and memory

The request-oriented cognitive cycle provides an explicit software path for text input, memory retrieval, attention, episodic and relational context, working memory, prediction/novelty, executive planning, and optional language generation.

- Memory retrieval is owner-scoped and consent-filtered. The design combines lexical ranking, optional local semantic embeddings, temporal reasoning, and rank fusion.
- Working memory is a bounded volatile workspace.
- Episodic replay is a constrained event-neighbour mechanism, not a complete autobiographical memory.
- Sleep-inspired consolidation and schema extraction produce reviewable proposals. They require explicit owner approval before creating inferred memories or schemas.
- Semantic embeddings can use a local FastEmbed/ONNX model. Without the optional semantic extra, the system falls back to lexical and temporal retrieval.
- Personal memory and consent state in the active runtime are volatile and process-local. Restarting the process clears them.

These are separable computational mechanisms; none is a biological hippocampus, cortex, or sleep process.

Read [Brain and intelligence](docs/BRAIN.md), [Cognitive memory](docs/COGNITIVE_MEMORY.md), [Long-term memory](docs/MEMORY.md), [Episodic circuits](docs/EPISODIC_CIRCUITS.md), [Cognitive evaluation](docs/COGNITIVE_EVALUATION.md), and [ADR 0001](docs/adr/0001-semantic-hybrid-temporal-memory.md).

### Action, prediction, and simulated embodiment

- A bounded contextual action selector learns over a small fixed action set and is evaluated against random and fixed-policy baselines.
- The forward model and adaptive controller are tested against frozen-model baselines and under noise/delayed-feedback conditions.
- The simulated body/environment closes the perception → prediction → action → outcome loop.
- The integrated continuous loop runs synthetic sensory tasks and reports reward, prediction error, activity, memory, and learning metrics.
- No camera, microphone, real-world sensor, physical body, or network-connected actuator is part of the continuous organism loop.

Read [Action selection](docs/ACTION_SELECTION.md), [Adaptive forward control](docs/ADAPTIVE_FORWARD_CONTROL.md), [Predictive world model](docs/PREDICTIVE_WORLD_MODEL.md), and [Simulated body and environment](docs/SIMULATED_BODY_ENVIRONMENT.md).

### Other software systems

The repository also contains bounded event scheduling, metabolism/resource governance, immune/anomaly handling, regeneration proposals, API authentication, and a separate monitoring/Control Room frontend. These are supporting software components and should not be mistaken for evidence of corresponding biological systems.

## Evaluation and quality

Evaluation is designed to answer narrower, falsifiable questions about implemented mechanisms. It does not combine unrelated scores into an "intelligence" score.

| Evaluation | Focus |
| --- | --- |
| Cognitive evaluation | Attention, working memory, novelty/prediction, memory retrieval, episodic replay, schema extraction, sleep-inspired proposals, and integrated request scenarios |
| Neural simulation baseline | Deterministic spike propagation, recurrent dynamics, event delivery, resource use, and throughput/RAM observations |
| Developmental learning | Experience-dependent cue prediction compared with frozen-plasticity and mature-from-start controls |
| Action selection | Learned contextual policy compared with random and fixed-policy baselines |
| Adaptive control | Learning-enabled and frozen forward models across noise and feedback-delay settings |
| Organism evaluation | Fixed-seed baselines, subsystem ablations, sensitivity runs, failure counts, and host-specific resource observations |

Run the offline evaluation suite from the repository root:

~~~bash
# Cognitive subsystem evaluation (offline and deterministic)
artificial-human cognition-eval --suite all --output-dir benchmark-output

# Consolidated neural and organism evaluation (fixed seeds 17, 23, 41)
python scripts/evaluate_organism.py --output benchmark-output/organism-evaluation.json

# Neural simulation performance/RAM baseline
python scripts/benchmark_neural_simulation.py --steps 20000 --output benchmark-output/neural-simulation-baseline.json
~~~

The consolidated organism report keeps the continuous organism loop separate from the earlier noisy body/controller experiment; their results are not pooled. Timing and Python-traced memory are host-specific observations, not correctness thresholds. Regression thresholds should be introduced only after repeated-run variance and task validity are understood.

Read [Cognitive evaluation](docs/COGNITIVE_EVALUATION.md), [Organism evaluation](docs/ORGANISM_EVALUATION.md), [Neural simulation baseline](docs/benchmarks/neural-simulation-baseline.md), and the [benchmark reports](docs/benchmarks/).

## Quick start

Requires **Python 3.12 or newer**.

~~~bash
# Clone
git clone https://github.com/aakashsiwach/artificial-human.git
cd artificial-human

# Create and activate a virtual environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# Install development dependencies
python -m pip install -e ".[dev]"

# Run a deterministic simulation
python -m artificial_human simulate --ticks 100

# Inspect runtime status
python -m artificial_human status

# Start the local API (127.0.0.1:8000)
python -m artificial_human serve
~~~

Once the API is running, local OpenAPI documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Keep the development API bound to loopback; do not expose it directly to a LAN or public network without the required authentication and deployment controls.

### Optional local semantic retrieval

~~~bash
python -m pip install -e ".[dev,semantic]"
~~~

This installs the local embedding dependency. The first semantic run may download the configured model weights. Set ARTIFICIAL_HUMAN_EMBEDDING_BACKEND=fastembed to enable the local FastEmbed backend. No external embedding provider is contacted unless a separately configured and consented remote path is used.

### Quality checks

~~~bash
ruff check .
mypy src
pytest --cov=artificial_human --cov-report=term-missing
artificial-human simulate
artificial-human memory-eval --backend none
artificial-human cognition-eval --suite all
~~~

For the real semantic-memory benchmark, install the optional extra and run:

~~~bash
artificial-human memory-eval --backend fastembed --output-dir docs/benchmarks
~~~

The standard CI workflow runs on Python 3.12 and 3.13. The separate semantic-memory benchmark installs the optional local model and runs its evaluation workflow.

## Repository map

~~~text
artificial-human/
├── src/artificial_human/
│   ├── neural/          # LIF simulation, circuits, organism loop, action and body
│   ├── development/     # developmental program and safe self-development contracts
│   ├── cognition/       # perception, attention, working memory, prediction, executive control
│   ├── memory/          # consent-aware in-memory memory and retrieval
│   ├── brain/           # optional language/provider orchestration
│   ├── metabolism/      # bounded compute/resource governance
│   ├── immune/          # anomaly detection and defence proposals
│   ├── runtime.py       # continuous organism scheduler
│   ├── organism.py      # composition root
│   └── api.py           # FastAPI application and protected endpoints
├── monitor/             # static Vercel Control Room frontend
├── scripts/             # reproducible neural/organism evaluation runners
├── tests/               # deterministic unit and integration tests
└── docs/                # research charter, architecture, subsystem records, reports
~~~

## Deployment architecture

~~~text
Owner's browser
     |
     v
Vercel Control Room (static frontend)
     |
     | same-origin /api proxy
     v
Render FastAPI service
     |
     v
ArtificialOrganism + continuous in-memory runtime
~~~

- **Vercel** hosts the static owner-facing Control Room from the monitor/ directory.
- **Render** hosts the FastAPI monitor/API service. The Control Room uses its same-origin proxy; owner-only routes remain authenticated.
- **GitHub Pages** hosts the public project overview page.
- No external database is attached to the active continuous organism runtime. Render's current free service has an ephemeral process and may sleep; process-local runtime experience is not durable.

See [Monitoring and deployment](docs/MONITORING.md), [Safety](docs/SAFETY.md), and [Safe self-development](docs/SAFE_SELF_DEVELOPMENT.md) before changing deployment or permission boundaries.

## Current status and next work

The latest merged foundation includes:

- **PR #61 / Issue #36:** continuous, bounded in-memory organism lifecycle.
- **PR #62 / Issue #38:** integrated cognitive/neural organism loop and simulated-body feedback.
- **PR #63 / Issue #39:** reproducible fixed-seed neural and organism evaluation suite.
- **PR #64 / Issue #37:** safe self-development proposal, patch-validation, sandbox-evidence, and human-approval contracts.

The project is still in active research and implementation. The following remain open:

- **[Issue #37 — Safe self-development](https://github.com/aakashsiwach/artificial-human/issues/37):** PR #64 adds immutable proposal/evidence contracts and fails closed without a trusted executor. A real, separately reviewed OS-level sandbox executor is still required to prove filesystem/process isolation, network denial, credential removal, and hard resource enforcement. The current tests use a fake adapter; they do not prove real OS isolation.
- **[Issue #40 — Living-organism completion gate](https://github.com/aakashsiwach/artificial-human/issues/40):** the complete end-to-end system, clean-checkout behaviour, long-running stability, and every acceptance criterion must be verified with reproducible evidence.
- **[Issue #16 — Parent programme](https://github.com/aakashsiwach/artificial-human/issues/16):** remains open until the completion gate is satisfied.

See the [development roadmap](docs/ROADMAP.md) and [research charter](docs/RESEARCH_CHARTER.md) for the authoritative sequence and scientific direction.

## Scientific, privacy, and safety boundaries

- Brain-inspired components are computational hypotheses, not anatomical replicas.
- The continuous organism loop currently uses synthetic sensory events and a simulated body; it does not ingest ambient camera/microphone data or control a physical actuator.
- Core neural dynamics and continuous runtime operation do not require an LLM. The optional language provider is a replaceable request adapter.
- Active personal memory is consent-aware, owner-scoped, bounded, and volatile. Restarting the process clears process-local experience and learned runtime state.
- SQLite support is retained for explicit legacy compatibility/evaluation paths; it is not opened by default in the active organism runtime.
- Retrieved content is untrusted data, not executable instruction. No code generated by the cognitive cycle is executed.
- PR #64 provides safe proposal/evidence contracts, not a production sandbox or autonomous self-modification. No automatic commit, merge, or deployment is performed.
- Human-participant, animal, biological tissue, neural-organoid, or biohybrid experiments require appropriate institutional, legal, ethical, and domain-expert governance.

## Documentation index

- [Research Charter and Project North Star](docs/RESEARCH_CHARTER.md)
- [Development roadmap](docs/ROADMAP.md)
- [Brain and intelligence](docs/BRAIN.md)
- [System architecture](docs/ARCHITECTURE.md)
- [Integrated organism loop](docs/INTEGRATED_ORGANISM_LOOP.md)
- [Continuous runtime](docs/CONTINUOUS_RUNTIME.md)
- [Neural plasticity](docs/NEURAL_PLASTICITY.md)
- [Neural homeostasis](docs/NEURAL_HOMEOSTASIS.md)
- [Population coding](docs/POPULATION_CODING.md)
- [Prenatal development](docs/PRENATAL_DEVELOPMENT.md)
- [Postnatal development](docs/POSTNATAL_DEVELOPMENT.md)
- [Sensory encoders](docs/SENSORY_ENCODERS.md)
- [Multisensory integration](docs/MULTISENSORY_INTEGRATION.md)
- [Cognitive memory](docs/COGNITIVE_MEMORY.md) and [long-term memory](docs/MEMORY.md)
- [Episodic circuits](docs/EPISODIC_CIRCUITS.md)
- [Action selection](docs/ACTION_SELECTION.md)
- [Adaptive forward control](docs/ADAPTIVE_FORWARD_CONTROL.md)
- [Predictive world model](docs/PREDICTIVE_WORLD_MODEL.md)
- [Simulated body and environment](docs/SIMULATED_BODY_ENVIRONMENT.md)
- [Cognitive evaluation](docs/COGNITIVE_EVALUATION.md) and [organism evaluation](docs/ORGANISM_EVALUATION.md)
- [Safe self-development](docs/SAFE_SELF_DEVELOPMENT.md)
- [Safety and privacy](docs/SAFETY.md)
- [Control Room and deployment](docs/MONITORING.md)
- [Founder profile](docs/FOUNDER.md)
- [Contributing](CONTRIBUTING.md)

## Founder and license

**Aakash Siwach** is the human founder, creator, and project owner.  
[GitHub](https://github.com/aakashsiwach) · [LinkedIn](https://www.linkedin.com/in/aakashsiwach/)

Licensed under **Apache-2.0**. See [LICENSE](LICENSE).
