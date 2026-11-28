# Control Room UI — brain-first navigation and status contract

## Purpose

The Control Room is an operator-facing window into Artificial Human's implemented code. It is not the organism itself and is not a place where a language model invents subsystem status. Status labels are maintained from the verified repository implementation and must be revised with code/documentation changes.

The public landing page and the first authenticated tab show the neural core only. The authenticated workspace has three tabs: **Brain**, **Developing**, and **Other**.

## Navigation

### Brain

The default first page is an abstract animated neural-core illustration. The same SVG is moved between the public landing surface and the authenticated Brain tab; the UI does not create a second simulation. The illustration is deliberately schematic and is not a brain scan, anatomical atlas, connectome, or literal visualisation of a human brain.

The authenticated Brain view displays a concise runtime state/tick label. It does not show memory content, personal data, a conversation, or a detailed metrics dashboard.

### Developing

The Developing tab contains:

1. A read-only runtime activity stream populated from the authenticated /monitor/overview response.
2. Source-verified module cards separated into **Active**, **Partial**, and **Dormant** groups.
3. The existing developmental model slider, population/projection snapshot, cognitive-system status, and scientific limitation note.

The moving code trace is a visual feed of observed runtime counters. It does not stream source changes, write code, execute generated code, or imply autonomous code evolution.

#### Active

The current source contains these mechanisms in the active organism loop or its directly-used neural substrate:

- ContinuousOrganismRuntime: one composition-owned bounded process-local runtime.
- DevelopmentalProgram: deterministic 12-neuron LIF developmental network.
- NeuralSimulation: membrane integration, inspectable spikes, excitatory/inhibitory synapses, delayed events, bounded current/queue behaviour and snapshots.
- Bounded STDP-inspired plasticity, homeostatic regulation, stage-dependent connectivity growth and functional pruning.
- Explicit synthetic sensory-event intake. The modalities are synthetic fixtures; there is no live sensor integration.
- Bounded attention and working-memory selection in the integrated organism loop.
- Episodic recall and a separate neural engram circuit with sparse features, context binding, partial-cue completion and interference controls.
- Prediction/error feedback, learned action selection over a fixed allowlist, and a bounded simulated body/environment.
- Internal scheduling, event backpressure, resource budgets, health state and bounded traces.

Active means the code exists and participates in the documented runtime path. It does not imply biological fidelity, human-like cognition, or consciousness.

#### Partial

- Owner-scoped long-term memory is a distinct consent-aware volatile memory path; it is not a complete human memory system or fully equivalent to the synthetic episodes in the core organism loop.
- Sleep-inspired consolidation detects replay patterns and prepares owner-review proposals; it is not an autonomous biological sleep process.
- Structured schema induction works from owner-annotated event metadata and creates reviewable proposals; it is not autonomous semantic understanding.
- The optional text-language adapter is a separate request interface and is not required by the continuous neural loop.
- Camera preview is local to the browser and is not fed into the organism.
- Safe self-development currently packages bounded, evidence-linked proposals and check contracts; the repository does not supply a real, independently audited OS-level sandbox executor.

#### Dormant

The UI deliberately renders the following as muted/grey, static cards with a dormant label:

- Live visual perception and camera-to-neural encoding.
- Auditory perception and microphone-to-neural encoding.
- Real-world tactile/vestibular sensor input.
- An integrated affective/emotional appraisal system.
- Autonomous intrinsic needs and motivational drives.
- A complete autobiographical memory system.
- Physical embodiment, human motor control, and real actuators.
- Speech/voice generation.
- Autonomous source-code rewriting, source adoption, automatic merge, or deployment by the organism.
- A whole-human-brain simulation or demonstrated consciousness.

A grey card means the capability is not implemented or not connected in the current runtime, not that the underlying research area is impossible.

### Other

The Other tab contains developer tools and operator utilities:

- Repository, issues, pull requests and GitHub Actions.
- Vercel Control Room, Render API, Render health endpoint and Render service settings.
- The intended main-branch CI/CD sequence.
- Copyable local setup, test, lint, type-check, simulation and server commands.
- Read-only endpoint inventory.
- Existing owner-only chat, camera-preview and passkey controls.
- Founder profile and project links.

The page only copies commands to the local clipboard. It never executes shell commands.

## Runtime data and privacy

The authenticated dashboard reads GET /monitor/overview. The endpoint deliberately returns operational snapshots and subsystem status, not prompts or memory content. The browser uses the same-origin /api/* proxy to reach Render. Owner authentication and server-side authorisation remain mandatory for protected routes.

The UI does not:
- request camera permission until the explicit camera control is pressed;
- upload camera frames;
- ask for API bearer tokens or provider API keys;
- store passwords, memory content, or model credentials;
- keep the free Render process awake with synthetic traffic;
- modify source files or trigger a deployment.

## CI/CD

### GitHub Actions

The workflow at .github/workflows/ci.yml runs for pull requests targeting main and pushes to main (plus the existing feat/** checks). It runs:
- Python 3.12 and 3.13 installation, Ruff, mypy and pytest;
- CLI simulation;
- deterministic offline memory and cognitive evaluation;
- neural simulation performance/RAM baseline;
- fixed-seed organism evaluation and ablations;
- Control Room JavaScript syntax and required-HTML-structure checks.

### Render

render.yaml declares:
- service name artificial-human-monitor-api;
- branch main;
- autoDeployTrigger checksPass;
- Uvicorn startup on the platform-provided port;
- health check /health.

The Render service is manually managed rather than Blueprint-managed, but its live settings were inspected: branch main and autoDeployTrigger checksPass. The live service settings remain authoritative; changing render.yaml alone does not rewrite manually-managed provider settings.

### Vercel

The static frontend lives in monitor/. The Vercel production deployment metadata was inspected and identifies GitHub source repository aakashsiwach/artificial-human and production commit ref main. Its production alias is https://artificial-human-control-room.vercel.app. The project Root Directory is monitor; no frontend build command is required for the current static HTML/CSS/JavaScript app. The committed monitor/vercel.json rewrite keeps browser calls to /api/* on the Vercel origin and forwards them to Render.

The Vercel Git integration deploys from main. Preview deployments may or may not be enabled for feature branches/PRs; a missing PR preview is not evidence that the production main integration is absent.

## Deployment links

- Control Room: https://artificial-human-control-room.vercel.app
- Render API: https://artificial-human-monitor-api.onrender.com
- Render health: https://artificial-human-monitor-api.onrender.com/health
- Render settings: https://dashboard.render.com/web/srv-dc37evbrjlhs73f1sgv0
- GitHub Actions: https://github.com/aakashsiwach/artificial-human/actions

## Local verification

Install the Python project and run the repository checks:

    python -m pip install -e ".[dev]"
    pytest
    ruff check .
    mypy src
    artificial-human simulate --ticks 100

The Control Room has no npm dependency:

    node --check monitor/app.js
    node --check monitor/mind-ui.js

## Scientific interpretation

A glowing node means the UI is animating an abstract visual; it is not a neuron-level readout. The runtime trace is a read-only projection of measured software counters. Active/dormant labels describe source-code implementation state. None of these views demonstrates a complete human mind, human-level intelligence, biological equivalence, sentience, or subjective experience.
