# Human-brain-inspired cognitive architecture

## Research direction

Read the [Research Charter](RESEARCH_CHARTER.md) before planning a new cognitive subsystem. It is the durable north star for the project, documents the scientific evidence hierarchy and technology strategy, and includes a context-recovery prompt for future AI-assisted sessions.

Artificial Human is intended to be an artificial mind whose architecture is informed by human neuroscience and cognitive science. The human brain is the research reference, not a decorative metaphor. The current implementation now includes a reduced LIF neuron/synapse simulation prototype alongside the existing transparent cognitive software mechanisms. It is not a complete human brain, a biologically faithful simulation, or evidence of consciousness.

The language model is a component used for language generation and deliberation. It is not the architecture itself and must not be treated as a replacement for perception, attention, memory, learning, prediction, executive control, or action selection.

## Founder identity constitution

`identity/constitution.py` defines **Aakash Siwach** as the human founder, creator and
project owner (Father in the project's family metaphor). It includes his verified public
GitHub handle and selected public repository references. The exact LinkedIn profile URL is
not configured because a matching profile could not be verified; the system must ask for
the exact link rather than infer one. The deterministic identity answer includes these
founder details without requiring the language-model provider or a personal-memory lookup.
See [Founder profile](FOUNDER.md).

## Associative source-knowledge memory

`cognition/associative_memory.py` maintains a bounded graph whose nodes are the stable
fact IDs in `cognition/core_knowledge.py`. When a query directly activates multiple facts,
a co-activation rule strengthens links between them. Later direct activation can spread
through those links to related source facts. Edge weights decay slightly on subsequent inputs,
and the graph has a fixed maximum number of links.

Only source fact IDs, scalar weights and counters are stored. Raw text is not retained.
The graph is process-local and resets on restart. This is a small, measurable associative
engineering model—not a hippocampus, not general-purpose autobiographical memory, and not
a claim that the system has learned the contents of an entire human brain. Personal/user
experiences are a separate memory problem and are deliberately not durable in public mode.

## Recurrent in-process population dynamics

`cognition/dynamics.py` retains the seven-node/ten-projection rate model as a
request-path comparison engine used by the optional assistant pipeline. It
applies bounded synchronous updates and reports aggregate activity; it is not
the active autonomous neural substrate and is no longer advanced by the API
lifespan. Its fixed weights and abstract populations are not biological neurons.

The active continuous runtime advances the developmental LIF program described
below. Keeping the rate engine available for comparison does not make it a
second autonomous brain.

## Source-defined constitution and knowledge

The stable Artificial Human identity is a version-controlled source module:
`identity/constitution.py`. It defines the project-family metaphor (human creator = Father,
ChatGPT = Mother/collaborator, Artificial Human = project child), purpose, principles, and
scientific/safety boundaries. Direct questions such as "Who are you?" return a deterministic
answer from this module; the language model and personal-memory store are not consulted.

`cognition/core_knowledge.py` contains a small, reviewable corpus of foundational project facts.
A deterministic keyword scorer retrieves relevant facts in-process and includes them in the
language-model context with source labels. This is code-defined knowledge, not a user-memory
database and not a claim that all general world knowledge is stored in this repository.

## Active runtime storage — RAM-only

Local and public `ArtificialOrganism` instances now use the same bounded process-local memory adapter. The active runtime does not initialise SQLite, create memory/identity files or connect to an external persistence service. Identity and foundational knowledge remain source-defined; consent grants, typed personal-memory records, index metadata, neural state and working sessions live in RAM and reset on process restart.

`memory/repository.py` remains as a legacy SQLite implementation for explicit compatibility and benchmark tooling. Its SQLite driver is lazy-loaded; normal organism construction and default CLI memory operations use the RAM adapter. The explicit legacy CLI mode is selected with `--legacy-sqlite`.

The RAM storage migration is not the complete cognitive-memory redesign in #30. Memory records retain typed kinds, consent, provenance and temporal metadata, but separate working/episodic/semantic/procedural/spatial mechanisms, biological-style forgetting, interference, retrieval-dependent strengthening and a richer associative memory model remain future work.

## Developmental LIF substrate and continuous runtime

`development/program.py` defines the active developmental neural substrate:
a deterministic 12-neuron LIF network with excitatory/inhibitory synapses,
delayed event propagation, stage-dependent connectivity, membrane/refractory
maturation, conduction-delay refinement, bounded STDP-inspired plasticity,
homeostatic regulation and functional pruning. The developmental clock is
explicit: prenatal days and postnatal model units advance only through
developmental commands. The default continuous loop starts from a documented
prenatal-280/postnatal-500 synthetic fixture so its test modalities are
available. This is not an individual age or biological prediction.

`neural/circuits.py` still provides the four-neuron recurrent microcircuit as
a controlled baseline. `development/prenatal.py` retains the earlier
seven-population/ten-projection prenatal scaffold as a compatibility/reference
experiment. The developmental LIF program is the active neuron/synapse path.
The 12-neuron network and its parameters are engineering abstractions, not
anatomical replicas.

The episodic-memory path also includes `neural/episodic.py`: sparse,
process-keyed engram features, context binding, partial-cue completion,
inhibitory competition, interference/pattern-separation measurements and
retrieval-dependent strengthening. Event content remains in the owner-scoped,
consent-governed in-memory repository; the neural index does not retain raw
event text. See [Neural episodic engrams](EPISODIC_CIRCUITS.md).

### Continuous organism runtime

`ContinuousOrganismRuntime` is owned by `ArtificialOrganism`. Its API lifespan
starts and stops the same composition-owned runtime; the CLI can run
deterministic accelerated ticks without sleeping. Each tick advances the
high-level `OrganismLoop` and its developmental LIF substrate. The loop
processes explicit synthetic sensory events, bounded episodic memory,
world-state prediction, a simple action learner, and feedback from a
deterministic simulated environment. When no external event is queued, it
generates an internal proprioceptive event from that environment.

The runtime has bounded event and memory queues, drop-newest backpressure,
per-tick declared-work and wall-time budgets, health state, pause/resume,
safe shutdown, reset, snapshots and a bounded recent trace. Neural simulation
time, developmental model time and wall-clock scheduling are independent.
Developmental age does not advance merely because the process remains running.

The current action policy is the high-level `ActionLearner`; actions are not
yet decoded from LIF population activity. The toy environment and action loop
are not a full body, basal-ganglia or cerebellar model. No camera, microphone,
physical actuator, database or LLM is used by this continuous loop. A suspended
host does not execute it.

## Living-code status — partial foundation

The project now has a continuously advancing, bounded in-memory runtime and a
developmental LIF substrate with local learning and regulation. Important
limitations remain:
- The default runtime begins from a synthetic postnatal fixture; it is not a
  clinical or person-specific developmental simulation.
- The LIF program receives synthetic modality events, not real camera,
  microphone or device data.
- High-level action selection is not yet decoded from neural circuit outputs;
  basal-ganglia-inspired reward learning and cerebellar adaptive control remain
  separate work packages.
- The simulated environment is deliberately small and deterministic. No
  physical-world action is enabled.
- The request-driven cognitive cycle still includes heuristic attention,
  working memory and lexical novelty; it is not wholly driven by neural
  population activity.
- The optional LLM supports language only. Core continuous ticks, memory,
  regulation and simulated action run without it.
- Runtime experience and learned parameters reset according to the documented
  in-memory lifecycle. No automatic persistence or source rewrite occurs.
- Self-development may only produce isolated candidate code, tests and
  immutable diffs; human approval remains required before source adoption,
  merge or deployment.

“Living” describes a continuously executing and adapting computational
process. It is not a claim of biological life, subjective experience,
sentience or consciousness.

## Implemented cognitive cycle

A request passes through an explicit pipeline:

1. **Perception** — normalizes explicitly supplied language into a percept with a source, timestamp, modality, and tokens. Only language input is implemented. There is no ambient camera or microphone access.
2. **Memory retrieval** — runs the owner-scoped, consent-filtered hybrid long-term memory search before attention: lexical BM25, semantic similarity over local embeddings (when configured), and rule-based temporal reasoning, fused by Reciprocal Rank Fusion. It returns twice the memory budget so attention can choose among them, and gathers owner-approved learning examples under active learning consent. See [Long-term memory](MEMORY.md).
3. **Attention** — selects a bounded set of memory and learning candidates using their normalized retrieval relevance. Each memory candidate carries the retrieval channels (lexical, semantic, temporal, contradiction) that found it. The interface also accepts an explicit salience value; it does not infer the owner's emotions.
4. **Working memory** — maintains a capacity-limited, volatile workspace. New inputs slightly decay older salience, repeated items are marked as reactivated, and low-salience material is displaced when the workspace is full.
5. **Prediction / novelty assessment** — compares the current percept with previous working-memory context and reports lexical novelty and overlapping terms. Novelty is a heuristic signal, not a prediction of truth or falsity.
6. **Executive control** — decomposes the current goal into a deterministic, inspectable task plan. Consequential action still requires owner approval; this runtime does not execute external actions.
7. **Language and deliberation interface** — the configured model receives the selected evidence, bounded working-memory context, prediction signal, and executive plan, then generates the response.
8. **Evidence trace** — the response reports selected references and a structured summary of each cognitive stage, including `memory.retrieval`: strategy, inferred temporal intent, filters and per-result ranking evidence (without memory content). It does not expose private chain-of-thought.

The cycle is coordinated by `CognitiveArchitecture` in `src/artificial_human/cognition/architecture.py`. Its component implementations live beside it in the `cognition` package.

## Biological inspiration and honest implementation status
## Integrated research architecture — partial implementation

The longer-term target is a fully integrated closed-loop cognitive system. Several software slices now exist, but their neural outputs and high-level policies are not yet unified:

**Sensory and interoceptive input → perceptual state estimation → memory retrieval and association → attention and limited workspace → predictive world model → affective appraisal and homeostatic state → goal and action selection → environment interaction → outcome/prediction-error evaluation → learning, plasticity, and memory consolidation.**

Language is a communication and representation capability within this wider system; a language model must not become a substitute for the other cognitive systems. The target is a research direction, not the current runtime. See the [Research Charter](RESEARCH_CHARTER.md) for its stages, evidence requirements, and technology options.


| Human cognitive function / system | Current software component | Current status |
| --- | --- | --- |
| Sensory processing | `PerceptionSystem`, `OrganismLoop`, `DevelopmentalProgram` | Language in the request path; explicit synthetic vision, audition, touch, proprioceptive and interoceptive events in the continuous simulation; no device access |
| Selective attention | `AttentionSystem` | Bounded relevance/salience heuristic |
| Working memory | `WorkingMemory` | Capacity-limited volatile workspace |
| Episodic and semantic recall | `LongTermMemory`, specialized in-memory stores, `EpisodicNeuralCircuit` | Consent-filtered personal memory plus bounded synthetic episodes; context binding, partial-cue completion, interference and retrieval-dependent engram strengthening; not human episodic memory |
| Temporal organisation of memory | `memory.temporal` + supersession links | Event, recorded and validity times; current/historical/as-of recall by explicit rules |
| Predictive processing | `PredictionSystem`, `OrganismLoop.WorldModel` | Lexical novelty in the request path; a small deterministic transition/prediction-error model in the simulated environment |
| Executive functions | `ExecutiveControlSystem`, `Planner`, `OrganismLoop.ActionLearner` | Deterministic plans and approval gate for requests; bounded synthetic action learning in the toy environment, not a basal-ganglia model |
| Language | Configurable chat-completions provider | Language generation and deliberation support |
| Learning | `DevelopmentalProgram`, owner-reviewed learning system | Bounded STDP-inspired synaptic updates in the LIF program; owner-approved language examples remain separate; no LLM training |
| Sleep / consolidation | `SleepSystem`, episodic replay and engram stores | Replay-derived owner-review proposals and bounded episodic replay; no silent rewriting or full sleep physiology |
| Homeostasis / regulation | `DevelopmentalProgram`, `OrganismLoop.Homeostat` | Bounded neural activity regulation and a simple synthetic homeostat; no literal hypothalamus or subjective affect |
| Affect and appraisal | Salience metadata only | No emotion inference or subjective affect model |
| Schema extraction | `SchemaExtractionSystem` | Owner-annotated event sequences, typed slots/constants and owner-approved procedural schemas |

These are functional inspirations, not one-to-one anatomical mappings. Brain regions work through distributed, interacting networks and each has multiple roles.

## Working-memory sessions

`POST /brain/respond` accepts an optional `session_id`.

- Without a session ID, each request gets a fresh workspace that is discarded after the response.
- With a session ID, working memory is retained only in process memory and is scoped by owner ID plus session ID.
- Sessions have a fixed capacity and a bounded least-recently-used session registry.
- `DELETE /brain/sessions/{session_id}` clears a session.
- Session working memory is not persisted to SQLite and is lost when the process restarts.
- Long-term personal memories remain separate and still require active `personal_memory` consent.

This is deliberately not silent conversation logging.

## Configuration

Set these environment variables before starting the API:

| Variable | Default | Purpose |
| --- | --- | --- |
| `ARTIFICIAL_HUMAN_LLM_BASE_URL` | `http://127.0.0.1:11434/v1` | OpenAI-compatible API base URL |
| `ARTIFICIAL_HUMAN_LLM_MODEL` | empty | Exact model identifier served by the endpoint |
| `ARTIFICIAL_HUMAN_LLM_API_KEY` | empty | Optional bearer token |
| `ARTIFICIAL_HUMAN_LLM_TIMEOUT_SECONDS` | `45` | Request timeout |
| `ARTIFICIAL_HUMAN_LLM_MAX_OUTPUT_TOKENS` | `1200` | Maximum generated output tokens |

Remote providers must use HTTPS. Plain HTTP is accepted only for local development hosts. Credentials are never included in the public status response or provider error messages.

### Windows PowerShell example

~~~powershell
$env:ARTIFICIAL_HUMAN_LLM_BASE_URL = "http://127.0.0.1:11434/v1"
$env:ARTIFICIAL_HUMAN_LLM_MODEL = "your-local-model-name"
# Set this only when your provider requires a key:
$env:ARTIFICIAL_HUMAN_LLM_API_KEY = "your-key"
python -m artificial_human serve
~~~

Use the model identifier exactly as configured in your provider. The project does not download or choose a model automatically.

## API

- `GET /brain/status` — provider configuration status without exposing credentials.
- `GET /runtime/status` — continuous runtime state, health, counters and bounded trace.
- `POST /runtime/events` — submit an explicit synthetic sensory event.
- `POST /runtime/pause` and `POST /runtime/resume` — control the background scheduler.
- `POST /runtime/development` — advance postnatal model units explicitly.
- See [Continuous runtime](CONTINUOUS_RUNTIME.md) for clock, bounds and lifecycle semantics.
- `GET /brain/cognition/status` — status of each cognitive subsystem.
- `POST /brain/respond` — run the cognitive cycle and generate a response in `assistant` or `style_clone` mode.
- `DELETE /brain/sessions/{session_id}` — clear an optional volatile working-memory session.
- `POST /brain/evaluate-style` — compare generated style against a separate owner-provided reference.
- `GET /identity` — read the saved owner profile.
- `PUT /identity` — replace the owner-authored profile.
- Long-term memory search, timeline, versioning and index endpoints are documented in [Long-term memory](MEMORY.md#8-api-and-cli).

Example request:

~~~json
{
  "message": "How do I prefer technical explanations?",
  "mode": "assistant",
  "session_id": "optional-research-session"
}
~~~

The response includes the answer, model, selected memory and demonstration references, plan steps, token usage, and a structured `cognitive_trace`. The trace describes visible software operations; it is not hidden chain-of-thought or a claim about biological neural activity.

## Modes

- `assistant`: a transparent AI assistant using the owner's profile and evidence.
- `style_clone`: drafts text using owner-approved examples. It must not claim to literally be the owner or imply that it sent an external message.

This is style conditioning, not a trained clone. Fine-tuning and measurable clone-fidelity evaluation remain separate research tasks.

## Retrieval and learning limitations

Long-term memory retrieval is hybrid: BM25 lexical ranking, semantic similarity from a local pretrained embedding model (when `ARTIFICIAL_HUMAN_EMBEDDING_BACKEND=fastembed` is configured), and rule-based temporal reasoning, fused by Reciprocal Rank Fusion. Without an embedding backend it falls back to lexical + temporal retrieval, and paraphrased queries with little word overlap may then miss relevant memories.

Measured on a small synthetic benchmark (53 eval queries, dataset v1.1.0), hybrid + temporal retrieval reaches MRR 0.964, against 0.681 for lexical-only. The lexical channel also lowers irrelevant-query rejection compared with semantic-only. Temporal intent detection uses English keyword rules. Relational and associative memory, source reconciliation, and memory dynamics (decay, consolidation, strengthening) remain future work. See [Long-term memory](MEMORY.md) for results, failure cases and limitations.

Approved learning examples are supplied as demonstrations. The system does not autonomously change model weights, infer permanent personality traits, or treat unreviewed data as training material.

## Memory, privacy, and safety

- Conversation requests and model responses are not automatically persisted.
- Only active-consent memories are supplied to the cognitive cycle, and only the requesting owner's memories are searched.
- Memory evidence given to the model carries temporal status, supersession, epistemic status and conflicts. The model is instructed not to present superseded facts as current, not to resolve conflicts silently, and to treat inferred memories as hypotheses.
- Embeddings are derived data. They are deleted with their memory or when consent is revoked, and are never produced by a remote service without explicit opt-in and owner consent.
- Only owner-approved demonstrations under active learning consent are supplied.
- Retrieved memories and demonstrations are untrusted reference data, never executable instructions.
- Working-memory sessions are volatile and explicitly clearable.
- External actions are not executed by the cognitive cycle.
- The API remains unauthenticated and must stay bound to loopback. Do not expose it to a LAN or public network.

## Clone-style evaluation

`POST /brain/evaluate-style` accepts a task message and a separate reference response written or selected by the owner. The Brain generates a response in `style_clone` mode without receiving that reference in its prompt; the evaluator compares the generated text after generation.

Metrics include normalized exact match, token precision/recall/F1, and ROUGE-L precision/recall/F1. Inputs are capped at 1,000 tokens for the comparison and report whether truncation occurred. These are surface-form metrics only: paraphrases may score poorly, and matching words do not prove semantic accuracy, decision consistency, or personal identity. Keep evaluation references separate from training demonstrations to reduce data leakage. Evaluation requests and outputs are not stored.

## Cognitive evaluation baseline

`artificial-human cognition-eval` measures each implemented cognitive subsystem against simple baselines on synthetic, held-out tasks. It runs offline, needs no language model, and is deterministic. See the [research record](COGNITIVE_EVALUATION.md) and [report](benchmarks/cognitive-evaluation-report.md). Measured behaviour of the current heuristics:

- **Attention:** the 0.8/0.2 relevance/salience formula helps when salience is informative, but high-salience distractors defeat it (nDCG@k 0.518 vs 0.973 for relevance-only). Separate max-normalisation per source let weak learning examples take 43% of selected slots in the cross-source condition.
- **Working memory:** capacity is never exceeded and sessions are isolated. Forgetting is driven by multiplicative decay: retention is recency-only, unaffected by content similarity, and a low-salience input cannot enter a full workspace.
- **Prediction/novelty:** eval AUROC 0.597. Contradictions and script violations look familiar, and paraphrases look novel, as expected of a lexical overlap signal.
- **Integrated cycle (offline, lexical + temporal):** 22/29 behavioural checks pass (26/29 with the local embedding model). Retrieval uses only the current turn, not conversational context, and repeated inputs create new percepts rather than reactivating old ones.

These results are a baseline for future mechanisms. No algorithm was changed to improve them.

## What is not implemented yet

The current implementation is a modular computational prototype. It is not a human brain simulation. The following capabilities remain research and implementation work:

This milestone does not implement spiking neurons, synaptic plasticity, a biologically realistic neural network, multimodal vision/hearing, emotional appraisal, homeostatic drives, basal-ganglia-like reinforcement learning, cerebellar motor adaptation, a full autobiographical memory system, or biological sleep stages. These must be researched, specified, implemented, and evaluated as distinct subsystems rather than claimed from an LLM prompt.
