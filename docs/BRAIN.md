# Human-brain-inspired cognitive architecture

## Research direction

Artificial Human is intended to be an artificial mind whose architecture is informed by human neuroscience and cognitive science. The human brain is the research reference, not a decorative metaphor. The current implementation is a set of transparent software mechanisms inspired by selected cognitive functions; it is not a neural simulation, a complete human brain, or evidence of consciousness.

The language model is a component used for language generation and deliberation. It is not the architecture itself and must not be treated as a replacement for perception, attention, memory, learning, prediction, executive control, or action selection.

## Implemented cognitive cycle

A request passes through an explicit pipeline:

1. **Perception** — normalizes explicitly supplied language into a percept with a source, timestamp, modality, and tokens. Only language input is implemented. There is no ambient camera or microphone access.
2. **Memory retrieval** — retrieves records visible under active owner consent and owner-approved learning examples under active learning consent. Retrieval is lexical BM25-style ranking in this milestone.
3. **Attention** — selects a bounded set of memory and learning candidates using their normalized retrieval relevance. The interface also accepts an explicit salience value; it does not infer the owner's emotions.
4. **Working memory** — maintains a capacity-limited, volatile workspace. New inputs slightly decay older salience, repeated items are marked as reactivated, and low-salience material is displaced when the workspace is full.
5. **Prediction / novelty assessment** — compares the current percept with previous working-memory context and reports lexical novelty and overlapping terms. Novelty is a heuristic signal, not a prediction of truth or falsity.
6. **Executive control** — decomposes the current goal into a deterministic, inspectable task plan. Consequential action still requires owner approval; this runtime does not execute external actions.
7. **Language and deliberation interface** — the configured model receives the selected evidence, bounded working-memory context, prediction signal, and executive plan, then generates the response.
8. **Evidence trace** — the response reports selected references and a structured summary of each cognitive stage. It does not expose private chain-of-thought.

The cycle is coordinated by `CognitiveArchitecture` in `src/artificial_human/cognition/architecture.py`. Its component implementations live beside it in the `cognition` package.

## Biological inspiration and honest implementation status

| Human cognitive function / system | Current software component | Current status |
| --- | --- | --- |
| Sensory processing | `PerceptionSystem` | Explicit language input only |
| Selective attention | `AttentionSystem` | Bounded relevance/salience heuristic |
| Working memory | `WorkingMemory` | Capacity-limited volatile workspace |
| Episodic and semantic recall | Existing memory repository + `MemoryRetriever` | Consent-filtered lexical retrieval across typed records |
| Predictive processing | `PredictionSystem` | Lexical novelty / overlap heuristic |
| Executive functions | `ExecutiveControlSystem` + `Planner` | Deterministic goal decomposition and approval gate |
| Language | Configurable chat-completions provider | Language generation and deliberation support |
| Learning | Existing owner-reviewed learning system | Approved demonstrations; no automatic model training |
| Sleep / consolidation | Existing `SleepSystem` | Owner-review proposals only; no silent rewriting |

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
- `GET /brain/cognition/status` — status of each cognitive subsystem.
- `POST /brain/respond` — run the cognitive cycle and generate a response in `assistant` or `style_clone` mode.
- `DELETE /brain/sessions/{session_id}` — clear an optional volatile working-memory session.
- `POST /brain/evaluate-style` — compare generated style against a separate owner-provided reference.
- `GET /identity` — read the saved owner profile.
- `PUT /identity` — replace the owner-authored profile.

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

The current retriever uses a lightweight BM25-style lexical ranking over consent-visible memory records. It is transparent and dependency-light, but it is not semantic/vector retrieval. A query with little word overlap may fail to retrieve a conceptually related memory. Embeddings, hybrid retrieval, temporal/relational memory, contradiction handling, and retrieval evaluations remain future work.

Approved learning examples are supplied as demonstrations. The system does not autonomously change model weights, infer permanent personality traits, or treat unreviewed data as training material.

## Memory, privacy, and safety

- Conversation requests and model responses are not automatically persisted.
- Only active-consent memories are supplied to the cognitive cycle.
- Only owner-approved demonstrations under active learning consent are supplied.
- Retrieved memories and demonstrations are untrusted reference data, never executable instructions.
- Working-memory sessions are volatile and explicitly clearable.
- External actions are not executed by the cognitive cycle.
- The API remains unauthenticated and must stay bound to loopback. Do not expose it to a LAN or public network.

## Clone-style evaluation

`POST /brain/evaluate-style` accepts a task message and a separate reference response written or selected by the owner. The Brain generates a response in `style_clone` mode without receiving that reference in its prompt; the evaluator compares the generated text after generation.

Metrics include normalized exact match, token precision/recall/F1, and ROUGE-L precision/recall/F1. Inputs are capped at 1,000 tokens for the comparison and report whether truncation occurred. These are surface-form metrics only: paraphrases may score poorly, and matching words do not prove semantic accuracy, decision consistency, or personal identity. Keep evaluation references separate from training demonstrations to reduce data leakage. Evaluation requests and outputs are not stored.

## What is not implemented yet

This milestone does not implement spiking neurons, synaptic plasticity, a biologically realistic neural network, multimodal vision/hearing, emotional appraisal, homeostatic drives, basal-ganglia-like reinforcement learning, cerebellar motor adaptation, a full autobiographical memory system, or biological sleep stages. These must be researched, specified, implemented, and evaluated as distinct subsystems rather than claimed from an LLM prompt.
