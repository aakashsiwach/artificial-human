# Brain and intelligence runtime

## What this milestone provides

The Brain is a model-backed reasoning layer around three explicit inputs:

1. The owner's saved identity profile.
2. Memories that are still visible under active `personal_memory` consent.
3. Owner demonstrations that have been approved under active `personal_learning` consent.

It sends those inputs to a configurable OpenAI-compatible chat-completions endpoint. The provider adapter supports local endpoints such as an Ollama-compatible `/v1` API as well as remote HTTPS endpoints.

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

- `GET /brain/status` — provider type, model name, configuration status, and whether authentication is configured. It never returns the key.
- `POST /brain/respond` — request a response in `assistant` or `style_clone` mode.
- `GET /identity` — read the saved owner profile.
- `PUT /identity` — replace the owner-authored profile.

Example request:

~~~json
{
  "message": "How do I prefer technical explanations?",
  "mode": "assistant"
}
~~~

The response includes the selected model, retrieved memory IDs and sources, approved demonstration IDs, and token usage when provided by the model endpoint.

## Modes

- `assistant`: an AI assistant that uses the owner's profile and evidence but does not claim to be the owner.
- `style_clone`: drafts text using owner-approved examples. It must not claim to literally be the owner or imply that it sent an external message.

This is style conditioning, not a trained clone. Fine-tuning and measurable clone-fidelity evaluation are later milestones.

## Retrieval

The current retriever uses a lightweight BM25-style lexical ranking over consent-visible memory records. It is deliberately transparent and dependency-light, but it is not semantic/vector retrieval. A query with little word overlap may fail to retrieve a conceptually related memory. Embeddings, hybrid retrieval, temporal decay, contradiction handling, and retrieval evaluations remain future work.

## Memory and privacy

- Conversation requests and model responses are not automatically persisted.
- Only active-consent memories are supplied to the Brain.
- Only owner-approved demonstrations under active learning consent are supplied as examples.
- Retrieved memories and demonstrations are treated as untrusted reference data, not executable instructions.
- The response carries evidence references for owner inspection.

The API remains unauthenticated and must stay bound to loopback. Do not expose it to a LAN or public network.
