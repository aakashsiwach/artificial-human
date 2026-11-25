# Control Room, interaction, and deployment

The Control Room is a separate static dashboard under `monitor/`. It connects to a protected Render API. The dashboard provides an aggregate prenatal-development view and, when a reasoning provider is configured, an authenticated conversation panel.

## Security model

- Public deployment mode is enabled with `ARTIFICIAL_HUMAN_PUBLIC_DEPLOYMENT=true`. In this mode, API docs and the OpenAPI schema are disabled.
- `ARTIFICIAL_HUMAN_API_TOKEN` is a private bearer secret. Keep it in Render environment settings; never commit it, put it in a URL, or paste it into issue reports.
- Set `ARTIFICIAL_HUMAN_CORS_ORIGINS` to the exact HTTPS Control Room origin. The current deployment uses `https://artificial-human-control-room.vercel.app`.
- `/health` is the only unauthenticated route and exists for platform health checks.
- `ARTIFICIAL_HUMAN_MONITOR_ONLY=true` applies a public route allowlist. The monitoring route `GET /monitor/overview` requires bearer authentication.
- When `ARTIFICIAL_HUMAN_INTERACTIVE_MODE=true` is also set, three additional authenticated routes are allowed: `GET /brain/status`, `POST /brain/respond`, and `DELETE /brain/sessions/{session_id}`. Every other API route remains unavailable (404), including memory, consent, identity, training, repair, and write-management endpoints.
- The dashboard holds the bearer token only in JavaScript memory for the current tab. It does not write the token to localStorage, cookies, URL parameters, logs, or telemetry. It clears the token when the page is closed.
- The monitor overview returns aggregate organism status, cognitive subsystem labels, and deterministic prenatal scaffold snapshots. It does not return owner IDs, prompts, memory records, or sensory content.

Use HTTPS for both dashboard and API. Do not reuse a token from another application. Rotate the token by changing the Render environment value and redeploying; reconnect the dashboard with the new token.

## Interactive reasoning provider

The conversation panel remains disabled until the API reports a fully configured reasoning provider, including its credential when required. The current deployment is prepared for Groq's free-tier GPT-OSS 120B reasoning model:

| Render environment variable | Value |
| --- | --- |
| `ARTIFICIAL_HUMAN_LLM_BASE_URL` | `https://api.groq.com/openai/v1` |
| `ARTIFICIAL_HUMAN_LLM_MODEL` | `openai/gpt-oss-120b` |
| `ARTIFICIAL_HUMAN_LLM_REASONING_EFFORT` | `medium` |
| `ARTIFICIAL_HUMAN_LLM_MAX_OUTPUT_TOKENS` | `1200` |
| `ARTIFICIAL_HUMAN_LLM_API_KEY` | Your private Groq API key (required; keep it as a Render environment secret) |

To activate it:

1. Create a free GroqCloud account and generate an API key at [GroqCloud API Keys](https://console.groq.com/keys).
2. Open the Render service's **Environment** settings: [artificial-human-monitor-api](https://dashboard.render.com/web/srv-dc37evbrjlhs73f1sgv0).
3. Add the key as `ARTIFICIAL_HUMAN_LLM_API_KEY`. Do not paste it into the dashboard or into chat.
4. Save the environment changes and let Render redeploy. Reopen the Control Room and check that it shows **MODEL READY**.

The API key must be created by the account owner; it cannot safely be generated on the user's behalf. The API key is never sent to the browser. The browser sends its separate Artificial Human bearer token to the server; the server calls Groq.

Groq currently documents GPT-OSS 120B as a reasoning model. Its published free-tier limits are 30 requests per minute, 1,000 requests per day, 8,000 tokens per minute, and 200,000 tokens per day; these are account/model limits and can change. Review the [model list](https://console.groq.com/docs/models), [reasoning documentation](https://console.groq.com/docs/reasoning), and [rate limits](https://console.groq.com/docs/rate-limits) before sustained use.

The conversation endpoint runs the existing explicit cognitive cycle before model generation. It uses a bounded, process-local working-memory session. Conversation turns are not automatically saved to long-term memory, and the public dashboard has no route for adding, reading, exporting, or modifying memories. Working-memory sessions are volatile and may disappear when the free service restarts, sleeps, or deploys. A successful text response is model output coordinated by the software architecture, not evidence of consciousness or biological equivalence.

The Render service is preconfigured with the Groq endpoint, GPT-OSS 120B model ID, medium reasoning effort, and a 1,200-token output cap. It remains intentionally not ready until the account owner's Groq API key is stored as the private Render environment variable shown above. The dashboard distinguishes this missing-key state and does not fabricate responses.

## Local verification

Install the project with its development dependencies, then run:

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

Run the API locally:

```bash
ARTIFICIAL_HUMAN_API_TOKEN=local-development-token uvicorn artificial_human.api:app --host 127.0.0.1 --port 8000
```

For local use, public-deployment mode is off by default, so existing API development remains unchanged. Enter the local URL and token in the Control Room only after enabling public mode when testing the protected flow.

## Render API

The current API service is a separate free Singapore Render web service. It runs from the `feat/monitoring-control-room` branch using `pip install .` and `uvicorn artificial_human.api:app --host 0.0.0.0 --port $PORT`. It does not modify the existing `testcase-api` service.

The root `render.yaml` is deployment configuration for a future Blueprint-based setup; the live service was created directly and is not currently managed by a Blueprint. The free instance has an ephemeral filesystem and sleeps after inactivity. Do not use this deployment for durable personal memory or always-on inference.

## Vercel dashboard

The Control Room is deployed as a static site with Root Directory `monitor` and no build command. The `monitor/vercel.json` file adds browser security headers. The stable URL is `https://artificial-human-control-room.vercel.app`.

## Interpretation

The prenatal display lets the operator inspect deterministic snapshots at any day from 0 to 280. The simulation is not automatically advancing and does not persist a developmental history between requests. It is a software abstraction, not a biological fetal-brain simulator, and does not imply consciousness or clinical measurements. Its values describe this model's internal parameters only.
