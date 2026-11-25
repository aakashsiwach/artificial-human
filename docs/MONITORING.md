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

The conversation panel remains disabled until the API reports a configured reasoning provider. Configure these variables on the Render service:

- `ARTIFICIAL_HUMAN_LLM_BASE_URL`: the OpenAI-compatible API base URL (must use HTTPS for a remote endpoint; local HTTP is permitted only for loopback/local development).
- `ARTIFICIAL_HUMAN_LLM_MODEL`: the exact model identifier expected by that endpoint.
- `ARTIFICIAL_HUMAN_LLM_API_KEY`: optional only when the endpoint does not require bearer authentication; otherwise store the key as a Render secret.

Do not put model credentials in the browser. The browser sends its bearer token to the Artificial Human API; the server calls the configured provider.

The conversation endpoint runs the existing explicit cognitive cycle before model generation. It uses a bounded, process-local working-memory session. Conversation turns are not automatically saved to long-term memory, and the public dashboard has no route for adding, reading, exporting, or modifying memories. Working-memory sessions are volatile and may disappear when the free service restarts, sleeps, or deploys. A successful text response is model output coordinated by the software architecture, not evidence of consciousness or biological equivalence.

As of the current free-tier deployment, no model provider is configured yet. The conversation panel therefore reports that the model is not configured instead of fabricating responses. Provider setup is required before the Send control becomes active.

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
