# Control Room and deployment

The Control Room is a separate static dashboard under `monitor/`. It connects to the protected, read-only API route `GET /monitor/overview`.

## Security model

- Set `ARTIFICIAL_HUMAN_PUBLIC_DEPLOYMENT=true` on every public API deployment.
- The Render Blueprint generates a random 256-bit `ARTIFICIAL_HUMAN_API_TOKEN`. Keep it in Render's secret manager; the API fails closed with HTTP 503 if it is missing.
- Set `ARTIFICIAL_HUMAN_CORS_ORIGINS` to the exact Vercel dashboard origin (comma-separated only when multiple trusted origins are needed).
- All routes except `GET /health` and CORS preflight require `Authorization: Bearer <token>`.
- API docs and the OpenAPI schema are disabled in public mode.
- The dashboard keeps the token in JavaScript memory only. It does not write the token to localStorage, cookies, URL parameters, logs, or telemetry.
- The monitoring overview only returns the operational status, cognitive subsystem labels, and deterministic prenatal scaffold snapshots. It does not return owner IDs, prompts, memory records, or sensory content.

Use HTTPS for both dashboard and API. Do not reuse a token from another application. Rotate the token by changing the host secret and reconnecting the dashboard.

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

The root `render.yaml` Blueprint creates a free Singapore web service. Review the Blueprint in Render before applying it. Set the required API token and dashboard CORS origin in Render's environment settings before using the monitor. The service's filesystem is ephemeral on the free plan; do not use it for durable personal memory. This first deployment is for aggregate monitoring only.

## Vercel dashboard

Import the repository in Vercel and set the project Root Directory to `monitor`. It is a static site with no build step. The `monitor/vercel.json` file adds browser security headers. After the first deployment, copy its exact HTTPS origin into Render's `ARTIFICIAL_HUMAN_CORS_ORIGINS`.

## Interpretation

The prenatal display lets the operator inspect deterministic snapshots at any day from 0 to 280. The simulation is not automatically advancing and does not persist a developmental history between requests. It is a software abstraction and does not imply biological equivalence, consciousness, or clinical measurements. Its values describe this model's internal parameters only.
