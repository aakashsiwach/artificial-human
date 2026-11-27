# Control Room, interaction, and deployment

The Control Room is a separate static dashboard under `monitor/`. It connects to a protected Render API. The dashboard provides an aggregate prenatal-development view and, when a reasoning provider is configured, an authenticated conversation panel.

## Owner authentication and passkeys

The Control Room is served from its Vercel origin and calls the Render API through the same-origin `/api/*` rewrite. The browser never asks the owner for an API URL or bearer token.

- The owner signs in with a username and password. The server checks a salted PBKDF2-SHA256 verifier (600,000 iterations) from protected environment configuration and issues a 7-day `HttpOnly; Secure; SameSite=Strict` session cookie.
- Failed password attempts are limited to five per source/username within 15 minutes per running process. The login identity is a single configured owner; this is not a general multi-user account system.
- After password login, the owner can register a platform passkey. WebAuthn requires user verification and binds the challenge, credential, RP ID and origin. Supported authenticators can prompt for Face ID, Touch ID, Windows Hello, Android biometrics or a device PIN. The private key and biometric matching remain on the authenticator.
- The server stores only a signed public-key credential record in an `HttpOnly` cookie. No passkey private key, biometric template or credential database is created. Passkey state is browser-specific; use password sign-in on another device and enroll that device separately.
- Password hashes and signing secrets are environment settings only. They must never be committed to Git, placed in URLs, exposed to the browser or written to organism memory.
- Legacy bearer-token authentication remains temporarily accepted for existing private integrations. The Control Room frontend no longer uses or prompts for it.
- The founder identity remains source-defined: Aakash Siwach is Father, founder and project owner; Artificial Human is the software project. This is project terminology, not a biological relationship.

Required Render environment variables:

| Variable | Purpose |
| --- | --- |
| `ARTIFICIAL_HUMAN_INTERACTIVE_MODE` | Set to `true` to enable the protected conversation routes |
| `ARTIFICIAL_HUMAN_AUTH_USERNAME` | Owner login name (template default: `aakash`) |
| `ARTIFICIAL_HUMAN_AUTH_PASSWORD_HASH` | PBKDF2-SHA256 verifier; never a plaintext password |
| `ARTIFICIAL_HUMAN_AUTH_SESSION_SECRET` | Random signing secret of at least 32 characters |
| `ARTIFICIAL_HUMAN_AUTH_ORIGIN` | Exact Control Room origin, e.g. `https://artificial-human-control-room.vercel.app` |
| `ARTIFICIAL_HUMAN_WEBAUTHN_RP_ID` | Control Room hostname, without scheme or path |

Generate a password verifier locally without printing the password:

```bash
python -c "from getpass import getpass; from artificial_human.auth import password_hash; print(password_hash(getpass('New Control Room password: ')))"
```

Set the printed verifier as `ARTIFICIAL_HUMAN_AUTH_PASSWORD_HASH` in Render. Generate the signing secret with a cryptographically secure random generator, for example `openssl rand -hex 32`. Keep both values private. `render.yaml` documents the required configuration, but the existing Render service was created directly and is not managed by a Blueprint; configure the live service through its Environment page.

## Camera privacy

The camera panel is opt-in. The browser requests `getUserMedia` only after the owner presses **Enable camera**. A captured image remains a local object URL in the current tab: no frame is uploaded to Render, Vercel, an LLM or the organism, and no camera frame is written to memory or disk. Tracks stop on explicit stop, sign-out, or page exit. Camera input is not yet a sensory encoder and does not contribute to neural state; that requires the later permission-scoped vision work in issue #27.

## Always-connected availability

The Control Room automatically restores a valid session after reload, polls its same-origin API while open, and refreshes when the tab becomes visible again. This removes manual API connection steps; it does not guarantee that the hosting process is continuously running.

Render documents that free web services spin down after 15 minutes without inbound traffic and take about a minute to wake on the next request. Paid compute instances do not spin down. Do not generate synthetic keep-alive traffic to bypass that limit. True always-on organism execution requires a paid, continuously running Render instance and explicit approval of the resulting charge. The current free service can still sleep, restart or pause its in-memory simulation.

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

The API key must be created by the account owner; it cannot safely be generated on the user's behalf. The API key is never sent to the browser. The browser sends its HttpOnly owner session cookie to the Render API through the same-origin Vercel proxy; the server calls Groq. Provider credentials never reach the browser.

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


## Volatile recurrent activity loop

The API process runs a small deterministic idle-activity approximation once per second while
its lifespan is active. User messages also trigger bounded recurrent updates. Both activity
and session state exist only in process memory. This is an inspectable software dynamics
experiment, not biological spontaneous neural firing. Render free-tier suspension pauses the
process and its loop; the design does not attempt to defeat platform sleep.


The Control Room's cognitive-system summary also reports the current number of volatile
associations and co-activation events. It does not display user text or personal memory.


## Founder identity display

The Control Room shows Aakash Siwach as the human founder, creator and project owner,
with his founder-supplied LinkedIn profile, verified GitHub profile, and selected public
repositories. Use the exact LinkedIn URL supplied by the founder:
`https://www.linkedin.com/in/aakashsiwach/`.
