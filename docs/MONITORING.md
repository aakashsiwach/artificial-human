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

The Control Room automatically restores a valid session after reload and fetches the latest snapshot after sign-in. It refreshes when the owner opens the tab again or uses the refresh control; it does not send artificial background traffic to keep a free service awake. This removes manual API connection steps; it does not guarantee that the hosting process is continuously running.

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


## Brain-first Control Room UI

The static frontend in monitor/ is organised into three views:

- **Brain** is the first public and authenticated view. It presents an abstract, animated neural-core illustration, not an anatomical scan or measured connectome.
- **Developing** displays the read-only runtime activity stream derived from the authenticated /monitor/overview snapshot, source-verified active/partial/dormant subsystem cards, and the existing developmental model slider, population/projection metrics, cognitive status and scientific limitation note.
- **Other** contains developer tools, repository and Actions links, deployment URLs, local commands, read-only endpoint references, and existing owner-only chat, camera-preview and passkey controls.

The activity stream cycles through runtime counters returned by the API. Its motion is a visual presentation of observed state. It does not execute source, generate code, modify files, or establish autonomous code evolution.

### Module status semantics

**Active** means the mechanism is implemented and participates in the documented code path: the composition-owned continuous runtime, 12-neuron developmental LIF substrate, excitatory/inhibitory synapse propagation, bounded STDP-inspired plasticity, homeostasis, developmental connectivity/pruning, synthetic input path, bounded attention/working memory, episodic engram circuit, prediction/feedback, fixed-action simulated control, and resource scheduling.

**Partial** identifies limited or gated systems: consent-aware volatile long-term memory, owner-reviewed replay-derived consolidation, owner-annotated schema proposals, optional text language adapter, local-only browser camera preview, and safe self-development proposal contracts. In particular, the current proposal contract does not itself provide a real, independently audited OS-level sandbox executor.

**Dormant** means the current source does not implement or connect the capability: camera-to-neural visual perception, microphone/audio perception, real physical touch/vestibular sensors, integrated affect/emotion, intrinsic motivational drives, complete autobiographical memory, physical motor control/actuators, speech/voice generation, unrestricted self-rewriting, or whole-human-brain emulation/consciousness.

These are engineering status labels, not claims of biological equivalence, human-level intelligence, feelings, sentience, or consciousness.

### Privacy and operational boundaries

- The public first page exposes only the abstract brain illustration and owner sign-in action.
- The authenticated dashboard uses the same-origin Vercel /api/* proxy to access the Render service.
- /monitor/overview returns operational snapshots and subsystem state, not prompts or personal memory content.
- Camera access remains explicit opt-in and local to the browser; captured frames are not uploaded or passed into the organism.
- Provider credentials and owner authentication secrets remain in private host environment settings.
- No fake keep-alive traffic is generated to keep a free Render service awake.
- UI commands are copied to the local clipboard only; the browser page never runs shell commands.

## Production deployment and CI/CD

| Target | URL | Configuration |
| --- | --- | --- |
| Vercel Control Room | https://artificial-human-control-room.vercel.app | Root Directory monitor; Production Branch should be main |
| Render API | https://artificial-human-monitor-api.onrender.com | FastAPI web service |
| Render health | https://artificial-human-monitor-api.onrender.com/health | Public health check |
| Render service settings | https://dashboard.render.com/web/srv-dc37evbrjlhs73f1sgv0 | Live service configuration |
| GitHub Actions | https://github.com/aakashsiwach/artificial-human/actions | Pull-request and main-branch CI |

The GitHub Actions workflow runs when a pull request targets main and when code is pushed to main (it also keeps the existing feat/** push checks). It runs Python 3.12 and 3.13 quality checks, Ruff, mypy, pytest, the CLI simulation, offline memory/cognitive evaluations, neural performance and organism evaluation, as well as Control Room JavaScript syntax and UI structure checks.

The Render Blueprint declares branch main and autoDeployTrigger checksPass. The live Render service was inspected and also reports branch main with checks-pass auto-deploy enabled. It was created manually rather than being managed by the Blueprint, so the live service settings remain the operational source of truth if the Blueprint is later changed.

The Vercel production deployment metadata was inspected and identifies GitHub source repository aakashsiwach/artificial-human with commit ref main. Its production aliases include https://artificial-human-control-room.vercel.app. The static frontend source lives under monitor/ and the project Root Directory is monitor. The committed monitor/vercel.json keeps browser API requests on the Vercel origin and rewrites /api/* to the Render backend. With these Git integrations, a merge to main triggers the production build/deployment for the frontend and API; GitHub Actions supplies the repository CI checks before the main commit is considered ready.

The CI workflow does not contain deployment secrets. Provider integrations handle the actual deployment. A free Render instance may sleep after inactivity; in-memory organism state resets when its process restarts, sleeps, or redeploys. This hosting tier is not an always-on or durable-memory environment.

For more UI structure, module definitions, safety boundaries and local verification commands, see [Control Room UI contract](CONTROL_ROOM_UI.md).
