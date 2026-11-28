# Safe self-development: proposal and evidence boundary

## Purpose

Self-development may produce a candidate change for human review. It must never
modify the running organism, the main source tree, or deployment state. The
active organism has no self-granted write, shell, network, credential, or
deployment capability.

## Proposal workflow

1. An evaluator emits an `EvaluationGap` containing a benchmark ID, metric,
   observed value, target value, directionality, evidence SHA-256 and bounded
   failing-case IDs.
2. `SafeSelfDevelopment.propose_from_gap` accepts only a measurable gap and
   packages a bounded hypothesis, unified diff, allowlisted checks and base
   commit SHA into a frozen `ImprovementProposal`.
3. The diff validator rejects path traversal, paths outside `src/`, `tests/`,
   `docs/` and two approved root documents, binary patches, symlinks, file
   deletions, renames, mode changes, secrets, prohibited execution/network
   primitives, excessive files and excessive changed lines. New text files
   are allowed so a regression test can be proposed.
4. A separately configured `IsolatedSandboxExecutor` may apply the candidate
   in a temporary isolated worktree and run only the exact allowlist:
   `ruff check .`, `mypy src`, and `pytest -q`.
5. The adapter returns immutable `SandboxCheckResult` attestations. The core
   verifies that every requested check ran against the same base revision and
   same sandbox ID, succeeded without timeout, stayed inside wall-time,
   memory and output budgets, and attested all required isolation controls.
6. A deterministic `SandboxEvidence` bundle records success or failure. Only
   verified evidence can produce a `HumanApprovalRecord`, whose scope is
   `commit_only`. Deployment is not covered by that approval.

## Default-deny execution boundary

The repository deliberately ships **no default sandbox executor**. Calling
`run_checks` without an explicitly injected trusted adapter fails closed.
The core does not invoke `subprocess`, a shell, Git, a filesystem, a network
client, or a model to execute candidate code. The adapter is a deployment
integration and must provide all of the following:

- a temporary worktree/container distinct from the live source tree;
- read-only access to the base revision and no write path back to the host;
- isolated filesystem and process namespaces;
- network disabled;
- host credentials and sensitive environment variables unavailable;
- hard CPU/wall-clock, memory and output limits;
- only the three exact allowlisted checks, without shell interpolation.

The booleans in `SandboxCheckResult` are adapter attestations, not cryptographic
proof of operating-system isolation. Production must trust and audit the
adapter implementation itself. Tests use a fake adapter to verify the contract;
they do not claim to execute untrusted patches in a real sandbox.

## Resource limits

Default per-proposal limits:

| Resource | Default | Absolute ceiling |
|---|---:|---:|
| Wall time per check | 90 s | 600 s |
| Peak memory | 1,536 MB | 4,096 MB |
| Captured output | 100 KB | 1 MB |
| Patch size | 50 KB | 200 KB |
| Changed files | 20 | 50 |
| Changed lines | 2,000 | 5,000 |
| Allowlisted checks | 3 | 3 |

The immutable proposal carries its base SHA, patch digest, file list and rollback
instructions. Evidence digests are deterministic for identical results. Failed
checks remain as audit evidence and never change the proposal's base source.

## Approval and deployment

There is no merge, commit, or deploy method in this module. Human approval
produces a record authorizing only a separate commit workflow. Any commit must
be reviewed against the exact proposal/evidence digest. Deployment requires a
separate explicit approval and independent release checks.
