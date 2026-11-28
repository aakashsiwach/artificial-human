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

The booleans in `SandboxCheckResult` are runtime probes and adapter attestations,
not cryptographic proof or an independent security audit. The ordinary unit
tests still use a fake adapter for contract edge cases; the dedicated Linux CI
job also runs the real Bubblewrap adapter against a disposable Git repository.
That integration evidence does not replace an independent security review of
the host kernel, Bubblewrap package, runner configuration or deployment host.

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


## Optional Linux implementation: Bubblewrap

BubblewrapSandboxExecutor is an opt-in implementation of the
IsolatedSandboxExecutor protocol in
src/artificial_human/development/bubblewrap_sandbox.py. It is constructed
only by an operator who explicitly injects it into SafeSelfDevelopment;
ArtificialOrganism does not instantiate it and the active organism receives
no new shell or source-write permission.

The adapter:

1. Verifies the requested base is a commit in the selected local Git repository.
2. Creates a bounded, manually extracted git archive snapshot. Symlinks,
   special files, path traversal, oversized files and oversized archives fail
   closed.
3. Mounts only that archive read-only, the validated proposal diff, a fixed
   runner, and the installed system/Python toolchain read-only. It does not
   mount the source checkout, host home, Git metadata, host /tmp, secrets or
   credentials.
4. Uses explicit user, PID, IPC, network and UTS namespaces; drops sandbox
   capabilities; provides a separate PID view and network namespace; and
   creates a size-limited tmpfs for /workspace and /tmp.
5. Applies the already-validated patch inside the tmpfs workspace, then
   executes only the exact allowlisted command vector without a shell.
6. Probes that the PID/network namespaces differ from the host, the host source
   root and private marker are not visible, the base mount is read-only, the
   workspace is a separate tmpfs, and host environment variables are absent.
   Evidence is accepted only when every probe succeeds.

Example operator setup (run from a Linux checkout after installing the
project's development tools and the OS bubblewrap package):

    from artificial_human.development.bubblewrap_sandbox import BubblewrapSandboxExecutor
    from artificial_human.development.self_development import SafeSelfDevelopment

    executor = BubblewrapSandboxExecutor("/path/to/artificial-human")
    try:
        self_development = SafeSelfDevelopment(executor=executor)
        evidence = self_development.run_checks(proposal)
    finally:
        executor.close()

SafeSelfDevelopment.run_checks calls the adapter's optional finish(proposal)
cleanup hook automatically. The adapter can also be used directly, in which
case callers must invoke finish or close.

### What the current resource controls mean

- Wall time is enforced by the host supervisor, which kills the sandbox process
  group when the per-check deadline expires.
- CPU time, address space, core dumps, open file descriptors and process count
  are bounded with POSIX resource limits inherited by every check child.
- The sandbox caps the process count at four and divides the configured memory
  budget across those processes using RLIMIT_AS. The aggregate address-space
  ceiling is therefore bounded by the configured memory budget. The host
  supervisor also samples aggregate process-group RSS and kills the group if
  actual resident memory crosses the same ceiling; this is an additional
  early-stop guard, not a kernel cgroup memory controller.
- Captured stdout/stderr are byte-counted and the process group is terminated
  when the output ceiling is crossed.
- Candidate workspace and temporary files live on size-limited tmpfs mounts.
- The adapter fails closed if the Linux namespaces, toolchain mounts or
  isolation probes are unavailable.

The GitHub Actions job sandbox-integration installs Bubblewrap on a fresh
Ubuntu runner and executes tests/test_bubblewrap_sandbox.py. It verifies all
three allowlisted checks in a disposable repository, checks all six isolation
attestations, confirms the patch did not modify the host checkout, and confirms
a host environment secret sentinel is not present inside the sandbox.

### Remaining review boundary

This is a concrete Linux namespace executor and an integration-tested
implementation, not an externally audited multi-tenant sandbox product. Before
using it for unattended production self-development, independently review the
Bubblewrap package and privilege mode, host kernel/user-namespace policy,
read-only toolchain mounts, resource-limit assumptions (especially aggregate
memory without a cgroup limit), and all code paths in this adapter. It remains
disabled by default. Human review is still required before any source adoption;
deployment requires separate approval.
