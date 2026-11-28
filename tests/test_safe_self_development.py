from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from artificial_human.development.self_development import (
    HumanApprovalRecord,
    ImprovementProposal,
    SafeSelfDevelopment,
    SandboxCheckResult,
    SandboxEvidence,
    SandboxLimits,
)


BASE = "a" * 40


def valid_diff(path: str = "src/example.py", added: str = "new_value = 2") -> str:
    return f"""diff --git a/{path} b/{path}
index 1111111..2222222 100644
--- a/{path}
+++ b/{path}
@@ -1 +1 @@
-old_value = 1
+{added}
"""


def make_proposal(
    *,
    checks: tuple[str, ...] = ("ruff check .", "mypy src", "pytest -q"),
    diff: str | None = None,
    guard: SafeSelfDevelopment | None = None,
) -> ImprovementProposal:
    return (guard or SafeSelfDevelopment()).propose(
        "Fix a deterministic benchmark regression",
        diff or valid_diff(),
        checks,
        base_revision=BASE,
    )


def make_result(
    check: str,
    *,
    sandbox_id: str = "sandbox-immutable-01",
    **overrides: object,
) -> SandboxCheckResult:
    values: dict[str, object] = {
        "check": check,
        "base_revision": BASE,
        "sandbox_id": sandbox_id,
        "exit_code": 0,
        "elapsed_seconds": 3.0,
        "peak_memory_mb": 256.0,
        "output_bytes": 4096,
        "timed_out": False,
        "isolated_worktree": True,
        "base_read_only": True,
        "network_disabled": True,
        "stdout_digest": "b" * 64,
        "stderr_digest": "c" * 64,
    }
    values.update(overrides)
    return SandboxCheckResult(**values)  # type: ignore[arg-type]


class FakeSandboxExecutor:
    def __init__(self, results: dict[str, SandboxCheckResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, SandboxLimits]] = []

    def run_check(
        self,
        proposal: ImprovementProposal,
        check: str,
        limits: SandboxLimits,
    ) -> SandboxCheckResult:
        assert proposal.base_revision == BASE
        self.calls.append((check, limits))
        return self.results[check]


def test_proposal_is_immutable_digest_bound_to_base_and_rollback_plan() -> None:
    proposal = make_proposal()

    assert proposal.base_revision == BASE
    assert proposal.file_paths == ("src/example.py",)
    assert len(proposal.patch_digest) == 64
    assert len(proposal.identifier) == 24
    assert proposal.rollback_plan.proposal_digest == proposal.patch_digest
    assert proposal.status == "awaiting_human_review"
    with pytest.raises(FrozenInstanceError):
        proposal.status = "approved"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("diff", "message"),
    [
        (valid_diff("src/../secrets.txt"), "path outside"),
        (valid_diff("/etc/passwd"), "path outside"),
        (valid_diff("src/token.py", "TOKEN = 'ghp_" + "A" * 30 + "'"), "credential"),
        (valid_diff("src/unsafe.py", "eval('candidate')"), "prohibited execution"),
        (
            valid_diff().replace("index 1111111..2222222 100644", "new file mode 120000"),
            "symlink",
        ),
    ],
)
def test_proposal_rejects_unsafe_patch_content(diff: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_proposal(checks=("pytest -q",), diff=diff)


def test_proposal_rejects_unapproved_checks_and_invalid_base_revision() -> None:
    guard = SafeSelfDevelopment()
    with pytest.raises(ValueError, match="allowlisted"):
        guard.propose(
            "hypothesis", valid_diff(), ("python -c 'print(1)'",), base_revision=BASE
        )
    with pytest.raises(ValueError, match="commit SHA"):
        guard.propose("hypothesis", valid_diff(), ("pytest -q",), base_revision="main")


def test_proposal_enforces_patch_and_changed_line_budgets() -> None:
    guard = SafeSelfDevelopment(limits=SandboxLimits(max_patch_bytes=20))
    with pytest.raises(ValueError, match="patch budget"):
        make_proposal(guard=guard, checks=("pytest -q",))

    guard = SafeSelfDevelopment(limits=SandboxLimits(max_changed_lines=1))
    with pytest.raises(ValueError, match="changed-line budget"):
        make_proposal(guard=guard, checks=("pytest -q",))


def test_running_checks_fails_closed_without_a_trusted_sandbox_executor() -> None:
    guard = SafeSelfDevelopment()
    proposal = make_proposal()

    with pytest.raises(RuntimeError, match="no trusted isolated sandbox"):
        guard.run_checks(proposal)


def test_verified_sandbox_evidence_is_reproducible_and_approval_is_commit_only() -> None:
    proposal = make_proposal()
    results = tuple(make_result(check) for check in proposal.checks)
    guard = SafeSelfDevelopment()

    first = guard.verify(proposal, results)
    second = guard.verify(proposal, results)

    assert first.verified
    assert first.status == "verified_for_human_review"
    assert first.evidence_digest == second.evidence_digest
    approval = guard.record_human_approval(
        proposal, first, "human-review-approval-2026-11-28"
    )
    assert isinstance(approval, HumanApprovalRecord)
    assert approval.scope == "commit_only"
    assert approval.proposal_digest == proposal.patch_digest
    assert approval.evidence_digest == first.evidence_digest


def test_run_checks_delegates_only_allowlisted_checks_and_verifies_attestations() -> None:
    proposal = make_proposal(checks=("ruff check .", "pytest -q"))
    limits = SandboxLimits(max_wall_seconds=12.0)
    executor = FakeSandboxExecutor({
        check: make_result(check) for check in proposal.checks
    })
    guard = SafeSelfDevelopment(executor=executor, limits=limits)

    evidence = guard.run_checks(proposal)

    assert evidence.verified
    assert [check for check, _ in executor.calls] == list(proposal.checks)
    assert all(passed_limits is limits for _, passed_limits in executor.calls)


def test_failed_or_unsafe_sandbox_results_are_retained_as_rejected_evidence() -> None:
    proposal = make_proposal(checks=("pytest -q",))
    guard = SafeSelfDevelopment()
    failed = guard.verify(
        proposal,
        (make_result("pytest -q", exit_code=1),),
    )
    unsafe = guard.verify(
        proposal,
        (make_result("pytest -q", network_disabled=False),),
    )

    assert not failed.verified
    assert failed.status == "checks_failed"
    assert failed.failure_reason == "check failed or timed out: pytest -q"
    assert failed.results[0].exit_code == 1
    assert not unsafe.verified
    assert "network denial" in (unsafe.failure_reason or "")
    with pytest.raises(ValueError, match="verified evidence"):
        guard.record_human_approval(proposal, failed, "approval")


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"timed_out": True}, "timed out"),
        ({"elapsed_seconds": 91.0}, "wall-clock budget"),
        ({"peak_memory_mb": 1537.0}, "memory budget"),
        ({"output_bytes": 100_001}, "output budget"),
        ({"base_read_only": False}, "read-only"),
        ({"isolated_worktree": False}, "isolation"),
        ({"base_revision": "b" * 40}, "different source revision"),
    ],
)
def test_sandbox_evidence_enforces_resource_and_isolation_limits(
    overrides: dict[str, object], reason: str
) -> None:
    proposal = make_proposal(checks=("pytest -q",))
    evidence = SafeSelfDevelopment().verify(
        proposal, (make_result("pytest -q", **overrides),)
    )

    assert not evidence.verified
    assert reason in (evidence.failure_reason or "")


def test_malformed_incomplete_and_cross_workspace_evidence_fails_closed() -> None:
    proposal = make_proposal(checks=("ruff check .", "pytest -q"))
    guard = SafeSelfDevelopment()

    incomplete = guard.verify(proposal, (make_result("ruff check ."),))
    mismatched = guard.verify(
        proposal,
        (
            make_result("pytest -q"),
            make_result("ruff check ."),
        ),
    )
    cross_workspace = guard.verify(
        proposal,
        (
            make_result("ruff check .", sandbox_id="sandbox-a"),
            make_result("pytest -q", sandbox_id="sandbox-b"),
        ),
    )
    malformed = guard.verify(proposal, (object(), object()))  # type: ignore[arg-type]

    assert not incomplete.verified
    assert not mismatched.verified
    assert not cross_workspace.verified
    assert not malformed.verified
    assert malformed.results == ()


def test_executor_exception_is_recorded_without_falling_back_to_host_execution() -> None:
    class FailingExecutor:
        def run_check(
            self,
            proposal: ImprovementProposal,
            check: str,
            limits: SandboxLimits,
        ) -> SandboxCheckResult:
            raise TimeoutError("sandbox unavailable")

    proposal = make_proposal(checks=("pytest -q",))
    evidence = SafeSelfDevelopment(executor=FailingExecutor()).run_checks(proposal)

    assert not evidence.verified
    assert evidence.status == "checks_failed"
    assert evidence.failure_reason == "sandbox executor failed: TimeoutError"
    assert evidence.results == ()


def test_evidence_dataclasses_are_immutable() -> None:
    proposal = make_proposal(checks=("pytest -q",))
    evidence = SafeSelfDevelopment().verify(
        proposal, (make_result("pytest -q"),)
    )
    with pytest.raises(FrozenInstanceError):
        evidence.verified = False  # type: ignore[misc]
    with pytest.raises(ValueError, match="human approval reference"):
        SafeSelfDevelopment.record_human_approval(proposal, evidence, " ")



def test_proposal_accepts_new_regression_test_files_but_rejects_file_deletion() -> None:
    new_file = """diff --git a/tests/test_regression.py b/tests/test_regression.py
new file mode 100644
--- /dev/null
+++ b/tests/test_regression.py
@@ -0,0 +1 @@
+def test_regression(): assert True
"""
    proposal = make_proposal(checks=("pytest -q",), diff=new_file)
    assert proposal.file_paths == ("tests/test_regression.py",)

    deleted_file = new_file.replace(
        "--- /dev/null", "--- a/tests/test_regression.py"
    ).replace(
        "+++ b/tests/test_regression.py", "+++ /dev/null"
    )
    with pytest.raises(ValueError, match="deletion"):
        make_proposal(checks=("pytest -q",), diff=deleted_file)


def test_sandbox_limits_have_absolute_upper_bounds() -> None:
    with pytest.raises(ValueError, match="max_wall_seconds"):
        SandboxLimits(max_wall_seconds=601.0)
    with pytest.raises(ValueError, match="max_memory_mb"):
        SandboxLimits(max_memory_mb=4097.0)
    with pytest.raises(ValueError, match="max_checks"):
        SandboxLimits(max_checks=4)


def test_approval_rejects_evidence_for_another_base_revision() -> None:
    proposal = make_proposal(checks=("pytest -q",))
    evidence = SafeSelfDevelopment().verify(
        proposal, (make_result("pytest -q"),)
    )
    forged = replace(evidence, base_revision="b" * 40)

    with pytest.raises(ValueError, match="verified evidence"):
        SafeSelfDevelopment.record_human_approval(
            proposal, forged, "human-approval"
        )


def test_sandbox_result_rejects_boolean_exit_code_and_non_boolean_isolation_flags() -> None:
    proposal = make_proposal(checks=("pytest -q",))
    guard = SafeSelfDevelopment()

    boolean_exit = guard.verify(
        proposal, (make_result("pytest -q", exit_code=False),)
    )
    string_flag = guard.verify(
        proposal, (make_result("pytest -q", network_disabled="yes"),)  # type: ignore[arg-type]
    )

    assert not boolean_exit.verified
    assert "invalid identity or isolation" in (boolean_exit.failure_reason or "")
    assert not string_flag.verified
    assert "isolation" in (string_flag.failure_reason or "")
