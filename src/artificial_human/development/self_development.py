"""Fail-closed proposal and evidence contracts for safe self-development.

This module never executes generated code, mutates the source tree, accesses
credentials, or performs network/filesystem operations. An injected trusted
sandbox adapter is the only component allowed to run allowlisted checks.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Protocol
from pathlib import PurePosixPath


_ALLOWED_CHECKS = frozenset({"ruff check .", "mypy src", "pytest -q"})
_ALLOWED_ROOTS = frozenset({"src", "tests", "docs"})
_ALLOWED_ROOT_FILES = frozenset({"README.md", "CONTRIBUTING.md"})
_SECRET_PATTERNS = (
    re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
)
_PROHIBITED_ADDED_CODE = re.compile(
    r"(?:\beval\s*\(|\bexec\s*\(|\bos\.system\s*\(|"
    r"\bsubprocess\.|\bPopen\s*\(|\bsocket\.|"
    r"\brequests\.|\bhttpx\.|\burllib\.request\b|"
    r"\b__import__\s*\(|\bctypes\.)"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    """Hard ceilings passed to the isolated executor and checked on return."""

    max_wall_seconds: float = 90.0
    max_memory_mb: float = 1536.0
    max_output_bytes: int = 100_000
    max_patch_bytes: int = 50_000
    max_files: int = 20
    max_changed_lines: int = 2_000
    max_checks: int = 3

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_wall_seconds, bool)
            or not isinstance(self.max_wall_seconds, (int, float))
            or not math.isfinite(self.max_wall_seconds)
            or self.max_wall_seconds <= 0
        ):
            raise ValueError("max_wall_seconds must be finite and positive")
        if (
            isinstance(self.max_memory_mb, bool)
            or not isinstance(self.max_memory_mb, (int, float))
            or not math.isfinite(self.max_memory_mb)
            or self.max_memory_mb <= 0
        ):
            raise ValueError("max_memory_mb must be finite and positive")
        for name in ("max_output_bytes", "max_patch_bytes", "max_files", "max_changed_lines", "max_checks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class RollbackPlan:
    base_revision: str
    proposal_digest: str
    instructions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ImprovementProposal:
    identifier: str
    base_revision: str
    hypothesis: str
    diff: str
    checks: tuple[str, ...]
    file_paths: tuple[str, ...]
    patch_digest: str
    rollback_plan: RollbackPlan
    status: str = "awaiting_human_review"


@dataclass(frozen=True, slots=True)
class SandboxCheckResult:
    """Attested result returned by a trusted isolated sandbox adapter."""

    check: str
    base_revision: str
    sandbox_id: str
    exit_code: int
    elapsed_seconds: float
    peak_memory_mb: float
    output_bytes: int
    timed_out: bool
    isolated_worktree: bool
    base_read_only: bool
    network_disabled: bool
    stdout_digest: str
    stderr_digest: str


@dataclass(frozen=True, slots=True)
class SandboxEvidence:
    proposal_id: str
    base_revision: str
    results: tuple[SandboxCheckResult, ...]
    verified: bool
    status: str
    failure_reason: str | None
    evidence_digest: str


@dataclass(frozen=True, slots=True)
class HumanApprovalRecord:
    proposal_id: str
    proposal_digest: str
    evidence_digest: str
    approval_reference: str
    scope: str = "commit_only"


class IsolatedSandboxExecutor(Protocol):
    """Trusted adapter contract; implementation must isolate the worktree."""

    def run_check(
        self,
        proposal: ImprovementProposal,
        check: str,
        limits: SandboxLimits,
    ) -> SandboxCheckResult:
        """Run one allowlisted check without exposing host credentials/network."""
        ...


class SafeSelfDevelopment:
    """Build immutable code proposals and verify external sandbox attestations.

    No default executor is installed. Running checks fails closed until an
    explicitly trusted isolated-sandbox adapter is injected.
    """

    def __init__(
        self,
        *,
        executor: IsolatedSandboxExecutor | None = None,
        limits: SandboxLimits | None = None,
    ) -> None:
        self._executor = executor
        self.limits = limits or SandboxLimits()

    @staticmethod
    def _safe_path(path: str) -> bool:
        if not path or "\\" in path or path.startswith("/"):
            return False
        parsed = PurePosixPath(path)
        if parsed.is_absolute() or any(part in {"", ".", "..", ".git"} for part in parsed.parts):
            return False
        if parsed.parts[0] in _ALLOWED_ROOTS:
            return True
        return len(parsed.parts) == 1 and path in _ALLOWED_ROOT_FILES

    def _validate_diff(self, diff: str) -> tuple[str, ...]:
        if not isinstance(diff, str) or not diff.strip():
            raise ValueError("proposal diff must be non-empty text")
        try:
            encoded = diff.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("proposal diff must be valid UTF-8") from None
        if len(encoded) > self.limits.max_patch_bytes:
            raise ValueError("proposal diff exceeds the configured patch budget")
        if "\x00" in diff or "GIT binary patch" in diff or "Binary files " in diff:
            raise ValueError("binary or malformed patches are not accepted")
        if re.search(r"(?m)^(?:new file mode 120000|old mode |new mode |rename from |rename to )", diff):
            raise ValueError("symlink, mode-change and rename patches are not accepted")

        paths: list[str] = []
        for match in re.finditer(r"(?m)^diff --git a/(.+) b/(.+)$", diff):
            old_path, new_path = match.groups()
            if old_path != new_path or not self._safe_path(old_path):
                raise ValueError("patch contains a path outside the approved source roots")
            paths.append(old_path)
        if not paths or len(paths) > self.limits.max_files or len(set(paths)) != len(paths):
            raise ValueError("patch file count is empty, duplicated or exceeds the limit")
        if len(re.findall(r"(?m)^--- a/", diff)) != len(paths) or len(re.findall(r"(?m)^\+\+\+ b/", diff)) != len(paths):
            raise ValueError("patch must contain paired text-file headers for every file")

        changed_lines = 0
        for line in diff.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                changed_lines += 1
                if _PROHIBITED_ADDED_CODE.search(line[1:]):
                    raise ValueError("proposal contains a prohibited execution or network primitive")
            elif line.startswith("-") and not line.startswith("---"):
                changed_lines += 1
        if changed_lines > self.limits.max_changed_lines:
            raise ValueError("proposal exceeds the configured changed-line budget")
        for pattern in _SECRET_PATTERNS:
            if pattern.search(diff):
                raise ValueError("proposal contains a credential or private-key pattern")
        return tuple(sorted(paths))

    def propose(
        self,
        hypothesis: str,
        diff: str,
        checks: tuple[str, ...],
        *,
        base_revision: str,
    ) -> ImprovementProposal:
        if not isinstance(hypothesis, str) or not hypothesis.strip() or len(hypothesis) > 2_000:
            raise ValueError("hypothesis must contain 1..2000 characters")
        if not isinstance(base_revision, str) or not _REVISION.fullmatch(base_revision):
            raise ValueError("base_revision must be a full 40-character commit SHA")
        if (
            not isinstance(checks, tuple)
            or not checks
            or len(checks) > self.limits.max_checks
            or len(set(checks)) != len(checks)
            or any(check not in _ALLOWED_CHECKS for check in checks)
        ):
            raise ValueError("checks must be unique, non-empty and allowlisted")

        paths = self._validate_diff(diff)
        patch_digest = hashlib.sha256(diff.encode("utf-8")).hexdigest()
        identifier = hashlib.sha256(
            f"{base_revision}\0{hypothesis}\0{patch_digest}".encode("utf-8")
        ).hexdigest()[:24]
        rollback = RollbackPlan(
            base_revision=base_revision,
            proposal_digest=patch_digest,
            instructions=(
                "Discard the isolated temporary worktree; never copy candidate files into the live source tree.",
                f"Return to immutable base revision {base_revision}.",
                "Retain the proposal and failed-check evidence for audit; no merge or deployment was performed.",
            ),
        )
        return ImprovementProposal(
            identifier=identifier,
            base_revision=base_revision,
            hypothesis=hypothesis.strip(),
            diff=diff,
            checks=checks,
            file_paths=paths,
            patch_digest=patch_digest,
            rollback_plan=rollback,
        )

    @staticmethod
    def _evidence_digest(
        proposal: ImprovementProposal,
        results: tuple[SandboxCheckResult, ...],
        status: str,
        failure_reason: str | None,
    ) -> str:
        payload = {
            "proposal_id": proposal.identifier,
            "base_revision": proposal.base_revision,
            "status": status,
            "failure_reason": failure_reason,
            "results": [
                {
                    "check": result.check,
                    "base_revision": result.base_revision,
                    "sandbox_id": result.sandbox_id,
                    "exit_code": result.exit_code,
                    "elapsed_seconds": result.elapsed_seconds,
                    "peak_memory_mb": result.peak_memory_mb,
                    "output_bytes": result.output_bytes,
                    "timed_out": result.timed_out,
                    "isolated_worktree": result.isolated_worktree,
                    "base_read_only": result.base_read_only,
                    "network_disabled": result.network_disabled,
                    "stdout_digest": result.stdout_digest,
                    "stderr_digest": result.stderr_digest,
                }
                for result in results
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def verify(
        self,
        proposal: ImprovementProposal,
        results: tuple[SandboxCheckResult, ...],
    ) -> SandboxEvidence:
        failure: str | None = None
        if not isinstance(results, tuple) or len(results) != len(proposal.checks):
            failure = "sandbox evidence is incomplete or has unexpected check count"
        elif any(not isinstance(item, SandboxCheckResult) for item in results):
            failure = "sandbox evidence contains an invalid result type"
        else:
            sandbox_ids = {item.sandbox_id for item in results}
            if len(sandbox_ids) != 1 or not next(iter(sandbox_ids), ""):
                failure = "checks must run in one identified isolated workspace"
            else:
                for expected, result in zip(proposal.checks, results, strict=True):
                    if result.check != expected:
                        failure = "sandbox evidence does not match the requested check order"
                        break
                    if result.base_revision != proposal.base_revision:
                        failure = "sandbox checked a different source revision"
                        break
                    if not result.isolated_worktree or not result.base_read_only or not result.network_disabled:
                        failure = "sandbox isolation, read-only base or network denial was not attested"
                        break
                    if result.timed_out or result.exit_code != 0:
                        failure = f"check failed or timed out: {expected}"
                        break
                    if (
                        isinstance(result.elapsed_seconds, bool)
                        or not isinstance(result.elapsed_seconds, (int, float))
                        or not math.isfinite(result.elapsed_seconds)
                        or result.elapsed_seconds < 0
                        or result.elapsed_seconds > self.limits.max_wall_seconds
                    ):
                        failure = f"wall-clock budget exceeded or invalid: {expected}"
                        break
                    if (
                        isinstance(result.peak_memory_mb, bool)
                        or not isinstance(result.peak_memory_mb, (int, float))
                        or not math.isfinite(result.peak_memory_mb)
                        or result.peak_memory_mb < 0
                        or result.peak_memory_mb > self.limits.max_memory_mb
                    ):
                        failure = f"memory budget exceeded or invalid: {expected}"
                        break
                    if (
                        isinstance(result.output_bytes, bool)
                        or not isinstance(result.output_bytes, int)
                        or result.output_bytes < 0
                        or result.output_bytes > self.limits.max_output_bytes
                    ):
                        failure = f"output budget exceeded or invalid: {expected}"
                        break
                    if not _SHA256.fullmatch(result.stdout_digest) or not _SHA256.fullmatch(result.stderr_digest):
                        failure = f"stdout/stderr digest is missing or invalid: {expected}"
                        break

        status = "verified_for_human_review" if failure is None else "checks_failed"
        digest = self._evidence_digest(proposal, results, status, failure)
        return SandboxEvidence(
            proposal_id=proposal.identifier,
            base_revision=proposal.base_revision,
            results=results,
            verified=failure is None,
            status=status,
            failure_reason=failure,
            evidence_digest=digest,
        )

    def run_checks(self, proposal: ImprovementProposal) -> SandboxEvidence:
        """Delegate checks to an explicitly injected isolated sandbox adapter."""
        if self._executor is None:
            raise RuntimeError("no trusted isolated sandbox executor is configured")
        results: list[SandboxCheckResult] = []
        try:
            for check in proposal.checks:
                results.append(self._executor.run_check(proposal, check, self.limits))
        except Exception as exc:
            partial = tuple(results)
            reason = f"sandbox executor failed: {type(exc).__name__}"
            digest = self._evidence_digest(proposal, partial, "checks_failed", reason)
            return SandboxEvidence(
                proposal_id=proposal.identifier,
                base_revision=proposal.base_revision,
                results=partial,
                verified=False,
                status="checks_failed",
                failure_reason=reason,
                evidence_digest=digest,
            )
        return self.verify(proposal, tuple(results))

    @staticmethod
    def record_human_approval(
        proposal: ImprovementProposal,
        evidence: SandboxEvidence,
        approval_reference: str,
    ) -> HumanApprovalRecord:
        if not evidence.verified or evidence.proposal_id != proposal.identifier:
            raise ValueError("only verified evidence for this proposal can be approved")
        if not isinstance(approval_reference, str) or not approval_reference.strip():
            raise ValueError("a human approval reference is required")
        return HumanApprovalRecord(
            proposal_id=proposal.identifier,
            proposal_digest=proposal.patch_digest,
            evidence_digest=evidence.evidence_digest,
            approval_reference=approval_reference.strip(),
        )
