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
    r"(?m)(?:\beval\s*\(|\bexec\s*\(|\bos\.system\s*\(|"
    r"\bsubprocess\.|\bPopen\s*\(|\bsocket\.|"
    r"\brequests\.|\bhttpx\.|\burllib\.request\b|"
    r"\b__import__\s*\(|\bctypes\.|"
    r"^\s*(?:from\s+(?:subprocess|socket|requests|httpx|ctypes)\s+import|"
    r"import\s+(?:subprocess|socket|requests|httpx|ctypes)\b)"
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
            or not 0 < self.max_wall_seconds <= 600
        ):
            raise ValueError("max_wall_seconds must be finite and positive")
        if (
            isinstance(self.max_memory_mb, bool)
            or not isinstance(self.max_memory_mb, (int, float))
            or not math.isfinite(self.max_memory_mb)
            or not 0 < self.max_memory_mb <= 4096
        ):
            raise ValueError("max_memory_mb must be finite and positive")
        for name in ("max_output_bytes", "max_patch_bytes", "max_files", "max_changed_lines", "max_checks"):
            value = getattr(self, name)
            maximum = {
                "max_output_bytes": 1_000_000,
                "max_patch_bytes": 200_000,
                "max_files": 50,
                "max_changed_lines": 5_000,
                "max_checks": 3,
            }[name]
            if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= maximum:
                raise ValueError(f"{name} must be a positive integer no greater than {maximum}")


@dataclass(frozen=True, slots=True)
class EvaluationGap:
    """Small, provenance-carrying summary of a measured evaluation failure."""

    benchmark_id: str
    metric: str
    observed: float
    target: float
    higher_is_better: bool
    evidence_digest: str
    failing_case_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark_id, str) or not self.benchmark_id.strip() or len(self.benchmark_id) > 128:
            raise ValueError("benchmark_id must contain 1..128 characters")
        if not isinstance(self.metric, str) or not self.metric.strip() or len(self.metric) > 128:
            raise ValueError("metric must contain 1..128 characters")
        if (
            isinstance(self.observed, bool)
            or not isinstance(self.observed, (int, float))
            or not math.isfinite(self.observed)
            or isinstance(self.target, bool)
            or not isinstance(self.target, (int, float))
            or not math.isfinite(self.target)
        ):
            raise ValueError("observed and target metrics must be finite numbers")
        if not isinstance(self.higher_is_better, bool):
            raise ValueError("higher_is_better must be boolean")
        if not isinstance(self.evidence_digest, str) or not _SHA256.fullmatch(self.evidence_digest):
            raise ValueError("evaluation evidence_digest must be a SHA-256 hex digest")
        if (
            not isinstance(self.failing_case_ids, tuple)
            or not self.failing_case_ids
            or len(self.failing_case_ids) > 20
            or any(not isinstance(item, str) or not item.strip() or len(item) > 256 for item in self.failing_case_ids)
        ):
            raise ValueError("failing_case_ids must contain 1..20 bounded identifiers")

    @property
    def is_gap(self) -> bool:
        return self.observed < self.target if self.higher_is_better else self.observed > self.target


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
    trigger_benchmark_id: str | None = None
    trigger_evidence_digest: str | None = None
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
    filesystem_isolated: bool
    process_isolated: bool
    network_disabled: bool
    credentials_unavailable: bool
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
        if len(parsed.parts) >= 2 and parsed.parts[0] in _ALLOWED_ROOTS:
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
        if re.search(r"(?m)^(?:new file mode (?:100755|120000)|old mode |new mode |rename from |rename to )", diff):
            raise ValueError("symlink, mode-change and rename patches are not accepted")

        matches = list(re.finditer(r"(?m)^diff --git a/(.+) b/(.+)$", diff))
        paths: list[str] = []
        for index, match in enumerate(matches):
            old_path, new_path = match.groups()
            if old_path != new_path or not self._safe_path(old_path):
                raise ValueError("patch contains a path outside the approved source roots")
            block_end = matches[index + 1].start() if index + 1 < len(matches) else len(diff)
            block = diff[match.start():block_end]
            old_headers = re.findall(r"(?m)^--- (.+)$", block)
            new_headers = re.findall(r"(?m)^\+\+\+ (.+)$", block)
            if len(old_headers) != 1 or len(new_headers) != 1:
                raise ValueError("patch must contain one old/new text-file header per file")
            if old_headers[0] not in {f"a/{old_path}", "/dev/null"}:
                raise ValueError("patch old-file header does not match its diff path")
            if new_headers[0] != f"b/{new_path}":
                raise ValueError("file deletion or mismatched new-file header is not accepted")
            paths.append(old_path)
        if not paths or len(paths) > self.limits.max_files or len(set(paths)) != len(paths):
            raise ValueError("patch file count is empty, duplicated or exceeds the limit")

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
        trigger_benchmark_id: str | None = None,
        trigger_evidence_digest: str | None = None,
    ) -> ImprovementProposal:
        if not isinstance(hypothesis, str) or not hypothesis.strip() or len(hypothesis) > 2_000:
            raise ValueError("hypothesis must contain 1..2000 characters")
        if not isinstance(base_revision, str) or not _REVISION.fullmatch(base_revision):
            raise ValueError("base_revision must be a full 40-character commit SHA")
        if (
            not isinstance(checks, tuple)
            or not checks
            or len(checks) > self.limits.max_checks
            or any(not isinstance(check, str) or check not in _ALLOWED_CHECKS for check in checks)
            or len(set(checks)) != len(checks)
        ):
            raise ValueError("checks must be unique, non-empty and allowlisted")

        if (trigger_benchmark_id is None) != (trigger_evidence_digest is None):
            raise ValueError("benchmark trigger ID and evidence digest must be supplied together")
        if trigger_benchmark_id is not None and (
            not isinstance(trigger_benchmark_id, str)
            or not trigger_benchmark_id.strip()
            or len(trigger_benchmark_id) > 128
            or not isinstance(trigger_evidence_digest, str)
            or not _SHA256.fullmatch(trigger_evidence_digest)
        ):
            raise ValueError("benchmark trigger provenance is invalid")

        paths = self._validate_diff(diff)
        patch_digest = hashlib.sha256(diff.encode("utf-8")).hexdigest()
        identifier = hashlib.sha256(
            (
                f"{base_revision}\0{hypothesis}\0{patch_digest}\0"
                f"{trigger_benchmark_id or ''}\0{trigger_evidence_digest or ''}"
            ).encode()
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
            trigger_benchmark_id=trigger_benchmark_id,
            trigger_evidence_digest=trigger_evidence_digest,
        )

    def _validate_proposal(self, proposal: ImprovementProposal) -> None:
        if not isinstance(proposal, ImprovementProposal):
            raise ValueError("proposal must be an ImprovementProposal")
        if not isinstance(proposal.base_revision, str) or not _REVISION.fullmatch(proposal.base_revision):
            raise ValueError("proposal base revision is invalid")
        if not isinstance(proposal.diff, str) or hashlib.sha256(proposal.diff.encode()).hexdigest() != proposal.patch_digest:
            raise ValueError("proposal patch digest does not match its immutable diff")
        if self._validate_diff(proposal.diff) != proposal.file_paths:
            raise ValueError("proposal file manifest does not match its diff")
        if (
            not isinstance(proposal.checks, tuple)
            or not proposal.checks
            or len(proposal.checks) > self.limits.max_checks
            or any(not isinstance(check, str) or check not in _ALLOWED_CHECKS for check in proposal.checks)
            or len(set(proposal.checks)) != len(proposal.checks)
        ):
            raise ValueError("proposal check manifest is invalid")
        expected_identifier = hashlib.sha256(
            (
                f"{proposal.base_revision}\\0{proposal.hypothesis}\\0{proposal.patch_digest}\\0"
                f"{proposal.trigger_benchmark_id or ''}\\0{proposal.trigger_evidence_digest or ''}"
            ).encode()
        ).hexdigest()[:24]
        if proposal.identifier != expected_identifier:
            raise ValueError("proposal identifier does not match its content")
        if (
            proposal.rollback_plan.base_revision != proposal.base_revision
            or proposal.rollback_plan.proposal_digest != proposal.patch_digest
        ):
            raise ValueError("proposal rollback plan does not match its base or patch")
        if (proposal.trigger_benchmark_id is None) != (proposal.trigger_evidence_digest is None):
            raise ValueError("proposal benchmark provenance is incomplete")
        if proposal.trigger_benchmark_id is not None and (
            not isinstance(proposal.trigger_benchmark_id, str)
            or not proposal.trigger_benchmark_id.strip()
            or not isinstance(proposal.trigger_evidence_digest, str)
            or not _SHA256.fullmatch(proposal.trigger_evidence_digest)
        ):
            raise ValueError("proposal benchmark provenance is invalid")

    def propose_from_gap(
        self,
        gap: EvaluationGap,
        hypothesis: str,
        diff: str,
        checks: tuple[str, ...],
        *,
        base_revision: str,
    ) -> ImprovementProposal:
        """Package a candidate patch against a measurable, provenance-bound gap."""
        if not isinstance(gap, EvaluationGap) or not gap.is_gap:
            raise ValueError("evaluation evidence does not demonstrate a capability gap")
        return self.propose(
            hypothesis,
            diff,
            checks,
            base_revision=base_revision,
            trigger_benchmark_id=gap.benchmark_id,
            trigger_evidence_digest=gap.evidence_digest,
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
            "trigger_benchmark_id": proposal.trigger_benchmark_id,
            "trigger_evidence_digest": proposal.trigger_evidence_digest,
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
                    "filesystem_isolated": result.filesystem_isolated,
                    "process_isolated": result.process_isolated,
                    "network_disabled": result.network_disabled,
                    "credentials_unavailable": result.credentials_unavailable,
                    "stdout_digest": result.stdout_digest,
                    "stderr_digest": result.stderr_digest,
                }
                for result in results
            ],
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                default=lambda value: f"<{type(value).__name__}>",
            ).encode("utf-8")
        ).hexdigest()

    def verify(
        self,
        proposal: ImprovementProposal,
        results: tuple[SandboxCheckResult, ...],
    ) -> SandboxEvidence:
        self._validate_proposal(proposal)
        failure: str | None = None
        safe_results = (
            tuple(item for item in results if isinstance(item, SandboxCheckResult))
            if isinstance(results, tuple) else ()
        )
        if not isinstance(results, tuple) or len(results) != len(proposal.checks):
            failure = "sandbox evidence is incomplete or has unexpected check count"
        elif len(safe_results) != len(results):
            failure = "sandbox evidence contains an invalid result type"
        elif any(not isinstance(item.sandbox_id, str) for item in safe_results):
            failure = "sandbox evidence contains an invalid workspace identity"
        else:
            sandbox_ids = {item.sandbox_id for item in safe_results}
            if len(sandbox_ids) != 1 or not next(iter(sandbox_ids), "").strip():
                failure = "checks must run in one identified isolated workspace"
            else:
                for expected, result in zip(proposal.checks, safe_results, strict=True):
                    if result.check != expected:
                        failure = "sandbox evidence does not match the requested check order"
                        break
                    if (
                        not isinstance(result.sandbox_id, str)
                        or not result.sandbox_id.strip()
                        or isinstance(result.exit_code, bool)
                        or not isinstance(result.exit_code, int)
                        or not isinstance(result.timed_out, bool)
                    ):
                        failure = "sandbox result contains invalid identity or execution status"
                        break
                    isolation_requirements = (
                        (result.isolated_worktree, "isolated worktree"),
                        (result.base_read_only, "read-only base"),
                        (result.filesystem_isolated, "filesystem isolation"),
                        (result.process_isolated, "process isolation"),
                        (result.network_disabled, "network denial"),
                        (result.credentials_unavailable, "credential isolation"),
                    )
                    missing_control = next(
                        (label for enabled, label in isolation_requirements if enabled is not True),
                        None,
                    )
                    if missing_control is not None:
                        failure = f"sandbox {missing_control} was not attested"
                        break
                    if result.base_revision != proposal.base_revision:
                        failure = "sandbox checked a different source revision"
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
                    if (
                        not isinstance(result.stdout_digest, str)
                        or not isinstance(result.stderr_digest, str)
                        or not _SHA256.fullmatch(result.stdout_digest)
                        or not _SHA256.fullmatch(result.stderr_digest)
                    ):
                        failure = f"stdout/stderr digest is missing or invalid: {expected}"
                        break

        status = "verified_for_human_review" if failure is None else "checks_failed"
        digest = self._evidence_digest(proposal, safe_results, status, failure)
        return SandboxEvidence(
            proposal_id=proposal.identifier,
            base_revision=proposal.base_revision,
            results=safe_results,
            verified=failure is None,
            status=status,
            failure_reason=failure,
            evidence_digest=digest,
        )

    def run_checks(self, proposal: ImprovementProposal) -> SandboxEvidence:
        """Delegate checks to an explicitly injected isolated sandbox adapter."""
        self._validate_proposal(proposal)
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

    def record_human_approval(
        self,
        proposal: ImprovementProposal,
        evidence: SandboxEvidence,
        approval_reference: str,
    ) -> HumanApprovalRecord:
        self._validate_proposal(proposal)
        if not isinstance(evidence, SandboxEvidence):
            raise ValueError("only verified evidence for this proposal can be approved")
        recomputed = self.verify(proposal, evidence.results)
        if (
            not evidence.verified
            or evidence.status != "verified_for_human_review"
            or evidence.proposal_id != proposal.identifier
            or evidence.base_revision != proposal.base_revision
            or not isinstance(evidence.evidence_digest, str)
            or not _SHA256.fullmatch(evidence.evidence_digest)
            or recomputed.evidence_digest != evidence.evidence_digest
            or not recomputed.verified
        ):
            raise ValueError("only verified evidence for this proposal can be approved")
        if not isinstance(approval_reference, str) or not approval_reference.strip():
            raise ValueError("a human approval reference is required")
        return HumanApprovalRecord(
            proposal_id=proposal.identifier,
            proposal_digest=proposal.patch_digest,
            evidence_digest=evidence.evidence_digest,
            approval_reference=approval_reference.strip(),
        )
