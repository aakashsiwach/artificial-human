from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
from pathlib import Path

import pytest

from artificial_human.development.bubblewrap_sandbox import (
    BubblewrapSandboxExecutor,
    BubblewrapUnavailable,
)
from artificial_human.development.self_development import SafeSelfDevelopment


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
        },
    )
    return result.stdout.strip()


def _proposal_diff() -> str:
    content = (
        "from sandbox_demo import VALUE\n"
        "\n"
        "\n"
        "def test_candidate_value() -> None:\n"
        "    assert VALUE == 7\n"
    )
    encoded = content.encode("utf-8")
    blob_id = hashlib.sha1(f"blob {len(encoded)}\\0".encode("ascii") + encoded).hexdigest()[:7]
    lines = content.rstrip("\n").splitlines()
    patch_lines = [
        "diff --git a/tests/test_candidate.py b/tests/test_candidate.py",
        "new file mode 100644",
        f"index 0000000..{blob_id}",
        "--- /dev/null",
        "+++ b/tests/test_candidate.py",
        f"@@ -0,0 +1,{len(lines)} @@",
        *("+" + line for line in lines),
        "",
    ]
    return "\n".join(patch_lines)


def _temporary_repository(root: Path) -> str:
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[tool.ruff]\ntarget-version = "py312"\n'
        '[tool.mypy]\npython_version = "3.12"\n',
        encoding="utf-8",
    )
    package = root / "src" / "sandbox_demo"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE: int = 7\n", encoding="utf-8")
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_base.py").write_text(
        "from sandbox_demo import VALUE\n\n"
        "def test_base_value() -> None:\n"
        "    assert VALUE == 7\n",
        encoding="utf-8",
    )
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Sandbox Integration")
    _git(root, "config", "user.email", "sandbox@example.invalid")
    _git(root, "add", "pyproject.toml", "src", "tests")
    _git(root, "commit", "-q", "-m", "sandbox base")
    return _git(root, "rev-parse", "HEAD")


@pytest.mark.skipif(
    platform.system() != "Linux" or shutil.which("bwrap") is None,
    reason="the real bubblewrap integration test requires Linux and bwrap",
)
def test_real_bubblewrap_runs_allowlisted_checks_in_isolated_tmpfs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    repository = tmp_path / "source"
    base_revision = _temporary_repository(repository)
    host_only_file = repository / "host-only-untracked.txt"
    host_only_file.write_text("must not enter the source archive", encoding="utf-8")
    monkeypatch.setenv("ARTIFICIAL_HUMAN_SANDBOX_SECRET_SENTINEL", "host-secret-not-mounted")

    proposal = SafeSelfDevelopment().propose(
        "Add a regression test without modifying the host checkout",
        _proposal_diff(),
        ("ruff check .", "mypy src", "pytest -q"),
        base_revision=base_revision,
    )
    executor = BubblewrapSandboxExecutor(
        repository,
        work_parent=tmp_path / "executor-work",
    )
    guard = SafeSelfDevelopment(executor=executor)

    try:
        evidence = guard.run_checks(proposal)
    finally:
        executor.close()

    assert evidence.verified, f"{evidence.failure_reason}; results={evidence.results!r}"
    assert evidence.status == "verified_for_human_review"
    assert len(evidence.results) == 3
    assert len({item.sandbox_id for item in evidence.results}) == 1
    for result in evidence.results:
        assert result.exit_code == 0
        assert not result.timed_out
        assert result.isolated_worktree
        assert result.base_read_only
        assert result.filesystem_isolated
        assert result.process_isolated
        assert result.network_disabled
        assert result.credentials_unavailable
        assert result.peak_memory_mb <= 1536
        assert result.output_bytes <= 100_000
        assert len(result.stdout_digest) == 64
        assert len(result.stderr_digest) == 64

    # The proposal is applied only to the namespace's tmpfs workspace.
    assert not (repository / "tests" / "test_candidate.py").exists()
    assert host_only_file.read_text(encoding="utf-8") == "must not enter the source archive"
    assert os.environ["ARTIFICIAL_HUMAN_SANDBOX_SECRET_SENTINEL"] == "host-secret-not-mounted"


def test_executor_rejects_unavailable_bubblewrap(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "source"
    _temporary_repository(repository)
    with pytest.raises(BubblewrapUnavailable, match="bubblewrap"):
        BubblewrapSandboxExecutor(repository, bubblewrap_path=tmp_path / "missing-bwrap")


def test_ci_requires_bubblewrap_when_requested() -> None:
    if os.getenv("ARTIFICIAL_HUMAN_REQUIRE_BWRAP") == "1":
        assert platform.system() == "Linux"
        assert shutil.which("bwrap"), "CI requires the bubblewrap binary"
