"""Opt-in Linux OS-isolated executor backed by bubblewrap.

The organism never constructs this adapter implicitly.  It is a trusted
operator-side integration for running validated proposals in a fresh tmpfs
workspace with explicit mount, PID, IPC, UTS, user and network namespaces.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
import platform
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from artificial_human.development.self_development import (
    ImprovementProposal,
    SandboxCheckResult,
    SandboxLimits,
)

_ALLOWED_COMMANDS = {
    "ruff check .": ("ruff", ("check", ".")),
    "mypy src": ("mypy", ("src",)),
    "pytest -q": ("pytest", ("-q",)),
}
_PROBE_PREFIX = b"__AH_SANDBOX_PROBE__"
_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
_MAX_ARCHIVE_FILES = 20_000
_MAX_ARCHIVE_FILE_BYTES = 32 * 1024 * 1024
_RSS_SCAN_INTERVAL = 0.05

_RUNNER = r'''
from __future__ import annotations
import json
import os
import resource
import shutil
import socket
import subprocess
import sys
from pathlib import Path

CONFIG = json.loads(Path("/config.json").read_text(encoding="utf-8"))
LIMITS = CONFIG["limits"]

def probe():
    result = {}
    host_net = int(CONFIG["host_netns_inode"])
    host_pid = int(CONFIG["host_pidns_inode"])
    result["network_namespace"] = os.stat("/proc/self/ns/net").st_ino != host_net
    result["process_namespace"] = (
        os.stat("/proc/self/ns/pid").st_ino != host_pid and os.getppid() == 1
    )
    def mount_id(path):
        for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) > 5 and fields[4].replace("\\040", " ") == path:
                return fields[0]
        return None
    result["workspace_tmpfs"] = (
        mount_id("/workspace") is not None
        and mount_id("/staging") is not None
        and mount_id("/workspace") != mount_id("/staging")
    )
    result["source_not_mounted"] = not os.path.exists(CONFIG["host_source_root"])
    result["host_marker_not_visible"] = not os.path.exists(CONFIG["host_marker"])
    probe_path = "/staging/.artificial-human-readonly-probe"
    try:
        with open(probe_path, "xb") as handle:
            handle.write(b"must-not-be-writable")
        os.unlink(probe_path)
        result["staging_read_only"] = False
    except OSError:
        result["staging_read_only"] = True
    allowed_env = set(CONFIG["allowed_env"])
    result["environment_cleared"] = set(os.environ).issubset(allowed_env)
    network_reachable = False
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=0.25):
            network_reachable = True
    except OSError:
        network_reachable = False
    result["network_unreachable"] = not network_reachable
    return result

def main():
    checks = probe()
    payload = {
        "network_disabled": checks["network_namespace"] and checks["network_unreachable"],
        "process_isolated": checks["process_namespace"],
        "base_read_only": checks["staging_read_only"],
        "filesystem_isolated": (
            checks["workspace_tmpfs"]
            and checks["source_not_mounted"]
            and checks["host_marker_not_visible"]
        ),
        "credentials_unavailable": checks["environment_cleared"],
        "isolated_worktree": (
            checks["workspace_tmpfs"]
            and checks["source_not_mounted"]
            and checks["host_marker_not_visible"]
            and checks["staging_read_only"]
        ),
    }
    print("__AH_SANDBOX_PROBE__" + json.dumps(payload, sort_keys=True), flush=True)
    if not all(payload.values()):
        return 125

    os.umask(0o077)
    cpu_seconds = max(1, int(LIMITS["wall_seconds"]))
    memory_bytes = int(LIMITS["memory_mb"] * 1024 * 1024)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    except (AttributeError, OSError, ValueError):
        pass

    project = Path("/workspace/project")
    shutil.copytree("/staging", project, symlinks=False)
    patch = Path("/input.patch").read_bytes()
    applied = subprocess.run(
        [CONFIG["git"], "apply", "--no-ext-diff", "--whitespace=nowarn", "-"],
        cwd=project,
        input=patch,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env={
            "PATH": os.environ["PATH"],
            "HOME": os.environ["HOME"],
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0",
        },
    )
    if applied.returncode != 0:
        sys.stderr.buffer.write(applied.stderr[:8192])
        return 126

    os.chdir(project)
    command = CONFIG["command"]
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "TMPDIR": os.environ["TMPDIR"],
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "PYTHONPATH": "/workspace/project/src",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PIP_NO_INDEX": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "RUFF_CACHE_DIR": "/tmp/ruff-cache",
        "MYPY_CACHE_DIR": "/tmp/mypy-cache",
    }
    os.execve(command[0], command, env)

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        print("sandbox runner failed: " + type(exc).__name__, file=sys.stderr, flush=True)
        raise
'''


class BubblewrapUnavailable(RuntimeError):
    """The host cannot provide the required bubblewrap isolation controls."""


@dataclass(slots=True)
class _Session:
    proposal_id: str
    base_revision: str
    sandbox_id: str
    temporary: tempfile.TemporaryDirectory[str]
    root: Path
    staging: Path
    marker: Path


class BubblewrapSandboxExecutor:
    """Run allowlisted checks in fresh, networkless bubblewrap namespaces.

    Only the immutable Git archive and the proposed diff enter the sandbox.
    The live repository, host home, environment secrets and Git metadata are
    not mounted. Candidate files are materialised only on a size-limited tmpfs.
    """

    def __init__(
        self,
        source_root: str | os.PathLike[str],
        *,
        bubblewrap_path: str | os.PathLike[str] | None = None,
        work_parent: str | os.PathLike[str] | None = None,
        workspace_size_mb: int = 256,
        tmp_size_mb: int = 32,
    ) -> None:
        if platform.system() != "Linux":
            raise BubblewrapUnavailable("bubblewrap sandbox execution requires Linux")
        if (
            isinstance(workspace_size_mb, bool)
            or not isinstance(workspace_size_mb, int)
            or not 32 <= workspace_size_mb <= 512
        ):
            raise ValueError("workspace_size_mb must be an integer in [32, 512]")
        if isinstance(tmp_size_mb, bool) or not isinstance(tmp_size_mb, int) or not 8 <= tmp_size_mb <= 128:
            raise ValueError("tmp_size_mb must be an integer in [8, 128]")

        binary = str(bubblewrap_path) if bubblewrap_path is not None else shutil.which("bwrap")
        if not binary:
            raise BubblewrapUnavailable("bubblewrap (bwrap) is not installed")
        resolved_binary = Path(binary).resolve()
        try:
            info = resolved_binary.stat()
        except OSError as exc:
            raise BubblewrapUnavailable("bubblewrap binary cannot be inspected") from exc
        if not resolved_binary.is_absolute() or not stat.S_ISREG(info.st_mode):
            raise BubblewrapUnavailable("bubblewrap must resolve to a regular executable")
        if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise BubblewrapUnavailable("bubblewrap binary must not be group/world writable")
        if not os.access(resolved_binary, os.X_OK):
            raise BubblewrapUnavailable("bubblewrap binary is not executable")

        self.bubblewrap_path = str(resolved_binary)
        self.source_root = Path(source_root).resolve(strict=True)
        if not self.source_root.is_dir():
            raise ValueError("source_root must be a directory")
        git = shutil.which("git")
        if not git:
            raise BubblewrapUnavailable("git is required to stage an immutable source archive")
        self.git_path = str(Path(git).resolve())
        self.python_path = str(Path(sys.executable).resolve())
        self.python_prefix = Path(sys.prefix).resolve()
        self.python_base_prefix = Path(sys.base_prefix).resolve()
        if not self._is_visible_system_path(self.git_path):
            raise BubblewrapUnavailable("git is outside the mounted system/Python toolchain")
        protected_roots = (
            Path("/usr"),
            Path("/bin"),
            Path("/sbin"),
            Path("/lib"),
            Path("/lib64"),
            self.python_prefix,
            self.python_base_prefix,
        )
        if any(self._under(self.source_root, root) for root in protected_roots):
            raise ValueError("source_root must not be inside a mounted system or Python toolchain directory")
        self.commands: dict[str, tuple[str, ...]] = {}
        for check, (program, arguments) in _ALLOWED_COMMANDS.items():
            located = shutil.which(program)
            if not located:
                raise BubblewrapUnavailable(f"required allowlisted tool is not installed: {program}")
            executable = str(Path(located).resolve())
            if not self._is_visible_system_path(executable):
                raise BubblewrapUnavailable(
                    f"{program} is outside the mounted system/Python toolchain; install it in the active environment"
                )
            self.commands[check] = (executable, *arguments)
        if not self._is_visible_system_path(self.python_path):
            raise BubblewrapUnavailable("the active Python interpreter is outside the mounted toolchain")

        self.work_parent = Path(work_parent).resolve() if work_parent else None
        if self.work_parent is not None:
            if self._under(self.work_parent, self.source_root):
                raise ValueError("work_parent must be outside the source checkout")
            self.work_parent.mkdir(parents=True, exist_ok=True)
        self.workspace_size_mb = workspace_size_mb
        self.tmp_size_mb = tmp_size_mb
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.RLock()
        atexit.register(self.close)

    @staticmethod
    def _under(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    def _is_visible_system_path(self, raw_path: str) -> bool:
        path = Path(raw_path).resolve()
        roots = (Path("/usr"), Path("/bin"), Path("/sbin"), Path("/lib"), Path("/lib64"))
        prefixes = (self.python_prefix, self.python_base_prefix)
        return any(self._under(path, root) for root in roots) or any(
            self._under(path, root) for root in prefixes
        )

    def _git(self, arguments: tuple[str, ...], *, timeout: float = 20.0) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [self.git_path, *arguments],
            cwd=self.source_root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            check=False,
            env={
                "PATH": "/usr/bin:/bin",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_TERMINAL_PROMPT": "0",
            },
        )

    @staticmethod
    def _extract_archive(archive_path: Path, destination: Path) -> None:
        total_bytes = 0
        total_files = 0
        destination.mkdir(mode=0o700)
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive:
                relative = PurePosixPath(member.name)
                if (
                    relative.is_absolute()
                    or not relative.parts
                    or any(part in {"", ".", ".."} for part in relative.parts)
                    or "\\" in member.name
                ):
                    raise RuntimeError("Git archive contains an unsafe path")
                target = destination.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise RuntimeError("Git archive contains a link or special file")
                total_files += 1
                total_bytes += member.size
                if total_files > _MAX_ARCHIVE_FILES or member.size > _MAX_ARCHIVE_FILE_BYTES:
                    raise RuntimeError("Git archive exceeds file-count or per-file limits")
                if total_bytes > _MAX_ARCHIVE_BYTES:
                    raise RuntimeError("Git archive exceeds total size limit")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError("Git archive contains an unreadable file")
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                descriptor = os.open(target, flags, member.mode & 0o755)
                with os.fdopen(descriptor, "wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                source.close()

    def _create_session(self, proposal: ImprovementProposal) -> _Session:
        result = self._git(("rev-parse", "--show-toplevel"))
        if result.returncode != 0:
            raise RuntimeError("source_root is not a readable Git working tree")
        top = Path(result.stdout.decode("utf-8", "strict").strip()).resolve()
        if top != self.source_root:
            raise RuntimeError("source_root must be the Git repository root")
        exists = self._git(("cat-file", "-e", f"{proposal.base_revision}^{{commit}}"))
        if exists.returncode != 0:
            raise RuntimeError("proposal base revision is not present in the local Git object database")

        temporary = tempfile.TemporaryDirectory(
            prefix="artificial-human-sandbox-",
            dir=str(self.work_parent) if self.work_parent else None,
        )
        root = Path(temporary.name)
        archive_path = root / "source.tar"
        with archive_path.open("wb") as archive:
            staged = subprocess.run(
                [self.git_path, "archive", "--format=tar", proposal.base_revision],
                cwd=self.source_root,
                stdin=subprocess.DEVNULL,
                stdout=archive,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
                env={
                    "PATH": "/usr/bin:/bin",
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_GLOBAL": os.devnull,
                    "GIT_TERMINAL_PROMPT": "0",
                },
            )
        if staged.returncode != 0:
            temporary.cleanup()
            raise RuntimeError("failed to stage the immutable proposal base revision")
        if archive_path.stat().st_size > _MAX_ARCHIVE_BYTES:
            temporary.cleanup()
            raise RuntimeError("Git archive exceeds total size limit")

        staging = root / "staging"
        try:
            self._extract_archive(archive_path, staging)
        except Exception:
            temporary.cleanup()
            raise
        archive_path.unlink()
        marker = root / "host-only-marker"
        marker.write_text("must not be visible in the sandbox", encoding="utf-8")
        return _Session(
            proposal_id=proposal.identifier,
            base_revision=proposal.base_revision,
            sandbox_id="bwrap-" + uuid.uuid4().hex,
            temporary=temporary,
            root=root,
            staging=staging,
            marker=marker,
        )

    def _get_session(self, proposal: ImprovementProposal) -> _Session:
        with self._lock:
            session = self._sessions.get(proposal.identifier)
            if session is not None:
                if session.base_revision != proposal.base_revision:
                    raise RuntimeError("proposal identifier was reused with a different base revision")
                return session
            session = self._create_session(proposal)
            self._sessions[proposal.identifier] = session
            return session

    @staticmethod
    def _add_ro_directory(arguments: list[str], path: Path) -> None:
        if path.exists() and path.is_dir():
            resolved = path.resolve()
            if resolved not in (Path("/"),):
                arguments.extend(["--ro-bind", str(resolved), str(resolved)])

    def _mount_arguments(self, session: _Session, limits: SandboxLimits, config_path: Path, patch_path: Path, runner_path: Path) -> list[str]:
        args = [
            self.bubblewrap_path,
            "--unshare-user",
            "--unshare-ipc",
            "--unshare-pid",
            "--unshare-net",
            "--unshare-uts",
            "--new-session",
            "--die-with-parent",
            "--cap-drop", "ALL",
            "--hostname", "artificial-human-sandbox",
            "--clearenv",
        ]
        mounted_roots: list[Path] = []
        self._add_ro_directory(args, Path("/usr"))
        mounted_roots.append(Path("/usr"))

        for system_path in (Path("/bin"), Path("/sbin"), Path("/lib"), Path("/lib64")):
            if system_path.is_symlink():
                args.extend(["--symlink", os.readlink(system_path), str(system_path)])
            elif system_path.is_dir():
                self._add_ro_directory(args, system_path)
                mounted_roots.append(system_path.resolve())

        prefixes = {self.python_prefix, self.python_base_prefix}
        for prefix in sorted(prefixes, key=lambda item: len(str(item))):
            if prefix.exists() and not any(self._under(prefix, root) for root in mounted_roots):
                self._add_ro_directory(args, prefix)
                mounted_roots.append(prefix)

        args.extend(["--dir", "/etc"])
        passwd = session.root / "passwd"
        group = session.root / "group"
        passwd.write_text(
            "root:x:0:0:root:/tmp/home:/bin/sh\n"
            "nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n",
            encoding="utf-8",
        )
        group.write_text("root:x:0:\nnogroup:x:65534:\n", encoding="utf-8")
        for host_path, sandbox_path in (
            (Path("/etc/ld.so.cache"), "/etc/ld.so.cache"),
        ):
            if host_path.is_file():
                args.extend(["--ro-bind", str(host_path), sandbox_path])
        args.extend(["--ro-bind", str(passwd), "/etc/passwd", "--ro-bind", str(group), "/etc/group"])
        args.extend([
            "--proc", "/proc",
            "--dev", "/dev",
            "--size", str(self.tmp_size_mb * 1024 * 1024), "--tmpfs", "/tmp",
            "--size", str(self.workspace_size_mb * 1024 * 1024), "--tmpfs", "/workspace",
            "--dir", "/tmp/home",
            "--ro-bind", str(session.staging), "/staging",
            "--ro-bind", str(config_path), "/config.json",
            "--ro-bind", str(patch_path), "/input.patch",
            "--ro-bind", str(runner_path), "/runner.py",
            "--chdir", "/",
            "--setenv", "HOME", "/tmp/home",
            "--setenv", "TMPDIR", "/tmp",
            "--setenv", "PATH", os.pathsep.join([
                str(self.python_prefix / "bin"),
                str(self.python_base_prefix / "bin"),
                "/usr/bin",
                "/bin",
                "/usr/local/bin",
            ]),
            "--setenv", "LANG", "C.UTF-8",
            "--setenv", "LC_ALL", "C.UTF-8",
            "--setenv", "TZ", "UTC",
            "--setenv", "PYTHONNOUSERSITE", "1",
            "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            "--setenv", "PIP_NO_INDEX", "1",
            "--setenv", "PIP_DISABLE_PIP_VERSION_CHECK", "1",
            "--setenv", "PWD", "/",
            "--",
            self.python_path,
            "/runner.py",
            "/config.json",
        ])
        return args

    @staticmethod
    def _read_process_tree_rss(root_pid: int) -> int:
        total_kib = 0
        pending = [root_pid]
        seen: set[int] = set()
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            process = Path("/proc") / str(pid)
            try:
                children = (process / "task" / str(pid) / "children").read_text(encoding="ascii")
                pending.extend(int(value) for value in children.split())
                status = (process / "status").read_text(encoding="ascii")
            except (OSError, ValueError):
                continue
            for line in status.splitlines():
                if line.startswith("VmRSS:"):
                    try:
                        total_kib += int(line.split()[1])
                    except (ValueError, IndexError):
                        pass
                    break
        return total_kib * 1024

    @staticmethod
    def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def run_check(
        self,
        proposal: ImprovementProposal,
        check: str,
        limits: SandboxLimits,
    ) -> SandboxCheckResult:
        if check not in _ALLOWED_COMMANDS:
            raise ValueError("sandbox executor only accepts the fixed check allowlist")
        if not math.isfinite(limits.max_wall_seconds) or limits.max_wall_seconds <= 0:
            raise ValueError("wall-clock limit is invalid")
        session = self._get_session(proposal)
        command = self.commands[check]
        run_id = uuid.uuid4().hex
        patch_path = session.root / ("proposal-" + run_id + ".diff")
        runner_path = session.root / ("runner-" + run_id + ".py")
        config_path = session.root / ("config-" + run_id + ".json")
        patch_path.write_text(proposal.diff, encoding="utf-8")
        runner_path.write_text(_RUNNER, encoding="utf-8")
        host_net_inode = os.stat("/proc/self/ns/net").st_ino
        host_pid_inode = os.stat("/proc/self/ns/pid").st_ino
        allowed_env = [
            "HOME", "TMPDIR", "PATH", "LANG", "LC_ALL", "TZ", "PWD",
            "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE",
            "PIP_NO_INDEX", "PIP_DISABLE_PIP_VERSION_CHECK",
        ]
        config = {
            "command": list(command),
            "git": self.git_path,
            "limits": {
                "wall_seconds": limits.max_wall_seconds,
                "memory_mb": limits.max_memory_mb,
                "output_bytes": limits.max_output_bytes,
            },
            "host_netns_inode": host_net_inode,
            "host_pidns_inode": host_pid_inode,
            "host_source_root": str(self.source_root),
            "host_marker": str(session.marker),
            "allowed_env": allowed_env,
        }
        config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
        command_line = self._mount_arguments(session, limits, config_path, patch_path, runner_path)

        started = time.monotonic()
        stdout_data = bytearray()
        stderr_data = bytearray()
        output_count = 0
        peak_memory = 0.0
        timed_out = False
        overflow = False
        env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC"}
        process = subprocess.Popen(
            command_line,
            cwd="/",
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            close_fds=True,
            start_new_session=True,
        )
        assert process.stdout is not None and process.stderr is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        try:
            while process.poll() is None or selector.get_map():
                now = time.monotonic()
                elapsed = now - started
                rss_bytes = self._read_process_tree_rss(process.pid)
                peak_memory = max(peak_memory, rss_bytes / (1024 * 1024))
                if elapsed > limits.max_wall_seconds:
                    timed_out = True
                    self._kill_process_group(process)
                if peak_memory > limits.max_memory_mb:
                    timed_out = True
                    self._kill_process_group(process)
                for key, _ in selector.select(timeout=_RSS_SCAN_INTERVAL):
                    stream_name = key.data
                    try:
                        chunk = os.read(key.fileobj.fileno(), 65536)
                    except OSError:
                        chunk = b""
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    output_count += len(chunk)
                    target = stdout_data if stream_name == "stdout" else stderr_data
                    remaining = max(0, limits.max_output_bytes + 1 - len(stdout_data) - len(stderr_data))
                    if remaining:
                        target.extend(chunk[:remaining])
                    if output_count > limits.max_output_bytes:
                        overflow = True
                        self._kill_process_group(process)
                if process.poll() is not None and not selector.get_map():
                    break
        finally:
            selector.close()
        exit_code = process.wait()
        elapsed = time.monotonic() - started
        stdout_hash = hashlib.sha256(bytes(stdout_data)).hexdigest()
        stderr_hash = hashlib.sha256(bytes(stderr_data)).hexdigest()
        probe_ok = {
            "isolated_worktree": False,
            "base_read_only": False,
            "filesystem_isolated": False,
            "process_isolated": False,
            "network_disabled": False,
            "credentials_unavailable": False,
        }
        first_line, separator, remainder = bytes(stdout_data).partition(b"\n")
        if first_line.startswith(_PROBE_PREFIX):
            try:
                data = json.loads(first_line[len(_PROBE_PREFIX):].decode("utf-8"))
                probe_ok = {name: data.get(name) is True for name in probe_ok}
                stdout_hash = hashlib.sha256(remainder if separator else b"").hexdigest()
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
        if overflow:
            exit_code = exit_code if exit_code != 0 else 125
        return SandboxCheckResult(
            check=check,
            base_revision=proposal.base_revision,
            sandbox_id=session.sandbox_id,
            exit_code=exit_code,
            elapsed_seconds=elapsed,
            peak_memory_mb=peak_memory,
            output_bytes=output_count,
            timed_out=timed_out,
            isolated_worktree=probe_ok["isolated_worktree"],
            base_read_only=probe_ok["base_read_only"],
            filesystem_isolated=probe_ok["filesystem_isolated"],
            process_isolated=probe_ok["process_isolated"],
            network_disabled=probe_ok["network_disabled"],
            credentials_unavailable=probe_ok["credentials_unavailable"],
            stdout_digest=stdout_hash,
            stderr_digest=stderr_hash,
        )

    def finish(self, proposal: ImprovementProposal) -> None:
        """Remove the host-side archive/staging files after the proposal run."""
        with self._lock:
            session = self._sessions.pop(proposal.identifier, None)
        if session is not None:
            session.temporary.cleanup()

    def close(self) -> None:
        with self._lock:
            sessions = tuple(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.temporary.cleanup()
