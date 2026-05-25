"""Podman + gVisor sandbox for per-cell agent execution.

Drives the `podman` CLI via subprocess — we deliberately don't pull in a
Python container SDK (one less dep, one less moving part). Each `Sandbox` owns
one container; cleanup is idempotent and runs on every exit path (context
manager, explicit `stop()`, exceptions, signal). Network defaults to `none`;
opt in per call.

The runtime args (`--runtime=runsc --security-opt label=disable
--runtime-flag=ignore-cgroups`) are the working incantation on Fedora 43 with
rootless Podman 5.x and gVisor `runsc release-20260520.0`. The two extra
flags are not optional under that combination — see
`containers/Containerfile.agent-sandbox` and the docstring at the top of
`~/.config/containers/containers.conf` for the why.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import subprocess
import tarfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Self


@dataclass(slots=True, frozen=True)
class SandboxExecResult:
    """Outcome of one `Sandbox.exec` call.

    `stdout`/`stderr` are captured bytes (never `None`). `exit_code` is the
    process exit code on the *container* side; `127` if podman itself never
    got far enough to run the command. `duration_ms` includes podman overhead.
    `timed_out` is `True` iff we killed the exec because it ran past
    `timeout`.
    """

    stdout: bytes
    stderr: bytes
    exit_code: int
    duration_ms: int
    timed_out: bool


class SandboxError(RuntimeError):
    """Raised when a sandbox-level operation fails before exec can run.

    Exec-level failures (non-zero exit codes inside the container) are
    returned in `SandboxExecResult.exit_code`, not raised.
    """


@dataclass(slots=True)
class _PodmanArgs:
    """Assembled `podman run` invocation for an idle container.

    Pulled out as a dataclass so `_build_run_argv` is purely functional and
    cheap to unit-test without invoking podman.
    """

    image: str
    name: str
    runtime: str
    network: str
    mem_limit: str
    cpu_limit: float
    env: dict[str, str] = field(default_factory=dict)

    def to_argv(self) -> list[str]:
        argv = [
            "podman",
            "run",
            "--detach",
            "--rm",
            "--name",
            self.name,
            f"--memory={self.mem_limit}",
            f"--cpus={self.cpu_limit}",
            "--read-only",
            # /tmp and /workspace need to be writable for tool calls + uploads
            "--tmpfs=/tmp:rw,exec,nosuid,size=64m",
            "--tmpfs=/workspace:rw,exec,nosuid,size=256m",
        ]
        if self.runtime == "runsc":
            # gVisor-specific quirks required for rootless on Fedora 43.
            argv += [
                "--runtime=runsc",
                "--security-opt=label=disable",
                "--runtime-flag=ignore-cgroups",
            ]
        elif self.runtime:
            argv += [f"--runtime={self.runtime}"]
        argv += [f"--network={self.network}"]
        for k, v in sorted(self.env.items()):
            argv += ["--env", f"{k}={v}"]
        argv += [
            self.image,
            # Keep the container alive without consuming CPU; `exec` does the
            # real work. `sleep infinity` is the smallest binary we can rely
            # on inside fedora-minimal.
            "sleep",
            "infinity",
        ]
        return argv


def _build_run_argv(
    *,
    image: str,
    name: str,
    runtime: str,
    network: str,
    mem_limit: str,
    cpu_limit: float,
    env: dict[str, str] | None,
) -> list[str]:
    """Pure helper exposed for unit tests; mirrors `_PodmanArgs.to_argv`."""

    return _PodmanArgs(
        image=image,
        name=name,
        runtime=runtime,
        network=network,
        mem_limit=mem_limit,
        cpu_limit=cpu_limit,
        env=env or {},
    ).to_argv()


class Sandbox:
    """One-shot Podman + gVisor sandbox.

    Lifecycle:
        with Sandbox(workspace_files={"in.txt": b"hi"}) as box:
            result = box.exec(["cat", "in.txt"])

    `start()` is called automatically by `__enter__`; if you instantiate
    without the context manager you MUST call `stop()` in a `finally` block.
    """

    def __init__(
        self,
        image: str = "lab-agent-sandbox:0.1",
        runtime: str = "runsc",
        network: str | list[str] = "none",
        mem_limit: str = "1g",
        cpu_limit: float = 2.0,
        time_limit_sec: int = 120,
        workspace_files: dict[str, bytes] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        if isinstance(network, list):
            # v0.1: ephemeral netavark allow-lists are TODO for 6c. For now
            # we degrade to no-network so the wrong-default never silently
            # grants egress to the public internet.
            # NOTE: revisit in 6c when http_fetch lands.
            self._network_arg = "none"
        elif network in ("none", "host"):
            self._network_arg = network
        else:
            raise ValueError(
                f"network must be 'none', 'host', or a list of allowed hosts; got {network!r}"
            )

        self.image = image
        self.runtime = runtime
        self.mem_limit = mem_limit
        self.cpu_limit = cpu_limit
        self.time_limit_sec = time_limit_sec
        self.workspace_files = workspace_files or {}
        self.env = env or {}
        # Container name: short + unique + recognisable in `podman ps`.
        self.container_name = f"lab-sandbox-{uuid.uuid4().hex[:12]}"
        self._started = False
        self._stopped = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Create + run the container and stage workspace files into it.

        Idempotent: calling `start` twice is a no-op (the second call
        silently returns).
        """

        if self._started:
            return
        argv = _build_run_argv(
            image=self.image,
            name=self.container_name,
            runtime=self.runtime,
            network=self._network_arg,
            mem_limit=self.mem_limit,
            cpu_limit=self.cpu_limit,
            env=self.env,
        )
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                check=False,
                timeout=60,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxError(f"podman run timed out: {' '.join(argv)}") from exc
        if proc.returncode != 0:
            raise SandboxError(
                f"podman run failed (exit {proc.returncode}): "
                f"{proc.stderr.decode(errors='replace').strip()}"
            )
        self._started = True
        if self.workspace_files:
            self._stage_workspace_files()

    def stop(self) -> None:
        """Force-stop and remove the container. Idempotent and best-effort.

        Safe to call from `__exit__`, signal handlers, or in a `finally`
        block — never raises.
        """

        if self._stopped:
            return
        self._stopped = True
        if not self._started:
            return
        # `--time 0` skips the SIGTERM grace; `rm -f` covers the case where
        # the container is already gone.
        with contextlib.suppress(subprocess.SubprocessError, OSError):
            subprocess.run(
                ["podman", "rm", "-f", "--time", "0", self.container_name],
                capture_output=True,
                check=False,
                timeout=30,
            )

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # exec
    # ------------------------------------------------------------------

    def exec(
        self,
        cmd: list[str],
        stdin: bytes | None = None,
        timeout: int | None = None,
    ) -> SandboxExecResult:
        """Run `cmd` inside the container; return captured output.

        Hard-killed at `timeout` (defaults to `self.time_limit_sec`); we set
        both the Python-side `subprocess.run(timeout=...)` and don't trust it
        alone — `podman exec` doesn't propagate SIGKILL into the container
        gracefully, so the wall-time enforcement is belt-and-braces with the
        in-container `pkill` cleanup.
        """

        if not self._started:
            raise SandboxError("Sandbox.start() must be called before exec()")
        if self._stopped:
            raise SandboxError("Sandbox already stopped")
        effective_timeout = timeout if timeout is not None else self.time_limit_sec
        argv = ["podman", "exec"]
        if stdin is not None:
            argv.append("--interactive")
        argv.append(self.container_name)
        argv.extend(cmd)
        start_ns = time.monotonic_ns()
        timed_out = False
        try:
            proc = subprocess.run(
                argv,
                input=stdin,
                capture_output=True,
                check=False,
                timeout=effective_timeout,
            )
            stdout = proc.stdout or b""
            stderr = proc.stderr or b""
            exit_code = proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or b""
            stderr = exc.stderr or b""
            exit_code = 124  # GNU `timeout` convention
            # Try to kill the in-flight process; we don't care if it fails.
            with contextlib.suppress(subprocess.SubprocessError, OSError):
                subprocess.run(
                    [
                        "podman",
                        "exec",
                        self.container_name,
                        "pkill",
                        "-9",
                        "-f",
                        cmd[0] if cmd else "",
                    ],
                    capture_output=True,
                    check=False,
                    timeout=5,
                )
        duration_ms = (time.monotonic_ns() - start_ns) // 1_000_000
        return SandboxExecResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_ms=int(duration_ms),
            timed_out=timed_out,
        )

    # ------------------------------------------------------------------
    # workspace I/O
    # ------------------------------------------------------------------

    def read_workspace_file(self, path: str) -> bytes:
        """Read a file under `/workspace` out of the container.

        `path` is the path *inside* `/workspace` (e.g. `"out.txt"`, NOT
        `"/workspace/out.txt"`); leading slashes are stripped.

        Implementation note: we use `tar` via `podman exec`, NOT `podman cp`.
        Under gVisor, `/workspace` is a tmpfs mounted *inside* the sandbox by
        the runsc gofer; `podman cp` looks at the host-side container storage
        layer and sees an empty mountpoint. Tar-via-exec runs inside the
        sandbox, sees the real contents, and streams the bytes back on the
        exec stdout channel.
        """

        relpath = path.lstrip("/")
        # Use podman exec directly (not self.exec) so we capture the raw tar
        # stream without going through the SandboxExecResult parsing path.
        proc = subprocess.run(
            [
                "podman",
                "exec",
                self.container_name,
                "tar",
                "-C",
                "/workspace",
                "-cf",
                "-",
                relpath,
            ],
            capture_output=True,
            check=False,
            timeout=30,
        )
        if proc.returncode != 0:
            raise SandboxError(
                f"read_workspace_file({path!r}) failed: "
                f"{proc.stderr.decode(errors='replace').strip()}"
            )
        with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tf:
            member = next((m for m in tf.getmembers() if m.isfile()), None)
            if member is None:
                raise SandboxError(f"{path!r} is not a regular file inside sandbox")
            extracted = tf.extractfile(member)
            if extracted is None:
                raise SandboxError(f"could not extract {path!r} from sandbox tar")
            return extracted.read()

    def list_workspace_files(self) -> list[str]:
        """List files under `/workspace` (relative paths, sorted).

        Uses `find` inside the container — the host can't see `/workspace`
        directly since it's a tmpfs inside the container.
        """

        res = self.exec(
            ["find", "/workspace", "-type", "f", "-printf", "%P\\n"],
            timeout=15,
        )
        if res.exit_code != 0:
            raise SandboxError(
                f"list_workspace_files failed (exit {res.exit_code}): "
                f"{res.stderr.decode(errors='replace').strip()}"
            )
        out = res.stdout.decode("utf-8", errors="replace").strip()
        if not out:
            return []
        return sorted(line for line in out.split("\n") if line)

    def _stage_workspace_files(self) -> None:
        """Copy `self.workspace_files` into `/workspace`.

        Uses a single in-memory tar streamed to `tar -x` *inside* the
        container (via `podman exec --interactive`). We can't use `podman cp`
        here because the destination is a gVisor-managed tmpfs that podman's
        host-side copier can't reach. One exec, one round-trip, regardless of
        file count.
        """

        if shutil.which("podman") is None:
            raise SandboxError("podman not found in PATH")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            for raw_path, content in self.workspace_files.items():
                relpath = raw_path.lstrip("/")
                info = tarfile.TarInfo(name=relpath)
                info.size = len(content)
                info.mode = 0o644
                # uid/gid 10001 = `agent` user inside the image.
                info.uid = 10001
                info.gid = 10001
                tf.addfile(info, io.BytesIO(content))
        buf.seek(0)
        proc = subprocess.run(
            [
                "podman",
                "exec",
                "--interactive",
                self.container_name,
                "tar",
                "-C",
                "/workspace",
                "-xf",
                "-",
            ],
            input=buf.getvalue(),
            capture_output=True,
            check=False,
            timeout=60,
        )
        if proc.returncode != 0:
            raise SandboxError(
                f"staging workspace files failed: {proc.stderr.decode(errors='replace').strip()}"
            )


def gvisor_available() -> bool:
    """Probe whether `podman --runtime=runsc` can actually launch a container.

    Used by test skip guards and the CLI to fail fast with a useful message
    instead of an opaque runtime error. Cheap (~250 ms cold, <50 ms warm) and
    safe to call from test setup.
    """

    if shutil.which("podman") is None:
        return False
    if shutil.which("runsc") is None and not Path("/usr/local/bin/runsc").exists():
        return False
    try:
        proc = subprocess.run(
            [
                "podman",
                "run",
                "--rm",
                "--runtime=runsc",
                "--security-opt=label=disable",
                "--runtime-flag=ignore-cgroups",
                "--network=none",
                "lab-agent-sandbox:0.1",
                "true",
            ],
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return proc.returncode == 0
