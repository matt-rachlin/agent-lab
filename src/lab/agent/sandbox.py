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

**Network allow-list (v0.1, DNS-restricted)**

When `network=["a.example", "b.example"]` is passed:

  * The container joins the default rootless Podman bridge (NAT'd egress).
  * `/etc/hosts` is populated via `--add-host=NAME:IP` for each allow-listed
    name, where IP is resolved on the host at `start()` time.
  * `--dns=127.0.0.1` points DNS at a loopback port nothing is listening on,
    so any name NOT pre-resolved into `/etc/hosts` fails to resolve at all.

This is "DNS-restricted, not L3-firewalled". A model that already knows the
literal IP of a non-allow-listed host could still reach it. Acceptable for
v0.1 because:
  * Our agent reaches HTTP endpoints by hostname (`http_fetch`),
  * Full default-deny iptables injection into a rootless netns requires
    `slirp4netns` + custom port handlers we haven't yet built,
  * The MCP `http_fetch` tool re-validates the hostname against
    `LAB_HTTP_ALLOWLIST` before issuing the request, layered on top.

Tracked for hardening as part of EXP-002 follow-up if the model ever bypasses
the host-validation path.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import socket
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
    add_hosts: list[tuple[str, str]] = field(default_factory=list)
    dns_servers: list[str] = field(default_factory=list)
    # Read-only bind mount of the host KB root to a fixed path inside the
    # container (default `/kb`). When None, no KB mount is added — keeps the
    # default sandbox surface unchanged.
    kb_root_mount: Path | None = None
    kb_mount_target: str = "/kb"

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
        for host, ip in self.add_hosts:
            argv += [f"--add-host={host}:{ip}"]
        for dns in self.dns_servers:
            argv += [f"--dns={dns}"]
        if self.kb_root_mount is not None:
            # `:ro` makes the mount read-only; `Z` would relabel for SELinux
            # but conflicts with `label=disable` we already pass, so we leave
            # SELinux relabelling alone (gVisor doesn't care).
            argv += [f"-v={self.kb_root_mount}:{self.kb_mount_target}:ro"]
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
    add_hosts: list[tuple[str, str]] | None = None,
    dns_servers: list[str] | None = None,
    kb_root_mount: Path | None = None,
    kb_mount_target: str = "/kb",
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
        add_hosts=list(add_hosts or []),
        dns_servers=list(dns_servers or []),
        kb_root_mount=kb_root_mount,
        kb_mount_target=kb_mount_target,
    ).to_argv()


def _resolve_host_ipv4(name: str) -> str:
    """Look up the first IPv4 address for `name` on the host.

    Used at sandbox start time to bake allow-listed hostnames into the
    container's `/etc/hosts`. Raises `SandboxError` on lookup failure so
    `Sandbox.start()` fails fast instead of producing a broken sandbox.
    """

    try:
        infos = socket.getaddrinfo(name, None, family=socket.AF_INET)
    except socket.gaierror as exc:
        raise SandboxError(f"could not resolve allow-listed host {name!r}: {exc}") from exc
    for info in infos:
        sockaddr = info[4]
        if sockaddr and isinstance(sockaddr[0], str):
            return sockaddr[0]
    raise SandboxError(f"no IPv4 address returned for allow-listed host {name!r}")


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
        kb_root_mount: Path | None = None,
        kb_mount_target: str = "/kb",
    ) -> None:
        # Allow-list of (host, ip) pairs to bake into the container's
        # `/etc/hosts`; empty in non-allow-list modes.
        self._allowed_hosts: list[str] = []
        if isinstance(network, list):
            # DNS-restricted allow-list mode (see module docstring). We
            # resolve the listed hosts at `start()` time, not now, so the
            # constructor stays cheap and predictable for tests.
            if not network:
                # Empty list == nothing reachable == prefer the unambiguous
                # `none` mode over a no-op bridge.
                self._network_arg = "none"
            else:
                self._network_arg = "podman"  # default rootless bridge
                self._allowed_hosts = list(network)
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
        self.kb_root_mount: Path | None = (
            Path(kb_root_mount) if kb_root_mount is not None else None
        )
        self.kb_mount_target = kb_mount_target
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
        add_hosts: list[tuple[str, str]] = []
        dns_servers: list[str] = []
        if self._allowed_hosts:
            # Resolve each allow-listed host to a stable IPv4 at start time,
            # then pin DNS at a non-listening loopback port so unrelated names
            # cannot resolve.
            #
            # Special case: `host.containers.internal` is a podman-managed
            # alias for the host bridge gateway. We can't resolve it on the
            # host (it doesn't appear in /etc/hosts there) but podman injects
            # it into the container's /etc/hosts automatically — so we let
            # podman handle it AND skip the DNS-pin (the kb_query path needs
            # to resolve other names too, e.g. Ollama remote-model proxies).
            real_hosts = [
                h for h in self._allowed_hosts
                if h not in {"host.containers.internal", "host.docker.internal"}
            ]
            for host in real_hosts:
                add_hosts.append((host, _resolve_host_ipv4(host)))
            # Only pin DNS if ALL allow-listed hosts were resolvable on the
            # host. host.containers.internal alone (the kb_query case) keeps
            # default DNS so the container can reach the host via the alias.
            if real_hosts and len(real_hosts) == len(self._allowed_hosts):
                dns_servers = ["127.0.0.1"]
        argv = _build_run_argv(
            image=self.image,
            name=self.container_name,
            runtime=self.runtime,
            network=self._network_arg,
            mem_limit=self.mem_limit,
            cpu_limit=self.cpu_limit,
            env=self.env,
            add_hosts=add_hosts,
            dns_servers=dns_servers,
            kb_root_mount=self.kb_root_mount,
            kb_mount_target=self.kb_mount_target,
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
