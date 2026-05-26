"""Per-cell pool of long-lived MCP stdio servers.

Phase 6c shipped one MCP-server-spawn-per-call. That works for one or two
tool calls but adds ~150-250ms of init+teardown per call under gVisor — a
multi-turn agent makes that intolerable. `ToolPool` keeps one stdio server
process per `(sandbox, module)` pair, initialised once and reused for every
call within the lifetime of one cell.

Lifecycle:

    with ToolPool(sandbox) as pool:
        result = pool.invoke("lab.agent.tools.fs_read", "fs_read", {"path": "x"})

`stop()` is idempotent and runs from `__exit__`. We don't try to share pools
across cells in v0.1 — the sandbox itself is per-cell and the cleanup story
stays simple.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import threading
from types import TracebackType
from typing import Any, Self

from lab.agent.sandbox import Sandbox


class ToolPoolError(RuntimeError):
    """Raised when the pool itself fails (not when a tool call returns an error)."""


class _PooledServer:
    """One long-lived MCP stdio server process inside the sandbox.

    Owns the `podman exec` subprocess, the JSON-RPC framing state, and a
    lock so callers can't interleave concurrent calls on the same stdio
    pipe. We speak the JSON-RPC wire format directly (one JSON message per
    line) — same approach as `_drive_mcp_session_sync` in
    `inspect_bridge/tools.py`, but split out so the init handshake happens
    once.
    """

    __slots__ = ("_initialised", "_lock", "_next_id", "_proc", "module", "sandbox")

    def __init__(self, sandbox: Sandbox, module: str) -> None:
        self.sandbox = sandbox
        self.module = module
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._next_id = 1
        self._initialised = False

    def start(self) -> None:
        """Spawn the MCP server inside the sandbox + run the init handshake.

        Idempotent: second call is a no-op. Raises `ToolPoolError` if the
        server can't be reached.
        """

        if self._initialised:
            return
        argv = [
            "podman",
            "exec",
            "--interactive",
            self.sandbox.container_name,
            "python3",
            "-m",
            self.module,
        ]
        try:
            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise ToolPoolError(f"failed to spawn {self.module}: {exc}") from exc
        try:
            self._do_initialize()
        except Exception:
            self._terminate_proc()
            raise
        self._initialised = True

    def _do_initialize(self) -> None:
        # 1. initialize → expect response
        self._send(
            {
                "jsonrpc": "2.0",
                "id": self._claim_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "lab-tool-pool", "version": "0.1"},
                },
            }
        )
        self._recv()
        # 2. initialised notification (no response)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    def invoke(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call `tool_name` with `arguments`; return parsed result.

        Returns the structured content (if present) or the joined text
        content parsed as JSON when possible, raw string otherwise — same
        contract as `_drive_mcp_session_sync` for back-compat with the
        per-call path.
        """

        if not self._initialised:
            self.start()
        with self._lock:
            return self._call_locked(tool_name, arguments)

    def _call_locked(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        req_id = self._claim_id()
        self._send(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }
        )
        response = self._recv()
        if "error" in response:
            raise RuntimeError(str(response["error"]))
        result = response.get("result", {})
        if result.get("isError"):
            text_chunks = [
                c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"
            ]
            raise RuntimeError("; ".join(text_chunks) or "tool execution failed")
        if "structuredContent" in result:
            return result["structuredContent"]
        text_chunks = [
            c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"
        ]
        joined = "\n".join(text_chunks)
        try:
            return json.loads(joined)
        except json.JSONDecodeError:
            return joined

    def is_alive(self) -> bool:
        """Cheap liveness probe — `False` if the subprocess has exited."""

        if self._proc is None:
            return False
        return self._proc.poll() is None

    def stop(self) -> None:
        """Tear down the subprocess. Idempotent; safe in `__exit__`."""

        self._terminate_proc()
        self._initialised = False

    def _terminate_proc(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        with contextlib.suppress(Exception):
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=2)

    def _claim_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _send(self, payload: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise ToolPoolError(f"{self.module}: no stdin to write to")
        data = (json.dumps(payload) + "\n").encode("utf-8")
        try:
            proc.stdin.write(data)
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ToolPoolError(f"{self.module}: write failed: {exc}") from exc

    def _recv(self) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise ToolPoolError(f"{self.module}: no stdout to read from")
        line = proc.stdout.readline()
        if not line:
            stderr = b""
            if proc.stderr is not None:
                with contextlib.suppress(Exception):
                    stderr = proc.stderr.read() or b""
            raise ToolPoolError(
                f"{self.module}: server closed stdout unexpectedly: "
                f"{stderr.decode(errors='replace')}"
            )
        return json.loads(line.decode("utf-8"))  # type: ignore[no-any-return]


class ToolPool:
    """Per-cell pool of MCP servers, keyed by (sandbox, module).

    Thread-safe at the call level (each pooled server has its own lock);
    pool-level lookup is also guarded so concurrent `invoke()` from two
    threads in the same cell behaves correctly. Restarts a server on the
    next `invoke` if `is_alive()` returns False — covers crashes/oom kills.
    """

    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox
        self._servers: dict[str, _PooledServer] = {}
        self._mutex = threading.Lock()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def invoke(self, module: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call `tool_name` on the pooled `module`, spawning if needed."""

        server = self._get_or_spawn(module)
        if not server.is_alive():
            # The previous server crashed (stderr message landed in the log
            # at start of `_recv`). Drop it, spawn fresh, retry the call
            # once. If the fresh server also crashes, we surface that error
            # up the stack — don't loop forever.
            self._discard(module)
            server = self._get_or_spawn(module)
        return server.invoke(tool_name, arguments)

    def stop(self) -> None:
        """Tear down every pooled server. Idempotent."""

        with self._mutex:
            servers = list(self._servers.values())
            self._servers.clear()
        for server in servers:
            server.stop()

    # ------------------------------------------------------------------
    # context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # introspection helpers (used by tests)
    # ------------------------------------------------------------------

    def active_modules(self) -> list[str]:
        with self._mutex:
            return sorted(self._servers)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _get_or_spawn(self, module: str) -> _PooledServer:
        with self._mutex:
            server = self._servers.get(module)
            if server is None:
                server = _PooledServer(self.sandbox, module)
                self._servers[module] = server
        # Start outside the mutex so a slow init handshake on one module
        # doesn't block invocations on another module.
        server.start()
        return server

    def _discard(self, module: str) -> None:
        with self._mutex:
            server = self._servers.pop(module, None)
        if server is not None:
            server.stop()


__all__ = ["ToolPool", "ToolPoolError"]
