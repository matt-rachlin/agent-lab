"""Unit tests for `lab.agent.tool_pool`.

We don't actually spawn `podman exec` here — we plug a fake subprocess into
`_PooledServer` and exercise the pool's reuse / restart / teardown logic.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from lab.agent import tool_pool as tp_mod
from lab.agent.tool_pool import ToolPool, ToolPoolError, _PooledServer


class _FakeStdin:
    def __init__(self) -> None:
        self.buf: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> int:
        if self.closed:
            raise BrokenPipeError("stdin closed")
        self.buf.append(data)
        return len(data)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _FakeStdout:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        # Each response is one JSON line; readline pops one per call.
        self.responses: list[dict[str, Any]] = list(responses)

    def readline(self) -> bytes:
        if not self.responses:
            return b""
        return (json.dumps(self.responses.pop(0)) + "\n").encode("utf-8")


class _FakeProc:
    """Minimal stand-in for `subprocess.Popen`."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(responses)
        self.stderr = _FakeStdout([])
        self._alive = True
        self.killed = False
        self.waited = 0

    def poll(self) -> int | None:
        return None if self._alive else 0

    def wait(self, timeout: float | None = None) -> int:
        self.waited += 1
        self._alive = False
        return 0

    def kill(self) -> None:
        self.killed = True
        self._alive = False


@pytest.fixture
def fake_sandbox() -> Any:
    class _Sandbox:
        container_name = "test-sandbox"

    return _Sandbox()


def _seed_responses() -> list[dict[str, Any]]:
    # initialize response + a tool/call response with structured content.
    return [
        {"jsonrpc": "2.0", "id": 2, "result": {"protocolVersion": "2025-06-18"}},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "result": {"structuredContent": {"ok": True}},
        },
    ]


def test_pool_reuses_existing_server(monkeypatch: pytest.MonkeyPatch, fake_sandbox: Any) -> None:
    # Make every call return a fresh process so we can detect reuse-vs-new.
    procs: list[_FakeProc] = []

    def fake_popen(*args: Any, **kwargs: Any) -> _FakeProc:
        # Each fresh proc gets the init response plus N tool/call responses;
        # enough for the test to not run out.
        proc = _FakeProc(
            [
                {"jsonrpc": "2.0", "id": 2, "result": {"protocolVersion": "x"}},
                *[
                    {"jsonrpc": "2.0", "id": i + 3, "result": {"structuredContent": {"i": i}}}
                    for i in range(5)
                ],
            ]
        )
        procs.append(proc)
        return proc

    monkeypatch.setattr(tp_mod.subprocess, "Popen", fake_popen)

    pool = ToolPool(fake_sandbox)
    try:
        r1 = pool.invoke("lab.agent.tools.fs_read", "fs_read", {"path": "x"})
        r2 = pool.invoke("lab.agent.tools.fs_read", "fs_read", {"path": "y"})
        r3 = pool.invoke("lab.agent.tools.fs_read", "fs_read", {"path": "z"})
    finally:
        pool.stop()
    assert r1 == {"i": 0}
    assert r2 == {"i": 1}
    assert r3 == {"i": 2}
    # Exactly one subprocess spawned for three calls → pool reused it.
    assert len(procs) == 1


def test_pool_restarts_after_crash(monkeypatch: pytest.MonkeyPatch, fake_sandbox: Any) -> None:
    procs: list[_FakeProc] = []

    def fake_popen(*args: Any, **kwargs: Any) -> _FakeProc:
        proc = _FakeProc(
            [
                {"jsonrpc": "2.0", "id": 2, "result": {"protocolVersion": "x"}},
                {"jsonrpc": "2.0", "id": 3, "result": {"structuredContent": {"ok": True}}},
            ]
        )
        procs.append(proc)
        return proc

    monkeypatch.setattr(tp_mod.subprocess, "Popen", fake_popen)

    pool = ToolPool(fake_sandbox)
    try:
        pool.invoke("lab.agent.tools.fs_read", "fs_read", {"path": "x"})
        # Simulate a crash: mark the proc dead.
        procs[0]._alive = False
        pool.invoke("lab.agent.tools.fs_read", "fs_read", {"path": "y"})
    finally:
        pool.stop()
    assert len(procs) == 2  # restarted after detecting the crash


def test_pool_tears_down_cleanly(monkeypatch: pytest.MonkeyPatch, fake_sandbox: Any) -> None:
    procs: list[_FakeProc] = []

    def fake_popen(*args: Any, **kwargs: Any) -> _FakeProc:
        proc = _FakeProc(
            [
                {"jsonrpc": "2.0", "id": 2, "result": {"protocolVersion": "x"}},
                {"jsonrpc": "2.0", "id": 3, "result": {"structuredContent": {}}},
            ]
        )
        procs.append(proc)
        return proc

    monkeypatch.setattr(tp_mod.subprocess, "Popen", fake_popen)

    with ToolPool(fake_sandbox) as pool:
        pool.invoke("lab.agent.tools.fs_read", "fs_read", {})
        assert pool.active_modules() == ["lab.agent.tools.fs_read"]
    # After __exit__: every spawned proc had its stdin closed + waited.
    assert procs[0].stdin.closed
    assert procs[0].waited >= 1
    # stop() is idempotent.
    pool.stop()


def test_pool_independent_modules_get_independent_procs(
    monkeypatch: pytest.MonkeyPatch, fake_sandbox: Any
) -> None:
    procs: list[_FakeProc] = []

    def fake_popen(*args: Any, **kwargs: Any) -> _FakeProc:
        proc = _FakeProc(
            [
                {"jsonrpc": "2.0", "id": 2, "result": {"protocolVersion": "x"}},
                {"jsonrpc": "2.0", "id": 3, "result": {"structuredContent": {"ok": True}}},
            ]
        )
        procs.append(proc)
        return proc

    monkeypatch.setattr(tp_mod.subprocess, "Popen", fake_popen)

    pool = ToolPool(fake_sandbox)
    try:
        pool.invoke("lab.agent.tools.fs_read", "fs_read", {})
        pool.invoke("lab.agent.tools.fs_write", "fs_write", {})
    finally:
        pool.stop()
    assert len(procs) == 2


def test_pool_propagates_tool_error_without_killing_server(
    monkeypatch: pytest.MonkeyPatch, fake_sandbox: Any
) -> None:
    proc = _FakeProc(
        [
            {"jsonrpc": "2.0", "id": 2, "result": {"protocolVersion": "x"}},
            {
                "jsonrpc": "2.0",
                "id": 3,
                "result": {"isError": True, "content": [{"type": "text", "text": "nope"}]},
            },
            {
                "jsonrpc": "2.0",
                "id": 4,
                "result": {"structuredContent": {"ok": True}},
            },
        ]
    )

    monkeypatch.setattr(tp_mod.subprocess, "Popen", lambda *a, **kw: proc)

    pool = ToolPool(fake_sandbox)
    try:
        with pytest.raises(RuntimeError, match="nope"):
            pool.invoke("lab.agent.tools.fs_read", "fs_read", {})
        # Server still up: next call works.
        r = pool.invoke("lab.agent.tools.fs_read", "fs_read", {})
        assert r == {"ok": True}
    finally:
        pool.stop()


def test_pooled_server_init_failure_cleans_up(
    monkeypatch: pytest.MonkeyPatch, fake_sandbox: Any
) -> None:
    """When init handshake fails we should not leak the subprocess."""

    proc = _FakeProc([])  # immediate EOF → init read fails

    monkeypatch.setattr(tp_mod.subprocess, "Popen", lambda *a, **kw: proc)

    server = _PooledServer(fake_sandbox, "lab.agent.tools.fs_read")
    with pytest.raises(ToolPoolError):
        server.start()
    # We tore down the subprocess on the failure path.
    assert proc.stdin.closed or proc.waited >= 1
