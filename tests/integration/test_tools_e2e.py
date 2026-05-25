"""End-to-end tool tests — spin up the sandbox and exercise each MCP server.

Requires:
  * podman + runsc + `lab-agent-sandbox:0.1` (same prerequisites as
    `tests/integration/test_sandbox.py`).

Each test starts a fresh sandbox, runs the in-sandbox MCP server via
`podman exec`, drives one tool call through it, and asserts on the result.
"""

from __future__ import annotations

import pytest

from lab.agent.sandbox import Sandbox, gvisor_available
from lab.agent.tools import TOOL_SERVERS
from lab.inspect_bridge.tools import _invoke_tool_via_sandbox_sync

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _require_gvisor() -> None:
    if not gvisor_available():
        pytest.skip(
            "gVisor not available: install runsc and build lab-agent-sandbox:0.1 "
            "(see `just sandbox-build`)"
        )


def _call(sandbox: Sandbox, name: str, args: dict[str, object]) -> object:
    return _invoke_tool_via_sandbox_sync(sandbox, TOOL_SERVERS[name], name, args)


def test_fs_write_then_fs_read_round_trip() -> None:
    with Sandbox() as sb:
        out_write = _call(sb, "fs_write", {"path": "x.txt", "content": "hello\n"})
        assert out_write["bytes_written"] == 6  # type: ignore[index]
        out_read = _call(sb, "fs_read", {"path": "x.txt"})
        assert out_read["content"] == "hello\n"  # type: ignore[index]
        assert out_read["size"] == 6  # type: ignore[index]


def test_fs_grep_finds_matches_in_sandbox() -> None:
    files = {"a.txt": b"foo bar\nbaz\n", "b.txt": b"unrelated\n"}
    with Sandbox(workspace_files=files) as sb:
        out = _call(sb, "fs_grep", {"pattern": "foo", "path": "."})
        matches = out["matches"]  # type: ignore[index]
        assert len(matches) == 1
        assert "foo" in matches[0]["text"]


def test_shell_exec_runs_in_workspace() -> None:
    with Sandbox() as sb:
        out = _call(sb, "shell_exec", {"command": "pwd"})
        assert out["stdout"].strip() == "/workspace"  # type: ignore[index]
        assert out["exit_code"] == 0  # type: ignore[index]


def test_python_eval_basic_arithmetic() -> None:
    with Sandbox() as sb:
        out = _call(sb, "python_eval", {"code": "print(7 * 6)"})
        assert out["stdout"].strip() == "42"  # type: ignore[index]
        assert out["exit_code"] == 0  # type: ignore[index]


def test_fs_read_rejects_path_escape() -> None:
    with Sandbox() as sb, pytest.raises(Exception, match=r"escapes|outside"):
        _call(sb, "fs_read", {"path": "../etc/passwd"})


def test_fs_write_create_rejects_existing() -> None:
    with (
        Sandbox(workspace_files={"x.txt": b"old"}) as sb,
        pytest.raises(Exception, match="already exists"),
    ):
        _call(sb, "fs_write", {"path": "x.txt", "content": "new", "mode": "create"})
