"""Unit tests for `lab.agent.tools.shell_exec`.

We point the tool at `tmp_path` instead of `/workspace` (which doesn't exist
on the host) by monkeypatching `WORKSPACE_ROOT`. The semantics under test
(bash invocation, timeout handling, exit-code capture) don't depend on the
actual sandbox path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lab.agent.tools import shell_exec as shell_exec_mod


def _patch_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(shell_exec_mod, "WORKSPACE_ROOT", str(tmp_path))


def test_shell_exec_basic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_root(monkeypatch, tmp_path)
    out = shell_exec_mod.shell_exec(command="echo hi")
    assert out["stdout"].strip() == "hi"
    assert out["exit_code"] == 0
    assert out["timed_out"] is False


def test_shell_exec_captures_stderr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_root(monkeypatch, tmp_path)
    out = shell_exec_mod.shell_exec(command="echo OUT; echo ERR 1>&2; exit 3")
    assert "OUT" in out["stdout"]
    assert "ERR" in out["stderr"]
    assert out["exit_code"] == 3


def test_shell_exec_uses_workspace_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_root(monkeypatch, tmp_path)
    out = shell_exec_mod.shell_exec(command="pwd")
    assert out["stdout"].strip() == str(tmp_path)


def test_shell_exec_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_root(monkeypatch, tmp_path)
    out = shell_exec_mod.shell_exec(command="sleep 5", timeout_sec=1)
    assert out["timed_out"] is True
    assert out["exit_code"] == 124


def test_shell_exec_empty_rejected() -> None:
    with pytest.raises(ValueError, match="command must not be empty"):
        shell_exec_mod.shell_exec(command="")


def test_shell_exec_negative_timeout_rejected() -> None:
    with pytest.raises(ValueError, match="timeout_sec must be positive"):
        shell_exec_mod.shell_exec(command="x", timeout_sec=0)


def test_shell_exec_huge_timeout_rejected() -> None:
    with pytest.raises(ValueError, match="<= 300"):
        shell_exec_mod.shell_exec(command="x", timeout_sec=99999)
