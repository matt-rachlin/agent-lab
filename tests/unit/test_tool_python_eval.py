"""Unit tests for `lab.agent.tools.python_eval`."""

from __future__ import annotations

from pathlib import Path

import pytest
from lab.agent.tools import python_eval as python_eval_mod


def _patch_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(python_eval_mod, "WORKSPACE_ROOT", str(tmp_path))


def test_python_eval_basic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_root(monkeypatch, tmp_path)
    out = python_eval_mod.python_eval(code="print(2 + 2)")
    assert out["stdout"].strip() == "4"
    assert out["exit_code"] == 0
    assert out["timed_out"] is False


def test_python_eval_captures_stderr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_root(monkeypatch, tmp_path)
    out = python_eval_mod.python_eval(
        code="import sys; print('oops', file=sys.stderr); sys.exit(2)"
    )
    assert "oops" in out["stderr"]
    assert out["exit_code"] == 2


def test_python_eval_runs_in_workspace_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_root(monkeypatch, tmp_path)
    out = python_eval_mod.python_eval(code="import os; print(os.getcwd())")
    assert out["stdout"].strip() == str(tmp_path)


def test_python_eval_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_root(monkeypatch, tmp_path)
    out = python_eval_mod.python_eval(code="import time; time.sleep(5)", timeout_sec=1)
    assert out["timed_out"] is True
    assert out["exit_code"] == 124


def test_python_eval_empty_rejected() -> None:
    with pytest.raises(ValueError, match="code must not be empty"):
        python_eval_mod.python_eval(code="")


def test_python_eval_negative_timeout_rejected() -> None:
    with pytest.raises(ValueError, match="timeout_sec must be positive"):
        python_eval_mod.python_eval(code="x", timeout_sec=0)


def test_python_eval_huge_timeout_rejected() -> None:
    with pytest.raises(ValueError, match="<= 300"):
        python_eval_mod.python_eval(code="x", timeout_sec=99999)
