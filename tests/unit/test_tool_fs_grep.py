"""Unit tests for `lab.agent.tools.fs_grep`.

These DO shell out to `rg` (which has to be available on the host); the unit
boundary here is "does the wrapper produce the right argv and parse the right
output", not "does ripgrep work".
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lab.agent.tools import fs_grep as fs_grep_mod


@pytest.fixture(autouse=True)
def _require_rg() -> None:
    if shutil.which("rg") is None:
        pytest.skip("ripgrep not installed on host")


def _patch_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fs_grep_mod, "resolve_workspace_path", lambda p: tmp_path / p)


def test_fs_grep_finds_matches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_workspace(monkeypatch, tmp_path)
    (tmp_path / "a.txt").write_text("foo bar\nbaz\n")
    (tmp_path / "b.txt").write_text("baz only\n")
    out = fs_grep_mod.fs_grep(pattern="foo", path=".")
    assert len(out["matches"]) == 1
    assert out["matches"][0]["line_number"] == 1
    assert "foo" in out["matches"][0]["text"]


def test_fs_grep_glob_filter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_workspace(monkeypatch, tmp_path)
    (tmp_path / "a.py").write_text("hit me\n")
    (tmp_path / "b.txt").write_text("hit me\n")
    out = fs_grep_mod.fs_grep(pattern="hit", path=".", glob="*.py")
    assert len(out["matches"]) == 1
    assert out["matches"][0]["path"].endswith("a.py")


def test_fs_grep_no_matches_returns_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_workspace(monkeypatch, tmp_path)
    (tmp_path / "a.txt").write_text("nothing\n")
    out = fs_grep_mod.fs_grep(pattern="missing", path=".")
    assert out["matches"] == []
    assert out["truncated"] is False


def test_fs_grep_empty_pattern_rejected() -> None:
    with pytest.raises(ValueError, match="pattern must not be empty"):
        fs_grep_mod.fs_grep(pattern="")


def test_fs_grep_invalid_max_results_rejected() -> None:
    with pytest.raises(ValueError, match="max_results must be positive"):
        fs_grep_mod.fs_grep(pattern="x", max_results=0)


def test_fs_grep_path_escape_rejected() -> None:
    with pytest.raises(ValueError, match="escapes"):
        fs_grep_mod.fs_grep(pattern="x", path="../escape")
