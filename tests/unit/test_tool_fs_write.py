"""Unit tests for `lab.agent.tools.fs_write`."""

from __future__ import annotations

from pathlib import Path

import pytest

from lab.agent.tools import fs_write as fs_write_mod


def _patch_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fs_write_mod, "resolve_workspace_path", lambda p: tmp_path / p)


def test_fs_write_creates_new_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_workspace(monkeypatch, tmp_path)
    out = fs_write_mod.fs_write(path="new.txt", content="hi", mode="create")
    assert out["bytes_written"] == 2
    assert (tmp_path / "new.txt").read_text() == "hi"


def test_fs_write_create_rejects_existing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_workspace(monkeypatch, tmp_path)
    (tmp_path / "x.txt").write_text("old")
    with pytest.raises(FileExistsError):
        fs_write_mod.fs_write(path="x.txt", content="new", mode="create")


def test_fs_write_overwrite_replaces(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_workspace(monkeypatch, tmp_path)
    (tmp_path / "x.txt").write_text("old")
    fs_write_mod.fs_write(path="x.txt", content="new", mode="overwrite")
    assert (tmp_path / "x.txt").read_text() == "new"


def test_fs_write_append_preserves(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_workspace(monkeypatch, tmp_path)
    (tmp_path / "x.txt").write_text("old\n")
    fs_write_mod.fs_write(path="x.txt", content="more\n", mode="append")
    assert (tmp_path / "x.txt").read_text() == "old\nmore\n"


def test_fs_write_creates_parent_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_workspace(monkeypatch, tmp_path)
    fs_write_mod.fs_write(path="deep/nested/x.txt", content="ok")
    assert (tmp_path / "deep/nested/x.txt").read_text() == "ok"


def test_fs_write_invalid_mode_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_workspace(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="mode must be"):
        fs_write_mod.fs_write(path="x.txt", content="hi", mode="weird")  # type: ignore[arg-type]


def test_fs_write_path_escape_rejected(tmp_path: Path) -> None:
    # No monkeypatch — exercise the real resolver against the dot-dot path.
    with pytest.raises(ValueError, match="escapes"):
        fs_write_mod.fs_write(path="../escape.txt", content="x")
