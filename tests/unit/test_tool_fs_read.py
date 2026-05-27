"""Unit tests for `lab.agent.tools.fs_read` — schema + path-escape.

These do not run inside the sandbox. We exercise the tool function directly
against `tmp_path` to keep tests fast and hermetic; the path-validation logic
is identical on host and in the sandbox.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from lab.agent.tools import fs_read as fs_read_mod
from lab.agent.tools._common import PathEscapeError, resolve_workspace_path


def _patch_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Pretend `/workspace` lives at `tmp_path`."""
    monkeypatch.setattr(fs_read_mod, "resolve_workspace_path", lambda p: tmp_path / p)


def test_fs_read_reads_existing_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_workspace(monkeypatch, tmp_path)
    (tmp_path / "hello.txt").write_text("hi there", encoding="utf-8")
    out = fs_read_mod.fs_read(path="hello.txt")
    assert out["content"] == "hi there"
    assert out["size"] == 8
    assert out["truncated"] is False


def test_fs_read_truncates_at_max_bytes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_workspace(monkeypatch, tmp_path)
    (tmp_path / "big.txt").write_bytes(b"x" * 1024)
    out = fs_read_mod.fs_read(path="big.txt", max_bytes=100)
    assert out["truncated"] is True
    assert len(out["content"]) == 100
    assert out["size"] == 1024


def test_fs_read_missing_file_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_workspace(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError):
        fs_read_mod.fs_read(path="nope.txt")


def test_fs_read_rejects_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_workspace(monkeypatch, tmp_path)
    (tmp_path / "subdir").mkdir()
    with pytest.raises(IsADirectoryError):
        fs_read_mod.fs_read(path="subdir")


def test_fs_read_rejects_non_utf8(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_workspace(monkeypatch, tmp_path)
    (tmp_path / "bin.dat").write_bytes(b"\xff\xfe\xfd")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        fs_read_mod.fs_read(path="bin.dat")


def test_fs_read_max_bytes_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_bytes"):
        fs_read_mod.fs_read(path="x", max_bytes=0)


def test_path_escape_dotdot_raises() -> None:
    with pytest.raises(PathEscapeError, match="escapes"):
        resolve_workspace_path("../etc/passwd")


def test_path_escape_absolute_outside_raises() -> None:
    with pytest.raises(PathEscapeError, match="outside"):
        resolve_workspace_path("/etc/passwd")


def test_path_escape_empty_raises() -> None:
    with pytest.raises(PathEscapeError, match="empty"):
        resolve_workspace_path("")


def test_path_resolve_normalises_redundant_segments() -> None:
    assert str(resolve_workspace_path("./a/./b")) == "/workspace/a/b"
    assert str(resolve_workspace_path("a/b/../c")) == "/workspace/a/c"
    assert str(resolve_workspace_path("/workspace/sub/file")) == "/workspace/sub/file"
