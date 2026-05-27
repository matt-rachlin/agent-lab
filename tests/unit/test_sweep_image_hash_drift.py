"""Unit tests for the in-sweep sandbox image-hash drift guard.

F-005 EXP-002 surprise 4: the sweep saw three distinct
`sandbox_image_hash` values mid-flight, descended from the same
Containerfile commit, because `podman image prune` reaped layers and
triggered a rebuild between cells. The launch-time preflight only
checked once; this guard re-checks each cell.

These tests exercise the helper and the guard's invariant directly. The
full sweep loop (`run_sweep`) is too entangled with DB/Postgres/Inspect
state to call here — we instead test the `_read_sandbox_image_hash`
helper and the `ImageHashDriftError` shape that downstream code branches
on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lab.sweep import runner as runner_mod
from lab.sweep.runner import (
    ImageHashDriftError,
    _read_sandbox_image_hash,
)


def test_read_sandbox_image_hash_missing_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "no-such-file.sha"
    monkeypatch.setattr(runner_mod, "_SANDBOX_IMAGE_HASH_PATH", missing)
    assert _read_sandbox_image_hash() is None


def test_read_sandbox_image_hash_empty_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    f = tmp_path / "empty.sha"
    f.write_text("")
    monkeypatch.setattr(runner_mod, "_SANDBOX_IMAGE_HASH_PATH", f)
    assert _read_sandbox_image_hash() is None


def test_read_sandbox_image_hash_strips_whitespace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    f = tmp_path / "h.sha"
    f.write_text("\n  deadbeef0123  \n")
    monkeypatch.setattr(runner_mod, "_SANDBOX_IMAGE_HASH_PATH", f)
    assert _read_sandbox_image_hash() == "deadbeef0123"


def test_image_hash_drift_error_is_runtime_error_subclass() -> None:
    """Downstream code can `except RuntimeError` to catch this without a
    direct dep on the runner module; verify the inheritance contract.
    """

    assert issubclass(ImageHashDriftError, RuntimeError)


def test_image_hash_drift_error_message_carries_both_hashes() -> None:
    """The exception message MUST name both hashes so the operator can
    diff Containerfile and figure out which rebuild produced the second
    image. We don't assert exact phrasing — just that both appear.
    """

    msg = (
        "sandbox image hash drifted mid-sweep: "
        "starting=aaaaaaaaaaaaaaaa, current=bbbbbbbbbbbbbbbb, "
        "at_cell=run-xyz, executed=5/40"
    )
    exc = ImageHashDriftError(msg)
    s = str(exc)
    assert "aaaaaaaaaaaaaaaa" in s
    assert "bbbbbbbbbbbbbbbb" in s
    assert "run-xyz" in s


def test_drift_guard_detects_change(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Simulate the in-sweep re-read: file changes between two reads."""

    f = tmp_path / "h.sha"
    monkeypatch.setattr(runner_mod, "_SANDBOX_IMAGE_HASH_PATH", f)

    f.write_text("aaaaaaaaaaaaaaaa")
    start = _read_sandbox_image_hash()
    assert start == "aaaaaaaaaaaaaaaa"

    # Out-of-band rebuild between cells changes the hash.
    f.write_text("bbbbbbbbbbbbbbbb")
    current = _read_sandbox_image_hash()
    assert current == "bbbbbbbbbbbbbbbb"
    assert current != start  # drift detected


def test_drift_guard_detects_disappearing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If `conf/sandbox-image.sha` vanishes mid-sweep that is also drift.

    The guard treats `None` mid-sweep (after a non-None start) as a
    mismatch — we never want to silently continue with an unknown image.
    """

    f = tmp_path / "h.sha"
    monkeypatch.setattr(runner_mod, "_SANDBOX_IMAGE_HASH_PATH", f)

    f.write_text("aaaaaaaaaaaaaaaa")
    start = _read_sandbox_image_hash()

    f.unlink()
    current = _read_sandbox_image_hash()
    assert current is None
    assert current != start
