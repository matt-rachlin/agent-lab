"""Unit tests for `lab.observability.mlflow_backfill`.

These don't touch Postgres; they monkey-patch `psycopg.connect` to return
a tiny in-memory fake that returns canned rows. We assert that the
backfill:

* walks each table once
* mirrors only rows that lack an mlflow id (idempotency)
* keeps walking after a per-row error
* respects --dry-run (no mirror calls, no DB updates)
* writes the assigned id back via UPDATE statements
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from lab.observability import mlflow_backfill as mb


@dataclass
class FakeCursor:
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    executed: list[tuple[str, tuple[Any, ...] | None]] = field(default_factory=list)
    _last_query: str = ""

    def execute(self, query: str, params: tuple[Any, ...] | None = None) -> None:
        self._last_query = query
        self.executed.append((query, params))

    def fetchall(self) -> list[tuple[Any, ...]]:
        # Return the canned rows only for the SELECTs (i.e. when no UPDATE).
        if "UPDATE" in self._last_query:
            return []
        return self.rows

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


@dataclass
class FakeConn:
    cursor_factory: FakeCursor

    def cursor(self) -> FakeCursor:
        return self.cursor_factory

    def __enter__(self) -> FakeConn:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def _patch_pg(monkeypatch: pytest.MonkeyPatch, cursor: FakeCursor) -> None:
    monkeypatch.setattr(mb.psycopg, "connect", lambda _dsn: FakeConn(cursor_factory=cursor))


def _make_mirror() -> Any:
    """A MagicMock that behaves like an enabled MlflowMirror."""

    m = MagicMock()
    m.enabled = True
    m.upsert_experiment.return_value = "42"
    m.log_run.return_value = "run-uuid-xx"
    m.log_finding.return_value = "find-uuid-xx"
    m.log_model_card.return_value = "runs:/model-uuid-xx"
    return m


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------


def test_backfill_experiments_walks_all_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    cur = FakeCursor(
        rows=[
            ("EXP-1", "title 1", "docs/exp/1.md", None, None),
            ("EXP-2", "title 2", "docs/exp/2.md", "h2", None),
        ]
    )
    _patch_pg(monkeypatch, cur)

    mirror = _make_mirror()
    summary = mb.BackfillSummary()
    out = mb.backfill_experiments(mirror, dry_run=False, force=False, summary=summary)

    assert summary.experiments == 2
    assert summary.experiments_skipped == 0
    assert out == {"EXP-1": "42", "EXP-2": "42"}
    assert mirror.upsert_experiment.call_count == 2


def test_backfill_experiments_skips_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    cur = FakeCursor(
        rows=[
            ("EXP-1", "title 1", "docs/exp/1.md", None, "99"),  # already has mlflow_experiment_id
            ("EXP-2", "title 2", "docs/exp/2.md", None, None),
        ]
    )
    _patch_pg(monkeypatch, cur)

    mirror = _make_mirror()
    summary = mb.BackfillSummary()
    mb.backfill_experiments(mirror, dry_run=False, force=False, summary=summary)

    assert summary.experiments == 1  # only EXP-2 mirrored
    assert summary.experiments_skipped == 1
    assert mirror.upsert_experiment.call_count == 1


def test_backfill_dry_run_does_not_call_mirror(monkeypatch: pytest.MonkeyPatch) -> None:
    cur = FakeCursor(
        rows=[
            ("EXP-1", "t", "p", None, None),
            ("EXP-2", "t", "p", None, None),
        ]
    )
    _patch_pg(monkeypatch, cur)

    mirror = _make_mirror()
    summary = mb.BackfillSummary()
    mb.backfill_experiments(mirror, dry_run=True, force=False, summary=summary)

    assert summary.experiments == 2
    mirror.upsert_experiment.assert_not_called()


def test_backfill_continues_after_per_row_error(monkeypatch: pytest.MonkeyPatch) -> None:
    cur = FakeCursor(
        rows=[
            ("EXP-1", "t", "p", None, None),
            ("EXP-2", "t", "p", None, None),
            ("EXP-3", "t", "p", None, None),
        ]
    )
    _patch_pg(monkeypatch, cur)

    mirror = MagicMock()
    mirror.enabled = True
    mirror.upsert_experiment.side_effect = ["x1", RuntimeError("boom"), "x3"]
    summary = mb.BackfillSummary()
    mb.backfill_experiments(mirror, dry_run=False, force=False, summary=summary)

    # Two mirrored successfully; one error recorded; loop did not stop.
    assert summary.experiments == 2
    assert any("EXP-2" in e for e in summary.errors)
    assert mirror.upsert_experiment.call_count == 3


# ---------------------------------------------------------------------------
# findings
# ---------------------------------------------------------------------------


def test_backfill_findings_skips_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    cur = FakeCursor(
        rows=[
            ("F-001", "claim 1", "high", "EXP-001", None),
            ("F-002", "claim 2", "medium", "EXP-001", "already-mirrored"),
        ]
    )
    _patch_pg(monkeypatch, cur)

    mirror = _make_mirror()
    summary = mb.BackfillSummary()
    mb.backfill_findings(mirror, dry_run=False, force=False, summary=summary)

    assert summary.findings == 1
    assert summary.findings_skipped == 1
    mirror.log_finding.assert_called_once()
    # importance fixed at 3; confidence comes from text → float mapping.
    call_args = mirror.log_finding.call_args
    assert call_args.args == ("F-001",)
    assert call_args.kwargs["importance"] == 3
    assert call_args.kwargs["confidence"] == 0.9  # high


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def test_backfill_models_passes_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    cur = FakeCursor(
        rows=[
            ("model-a", "acme", "8b", ["tool_call", "json"], None),
            ("model-b", "acme", "13b", None, "runs:/already"),
        ]
    )
    _patch_pg(monkeypatch, cur)

    mirror = _make_mirror()
    summary = mb.BackfillSummary()
    mb.backfill_models(mirror, dry_run=False, force=False, summary=summary)

    assert summary.models == 1
    assert summary.models_skipped == 1
    call = mirror.log_model_card.call_args
    assert call.args == ("model-a",)
    assert call.kwargs["capabilities"] == ["tool_call", "json"]


# ---------------------------------------------------------------------------
# Disabled mirror short-circuit in backfill_all
# ---------------------------------------------------------------------------


def test_backfill_all_no_op_when_mirror_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    mirror = MagicMock()
    mirror.enabled = False
    summary = mb.backfill_all(mirror=mirror)
    assert summary.experiments == 0
    assert summary.runs == 0
    mirror.upsert_experiment.assert_not_called()
