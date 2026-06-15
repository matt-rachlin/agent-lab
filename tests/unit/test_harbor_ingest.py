"""Unit tests for lab.eval.external.harbor_ingest.

All tests mock psycopg.connect; no live DB required.

Scenarios:
1. Happy path -- 2-task JSONL, both tasks found, both rows inserted.
2. Unknown task -- third row has a non-existent slug -> skipped_unknown_task=1.
3. Idempotency -- second ingest returns rows_written=0 (ON CONFLICT DO NOTHING).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import lab.eval.external.harbor_ingest as hi_mod
from lab.eval.external.harbor_ingest import ingest_harbor_run

# ---------------------------------------------------------------------------
# In-memory psycopg doubles
# ---------------------------------------------------------------------------


@dataclass
class FakeCursor:
    """Records execute() calls; answers canned queries via side_effects list."""

    side_effects: list[list[tuple[Any, ...]]] = field(default_factory=list)
    executed: list[tuple[str, Any]] = field(default_factory=list)
    _row_idx: int = 0
    rowcount: int = 1

    def execute(self, query: str, params: Any = None) -> None:
        self.executed.append((query, params))

    def fetchone(self) -> tuple[Any, ...] | None:
        if self._row_idx < len(self.side_effects):
            rows = self.side_effects[self._row_idx]
            self._row_idx += 1
            return rows[0] if rows else None
        return None

    def fetchall(self) -> list[tuple[Any, ...]]:
        if self._row_idx < len(self.side_effects):
            rows = self.side_effects[self._row_idx]
            self._row_idx += 1
            return rows
        return []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


@dataclass
class FakeConn:
    cursor_obj: FakeCursor

    def cursor(self) -> FakeCursor:
        return self.cursor_obj

    def __enter__(self) -> FakeConn:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


def _make_conn(side_effects: list[list[tuple[Any, ...]]]) -> FakeConn:
    return FakeConn(cursor_obj=FakeCursor(side_effects=side_effects))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TASK_A = "adaptive-rejection-sampler"
_TASK_B = "matrix-multiply"
_TASK_UNKNOWN = "no-such-task"
_RUN_ID = "EXP-013-run-0001"


def _write_jsonl(tmp_path: Path, lines: list[dict[str, Any]]) -> Path:
    p = tmp_path / "results.jsonl"
    p.write_text("\n".join(json.dumps(obj) for obj in lines), encoding="utf-8")
    return p


def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub get_settings().pg_dsn so we never hit a real Postgres."""

    class _FakeSettings:
        pg_dsn = "postgresql://fake/fake"

    monkeypatch.setattr(hi_mod, "get_settings", _FakeSettings)


# ---------------------------------------------------------------------------
# Test 1 -- happy path
# ---------------------------------------------------------------------------


def test_happy_path_two_tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch)

    fixture = _write_jsonl(
        tmp_path,
        [
            {"task": _TASK_A, "passed": True, "score": 1.0},
            {"task": _TASK_B, "passed": False, "score": 0.0},
        ],
    )

    # side_effects for fetchone calls in order:
    #   1. SELECT evaluator_id after INSERT DO NOTHING -> (42,)
    #   2. SELECT task_id for _TASK_A -> (101,)
    #   3. SELECT task_id for _TASK_B -> (102,)
    side_effects: list[list[tuple[Any, ...]]] = [
        [(42,)],  # evaluator_id
        [(101,)],  # task_id for task A
        [(102,)],  # task_id for task B
    ]
    conn = _make_conn(side_effects)
    conn.cursor_obj.rowcount = 1  # each INSERT inserts 1 row

    with patch.object(hi_mod.psycopg, "connect", return_value=conn):
        counts = ingest_harbor_run(fixture, run_id=_RUN_ID)

    assert counts["rows_written"] == 2
    assert counts["passed"] == 1
    assert counts["failed"] == 1
    assert counts["skipped_unknown_task"] == 0

    # Verify an INSERT into eval_results was executed for each task.
    insert_stmts = [q for q, _ in conn.cursor_obj.executed if "INSERT INTO eval_results" in q]
    assert len(insert_stmts) == 2


# ---------------------------------------------------------------------------
# Test 2 -- unknown task slug -> skipped_unknown_task=1
# ---------------------------------------------------------------------------


def test_unknown_task_slug_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch)

    fixture = _write_jsonl(
        tmp_path,
        [
            {"task": _TASK_A, "passed": True, "score": 1.0},
            {"task": _TASK_B, "passed": False, "score": 0.25},
            {"task": _TASK_UNKNOWN, "passed": False, "score": 0.0},
        ],
    )

    side_effects: list[list[tuple[Any, ...]]] = [
        [(42,)],  # evaluator_id
        [(101,)],  # task_id for A
        [(102,)],  # task_id for B
        [],  # no task_id for UNKNOWN -> triggers skip
    ]
    conn = _make_conn(side_effects)
    conn.cursor_obj.rowcount = 1

    with patch.object(hi_mod.psycopg, "connect", return_value=conn):
        counts = ingest_harbor_run(fixture, run_id=_RUN_ID)

    assert counts["skipped_unknown_task"] == 1
    assert counts["rows_written"] == 2
    assert counts["passed"] == 1
    assert counts["failed"] == 1


# ---------------------------------------------------------------------------
# Test 3 -- idempotency: second ingest rows_written=0
# ---------------------------------------------------------------------------


def test_idempotent_second_ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch)

    fixture = _write_jsonl(
        tmp_path,
        [
            {"task": _TASK_A, "passed": True, "score": 1.0},
        ],
    )

    side_effects: list[list[tuple[Any, ...]]] = [
        [(42,)],  # evaluator_id
        [(101,)],  # task_id for A
    ]
    conn = _make_conn(side_effects)
    # Simulate ON CONFLICT DO NOTHING -> rowcount=0.
    conn.cursor_obj.rowcount = 0

    with patch.object(hi_mod.psycopg, "connect", return_value=conn):
        counts = ingest_harbor_run(fixture, run_id=_RUN_ID)

    assert counts["rows_written"] == 0
    assert counts["passed"] == 0
    assert counts["failed"] == 0
    assert counts["skipped_unknown_task"] == 0
