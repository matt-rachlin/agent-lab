"""Tests for `_snapshot_predicate_files` and `_collect_predicate_paths`.

F-009 EXP-006 follow-up: the tightened `multi-db-self-check` predicate is
an `all_of` composite that wraps a `db_query` sub-predicate alongside a
`workspace_file_contains` sub-predicate. The solver must walk the
composite at sandbox-teardown time and snapshot every nested workspace
path; otherwise the scorer sees an empty snapshot at scoring time and
the workspace half false-negatives. These tests pin the walk behaviour.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from lab.inspect_bridge.solver import (
    _collect_predicate_paths,
    _snapshot_predicate_files,
)


class _TaskMeta:
    """Tiny stand-in for the LabTask shape — only attribute the helper reads."""

    def __init__(self, predicate: dict[str, Any] | None) -> None:
        self.success_predicate = predicate


def test_collect_paths_workspace_file_predicate() -> None:
    paths = _collect_predicate_paths(
        {"type": "workspace_file_contains", "path": "mean.txt", "substring": "6.0"}
    )
    assert paths == ["mean.txt"]


def test_collect_paths_db_query_has_no_paths() -> None:
    paths = _collect_predicate_paths({"type": "db_query", "query": "SELECT 1", "expects_rows": 1})
    assert paths == []


def test_collect_paths_all_of_walks_subs() -> None:
    """The fix for F-009: all_of must yield every nested workspace path."""

    paths = _collect_predicate_paths(
        {
            "type": "all_of",
            "predicates": [
                {"type": "db_query", "query": "SELECT 1", "expects_rows": 1},
                {
                    "type": "workspace_file_contains",
                    "path": "mean.txt",
                    "substring": "6.0",
                },
                {"type": "workspace_file_exists", "path": "marker.txt"},
            ],
        }
    )
    assert paths == ["mean.txt", "marker.txt"]


def test_collect_paths_unknown_type_returns_empty() -> None:
    paths = _collect_predicate_paths({"type": "retrieval_recall", "expected_chunks": []})
    assert paths == []


def test_snapshot_calls_read_for_each_all_of_path() -> None:
    """Sandbox.read_workspace_file is invoked once per nested workspace path."""

    sandbox = MagicMock()
    _FILES: dict[str, bytes] = {"mean.txt": b"6.0\n"}

    def _read(p: str) -> bytes | None:
        return _FILES.get(p)

    sandbox.read_workspace_file.side_effect = _read

    task_meta = _TaskMeta(
        {
            "type": "all_of",
            "predicates": [
                {"type": "db_query", "query": "SELECT 1", "expects_rows": 1},
                {
                    "type": "workspace_file_contains",
                    "path": "mean.txt",
                    "substring": "6.0",
                },
            ],
        }
    )

    snap = _snapshot_predicate_files(task_meta, sandbox)
    assert snap == {"mean.txt": b"6.0\n"}
    sandbox.read_workspace_file.assert_called_once_with("mean.txt")


def test_snapshot_empty_when_sandbox_none() -> None:
    task_meta = _TaskMeta({"type": "workspace_file_exists", "path": "x"})
    assert _snapshot_predicate_files(task_meta, None) == {}


def test_snapshot_empty_for_db_query_only() -> None:
    """Pre-existing behaviour preserved: db_query alone needs no snapshot."""

    sandbox = MagicMock()
    task_meta = _TaskMeta({"type": "db_query", "query": "SELECT 1"})
    snap = _snapshot_predicate_files(task_meta, sandbox)
    assert snap == {}
    sandbox.read_workspace_file.assert_not_called()
