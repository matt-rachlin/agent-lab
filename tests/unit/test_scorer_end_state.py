"""Unit tests for `lab.inspect_bridge.scorer.end_state`."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import NOANSWER, Target
from inspect_ai.solver import TaskState
from lab.inspect_bridge import scorer as scorer_mod
from lab.inspect_bridge.scorer import end_state
from lab.tasks.registry import Task


def _state(
    *,
    success_predicate: dict[str, Any] | None = None,
    workspace_snapshot: dict[str, Any] | None = None,
) -> TaskState:
    task = Task.model_validate(
        {
            "suite": "test",
            "slug": "end-state-test",
            "input": "hi",
            "max_turns": 2,
            "tool_budget": 1,
            "success_predicate": success_predicate,
        }
    )
    lab_agent: dict[str, Any] = {
        "actual_turns": 1,
        "tool_call_count": 0,
        "terminated_reason": "model_finished",
        "workspace_snapshot": workspace_snapshot or {},
        "turns": [],
    }
    return TaskState(
        model="x",
        sample_id="s",
        epoch=0,
        input="hi",
        messages=[ChatMessageUser(content="hi")],
        metadata={"lab_task": task, "lab_agent": lab_agent},
    )


def _run(scorer: Any, state: TaskState) -> Any:
    return asyncio.run(scorer(state, Target("")))


def test_workspace_file_exists_pass() -> None:
    state = _state(
        success_predicate={"type": "workspace_file_exists", "path": "out.txt"},
        workspace_snapshot={"out.txt": b"any-content"},
    )
    s = _run(end_state(), state)
    assert s.value == 1.0


def test_workspace_file_exists_missing() -> None:
    state = _state(
        success_predicate={"type": "workspace_file_exists", "path": "out.txt"},
        workspace_snapshot={},
    )
    s = _run(end_state(), state)
    assert s.value == 0.0
    assert "not present" in (s.explanation or "")


def test_workspace_file_equals_pass() -> None:
    state = _state(
        success_predicate={
            "type": "workspace_file_equals",
            "path": "answer.txt",
            "expected": "the result is 42",
        },
        workspace_snapshot={"answer.txt": b"the result is 42\n"},
    )
    s = _run(end_state(), state)
    assert s.value == 1.0


def test_workspace_file_equals_fail() -> None:
    state = _state(
        success_predicate={
            "type": "workspace_file_equals",
            "path": "answer.txt",
            "expected": "the result is 42",
        },
        workspace_snapshot={"answer.txt": b"the result is 7"},
    )
    s = _run(end_state(), state)
    assert s.value == 0.0


def test_workspace_file_equals_case_insensitive() -> None:
    state = _state(
        success_predicate={
            "type": "workspace_file_equals",
            "path": "answer.txt",
            "expected": "PASS",
            "case_sensitive": False,
        },
        workspace_snapshot={"answer.txt": b"pass"},
    )
    s = _run(end_state(), state)
    assert s.value == 1.0


def test_workspace_file_contains_pass() -> None:
    state = _state(
        success_predicate={
            "type": "workspace_file_contains",
            "path": "out.txt",
            "substring": "PASS",
        },
        workspace_snapshot={"out.txt": b"---PASS---"},
    )
    s = _run(end_state(), state)
    assert s.value == 1.0


def test_workspace_file_contains_fail() -> None:
    state = _state(
        success_predicate={
            "type": "workspace_file_contains",
            "path": "out.txt",
            "substring": "OK",
        },
        workspace_snapshot={"out.txt": b"different content"},
    )
    s = _run(end_state(), state)
    assert s.value == 0.0


def test_no_predicate_returns_noanswer() -> None:
    state = _state()
    s = _run(end_state(), state)
    assert s.value == NOANSWER


def test_predicate_override_argument_wins() -> None:
    state = _state(workspace_snapshot={"x.txt": b"hello"})
    # Pass the predicate as an explicit arg — the task has no predicate.
    s = _run(
        end_state({"type": "workspace_file_contains", "path": "x.txt", "substring": "hello"}),
        state,
    )
    assert s.value == 1.0


def test_db_query_predicate_rejects_writes() -> None:
    state = _state(
        success_predicate={
            "type": "db_query",
            "query": "DROP TABLE tasks",
        }
    )
    s = _run(end_state(), state)
    assert s.value == 0.0
    assert "write-like" in (s.explanation or "")


def test_db_query_predicate_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scorer should call psycopg and pass when row count matches."""

    class FakeCursor:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

        def execute(self, q: str) -> None:
            self.q = q

        def fetchall(self) -> list[Any]:
            return [(1,)]

    class FakeConn:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

        def cursor(self) -> FakeCursor:
            return FakeCursor()

    monkeypatch.setattr(scorer_mod.psycopg, "connect", lambda dsn: FakeConn())
    state = _state(
        success_predicate={
            "type": "db_query",
            "query": "SELECT 1",
            "expects_rows": 1,
        }
    )
    s = _run(end_state(), state)
    assert s.value == 1.0


def test_unknown_predicate_type_returns_noanswer() -> None:
    """Unknown predicate types (e.g. `retrieval_recall` on a non-RAG scorer)
    should NOANSWER, not score 0.0 — see F-005 EXP-002 follow-up.
    """
    state = _state(success_predicate={"type": "magic_8_ball"})
    s = _run(end_state(), state)
    assert s.value == NOANSWER
    assert "not applicable" in (s.explanation or "")


def test_retrieval_recall_predicate_returns_noanswer() -> None:
    """The concrete case from F-005: `end_state` saw `retrieval_recall`
    (a RAG-only predicate type) and was scoring 0.0. After the fix it
    should NOANSWER so the scorer's mean isn't polluted on RAG tasks.
    """
    state = _state(
        success_predicate={
            "type": "retrieval_recall",
            "expected_chunks": ["chunk-1", "chunk-2"],
        }
    )
    s = _run(end_state(), state)
    assert s.value == NOANSWER
