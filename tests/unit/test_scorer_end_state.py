"""Unit tests for `lab.inspect_bridge.scorer.end_state`."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import NOANSWER, Target
from inspect_ai.solver import TaskState

from lab.inspect_bridge import scorer as scorer_mod
from lab.inspect_bridge.scorer import (
    _eval_all_of_predicate,
    _eval_single_predicate,
    end_state,
)
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


# ---------------------------------------------------------------------------
# `all_of` composite predicate — F-009 EXP-006 follow-up
#
# The `multi-db-self-check` task used a bare `db_query` predicate that always
# returned 1 row for a registered task; qwen3-30b-a3b-moe fired zero tool
# calls on 8/8 cells and still scored 1.0. The fix is the `all_of` composite:
# pair the db_query meta-check with a workspace_file_* sub-predicate so
# no-op trajectories fail end_state. These tests pin that behaviour.
# ---------------------------------------------------------------------------


def _patch_db_query(monkeypatch: pytest.MonkeyPatch, rows: list[Any]) -> None:
    """Fake out psycopg.connect so db_query sub-predicates can be tested."""

    class FakeCursor:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

        def execute(self, q: Any) -> None:
            self.q = q

        def fetchall(self) -> list[Any]:
            return rows

    class FakeConn:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

        def cursor(self) -> FakeCursor:
            return FakeCursor()

    monkeypatch.setattr(scorer_mod.psycopg, "connect", lambda dsn: FakeConn())


def test_all_of_passes_when_every_sub_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both sub-predicates pass → composite passes with value 1.0."""
    _patch_db_query(monkeypatch, [(1,)])
    state = _state(
        success_predicate={
            "type": "all_of",
            "predicates": [
                {"type": "db_query", "query": "SELECT 1", "expects_rows": 1},
                {
                    "type": "workspace_file_contains",
                    "path": "mean.txt",
                    "substring": "6.0",
                },
            ],
        },
        workspace_snapshot={"mean.txt": b"6.0\n"},
    )
    s = _run(end_state(), state)
    assert s.value == 1.0
    assert "PASS" in (s.explanation or "")


def test_all_of_fails_when_workspace_sub_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """db_query passes but the file half fails → composite fails."""
    _patch_db_query(monkeypatch, [(1,)])
    state = _state(
        success_predicate={
            "type": "all_of",
            "predicates": [
                {"type": "db_query", "query": "SELECT 1", "expects_rows": 1},
                {
                    "type": "workspace_file_contains",
                    "path": "mean.txt",
                    "substring": "6.0",
                },
            ],
        },
        # mean.txt has a wrong value (e.g. model misread the data).
        workspace_snapshot={"mean.txt": b"12.5\n"},
    )
    s = _run(end_state(), state)
    assert s.value == 0.0
    expl = s.explanation or ""
    assert "FAIL" in expl
    assert "workspace_file_contains" in expl
    # Per-sub report (forensic-audit follow-up): BOTH halves appear with
    # their verdicts — the explanation must say which subs passed, not
    # just name the first failure.
    assert "all_of FAIL 1/2 sub-predicates passed" in expl
    assert "[0 db_query] PASS" in expl
    assert "[1 workspace_file_contains] FAIL" in expl


def test_all_of_fails_on_no_tool_calls_trajectory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard for F-009 EXP-006: a model that fires zero tool
    calls produces an empty workspace_snapshot. Under the old bare-db_query
    predicate this scored 1.0 (db_query meta-check passes for any registered
    task). Under the tightened `all_of` composite the workspace_file_*
    sub-predicate now fails because mean.txt was never written, so the
    composite — and end_state — fails. This is the synthetic version of
    the qwen3-30b-a3b-moe behaviour from EXP-006.
    """
    _patch_db_query(monkeypatch, [(1,)])
    state = _state(
        success_predicate={
            "type": "all_of",
            "predicates": [
                {
                    "type": "db_query",
                    "query": (
                        "SELECT slug FROM tasks "
                        "WHERE suite = 'pbs-agent-v0.1' "
                        "AND slug = 'multi-db-self-check' "
                        "AND retired_at IS NULL"
                    ),
                    "expects_rows": 1,
                },
                {
                    "type": "workspace_file_contains",
                    "path": "mean.txt",
                    "substring": "6.0",
                },
            ],
        },
        # Crucial: the synthetic "narrate-instead-of-call" trajectory has
        # no tool calls and therefore no workspace artifacts.
        workspace_snapshot={},
    )
    s = _run(end_state(), state)
    assert s.value == 0.0, (
        "A zero-tool-call trajectory must fail end_state on the tightened "
        "multi-db-self-check predicate; otherwise F-009 EXP-006 regresses."
    )
    assert "FAIL" in (s.explanation or "")
    assert "not present" in (s.explanation or "")


def test_all_of_reports_all_subs_when_first_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """The db_query sub-predicate fails first, but the file half is STILL
    evaluated and reported (no short-circuit — the old behaviour stopped at
    the first failure and the explanation carried zero triage value)."""
    # Return zero rows so db_query fails.
    _patch_db_query(monkeypatch, [])
    state = _state(
        success_predicate={
            "type": "all_of",
            "predicates": [
                {"type": "db_query", "query": "SELECT 1", "expects_rows": 1},
                {
                    "type": "workspace_file_contains",
                    "path": "mean.txt",
                    "substring": "6.0",
                },
            ],
        },
        workspace_snapshot={"mean.txt": b"6.0\n"},
    )
    s = _run(end_state(), state)
    assert s.value == 0.0
    expl = s.explanation or ""
    assert "db_query" in expl
    # The second sub-predicate is evaluated despite the earlier failure.
    assert "all_of FAIL 1/2 sub-predicates passed" in expl
    assert "[0 db_query] FAIL" in expl
    assert "[1 workspace_file_contains] PASS" in expl


def test_all_of_explanation_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Many failing subs with long paths → explanation truncated with marker."""
    predicate: dict[str, Any] = {
        "type": "all_of",
        "predicates": [
            {"type": "workspace_file_exists", "path": f"deeply/nested/dir/{i:02d}/" + "x" * 60}
            for i in range(12)
        ],
    }
    value, explanation = _eval_all_of_predicate(predicate, {})
    assert value == 0.0
    assert len(explanation) <= 600
    assert explanation.endswith("…[truncated]")
    # The verdict header survives truncation.
    assert explanation.startswith("all_of FAIL 0/12 sub-predicates passed")


def test_all_of_rejects_empty_predicates_list() -> None:
    state = _state(success_predicate={"type": "all_of", "predicates": []})
    s = _run(end_state(), state)
    assert s.value == 0.0
    assert "non-empty" in (s.explanation or "")


def test_all_of_rejects_nested_composites() -> None:
    """`all_of` inside `all_of` is rejected (keeps the schema flat)."""
    state = _state(
        success_predicate={
            "type": "all_of",
            "predicates": [
                {
                    "type": "all_of",
                    "predicates": [
                        {"type": "workspace_file_exists", "path": "x"},
                    ],
                },
            ],
        },
        workspace_snapshot={"x": b""},
    )
    s = _run(end_state(), state)
    assert s.value == 0.0
    assert "nested all_of" in (s.explanation or "")


# ---------------------------------------------------------------------------
# `all_of` dispatch through `_eval_single_predicate`.
#
# The `end_state` scorer routes top-level `all_of` predicates explicitly
# (covered above); commit 600b7a8 additionally wired `all_of` into the
# `_eval_single_predicate` dispatch so any caller that goes through the
# generic single-predicate path handles composites too. These tests pin
# that dispatch directly with pure workspace sub-predicates (no DB).
# ---------------------------------------------------------------------------


def test_single_predicate_dispatch_routes_all_of_pass() -> None:
    predicate: dict[str, Any] = {
        "type": "all_of",
        "predicates": [
            {"type": "workspace_file_exists", "path": "a.txt"},
            {"type": "workspace_file_contains", "path": "b.txt", "substring": "ok"},
        ],
    }
    snapshot: dict[str, Any] = {"a.txt": b"x", "b.txt": b"all ok here"}
    value, explanation = _eval_single_predicate(predicate, snapshot)
    assert value == 1.0
    assert "all_of PASS" in explanation
    assert "2 sub-predicates" in explanation
    # The dispatch must agree with calling the composite evaluator directly.
    assert (value, explanation) == _eval_all_of_predicate(predicate, snapshot)


def test_single_predicate_dispatch_all_of_one_sub_fails() -> None:
    predicate: dict[str, Any] = {
        "type": "all_of",
        "predicates": [
            {"type": "workspace_file_exists", "path": "a.txt"},
            {"type": "workspace_file_contains", "path": "b.txt", "substring": "ok"},
        ],
    }
    # b.txt missing → second sub-predicate fails → composite fails.
    value, explanation = _eval_single_predicate(predicate, {"a.txt": b"x"})
    assert value == 0.0
    assert "all_of FAIL" in explanation
    # The failing sub-predicate is named with its index and type.
    assert "[1 workspace_file_contains]" in explanation


def test_single_predicate_dispatch_all_of_empty_list_fails() -> None:
    value, explanation = _eval_single_predicate({"type": "all_of", "predicates": []}, {})
    assert value == 0.0
    assert "non-empty 'predicates' list" in explanation


def test_single_predicate_dispatch_all_of_unknown_sub_type_fails_composite() -> None:
    """A NOANSWER-ish unknown sub-type counts as a *fail* at the composite
    level (the task author opted into multi-check; a typo'd sub-predicate
    must not read as 'not applicable')."""
    predicate: dict[str, Any] = {
        "type": "all_of",
        "predicates": [{"type": "magic_8_ball"}],
    }
    value, explanation = _eval_single_predicate(predicate, {})
    assert value == 0.0
    assert "all_of FAIL" in explanation
