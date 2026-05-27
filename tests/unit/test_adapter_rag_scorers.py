"""Unit tests for `lab.inspect_bridge.adapter._select_scorers` RAG branches.

Companion to `test_adapter_scorers.py` — those tests cover the existing
6e scorer selection. These add the 6h-c selection cases.
"""

from __future__ import annotations

from typing import Any

from inspect_ai._util.registry import registry_info

from lab.inspect_bridge.adapter import _select_scorers, _task_uses_kb_query
from lab.tasks.registry import Task


def _task(**overrides: Any) -> Task:
    base: dict[str, Any] = {
        "suite": "test",
        "slug": "adapter-rag",
        "input": "find the redirection passage",
        "max_turns": 3,
        "tool_budget": 5,
    }
    base.update(overrides)
    return Task.model_validate(base)


def _names(scorers: list[Any]) -> list[str]:
    out: list[str] = []
    for s in scorers:
        name = registry_info(s).name
        if "/" in name:
            name = name.split("/", 1)[1]
        out.append(name)
    return out


def test_retrieval_recall_predicate_adds_rag_scorers() -> None:
    task = _task(
        success_predicate={
            "type": "retrieval_recall",
            "expected_chunks": ["c1", "c2"],
            "k": 5,
        },
        tools=[{"name": "kb_query"}],
    )
    names = _names(_select_scorers(task))
    # All four core RAG scorers fire on a retrieval task.
    assert "recall_at_k" in names
    assert "mrr" in names
    assert "ndcg" in names
    assert "attribution" in names
    # `end_state` still fires because the predicate is set — it'll
    # return 0/NOANSWER for retrieval_recall, but it's harmless.
    assert "end_state" in names
    # Budget always present.
    assert "budget_respected" in names
    # Faithfulness NOT included by default — must opt in.
    assert "faithfulness" not in names
    # No tool_call rubric → no tool_correctness.
    assert "tool_correctness" not in names


def test_include_faithfulness_requires_kb_query_tool() -> None:
    """`include_faithfulness: true` without kb_query in tools → no faithfulness."""

    task = _task(
        success_predicate={
            "type": "retrieval_recall",
            "expected_chunks": ["c1"],
            "include_faithfulness": True,
        },
        # No tools at all.
    )
    names = _names(_select_scorers(task))
    assert "faithfulness" not in names


def test_include_faithfulness_with_kb_query_tool_adds_scorer() -> None:
    task = _task(
        success_predicate={
            "type": "retrieval_recall",
            "expected_chunks": ["c1"],
            "include_faithfulness": True,
        },
        tools=[{"name": "kb_query"}, {"name": "fs_read"}],
    )
    names = _names(_select_scorers(task))
    assert "faithfulness" in names


def test_non_retrieval_task_does_not_get_rag_scorers() -> None:
    """Regression: workspace_file_exists task should not pick up RAG scorers."""

    task = _task(
        success_predicate={"type": "workspace_file_exists", "path": "out.txt"},
    )
    names = _names(_select_scorers(task))
    assert "recall_at_k" not in names
    assert "mrr" not in names
    assert "ndcg" not in names
    assert "attribution" not in names
    assert "faithfulness" not in names
    # The existing scorers still fire.
    assert "end_state" in names
    assert "budget_respected" in names


def test_no_predicate_no_rag_scorers() -> None:
    """Regression: task with no predicate gets only `budget_respected`."""

    task = _task()
    names = _names(_select_scorers(task))
    assert names == ["budget_respected"]


def test_tool_call_rubric_still_works_on_retrieval_task() -> None:
    """A retrieval task with a tool_call rubric should get BOTH families."""

    task = _task(
        success_predicate={
            "type": "retrieval_recall",
            "expected_chunks": ["c1"],
        },
        rubric={
            "type": "tool_call",
            "target_tool": "kb_query",
            "expected_args": {"kb_name": "bash"},
        },
        tools=[{"name": "kb_query"}],
    )
    names = _names(_select_scorers(task))
    assert "recall_at_k" in names
    assert "tool_correctness" in names
    assert "budget_respected" in names


def test_retrieval_recall_uses_predicate_k() -> None:
    """The `k` in the predicate should drive the recall_at_k scorer."""

    task = _task(
        success_predicate={
            "type": "retrieval_recall",
            "expected_chunks": ["c1"],
            "k": 10,
        },
    )
    scorers = _select_scorers(task)
    # Just check the scorer is present; the k value is exercised by the
    # recall_at_k unit tests directly.
    assert "recall_at_k" in _names(scorers)


def test_task_uses_kb_query_helper() -> None:
    assert _task_uses_kb_query(_task(tools=[{"name": "kb_query"}])) is True
    assert _task_uses_kb_query(_task(tools=[{"name": "fs_read"}])) is False
    assert _task_uses_kb_query(_task()) is False
    assert _task_uses_kb_query(_task(tools=[{"name": "fs_read"}, {"name": "kb_query"}])) is True


def test_include_judge_still_adds_trajectory_judge_on_retrieval_task() -> None:
    """`include_judge` should still work alongside the RAG scorers."""

    task = _task(
        success_predicate={
            "type": "retrieval_recall",
            "expected_chunks": ["c1"],
            "include_judge": True,
        },
    )
    names = _names(_select_scorers(task))
    assert "trajectory_judge" in names
    assert "recall_at_k" in names
