"""Unit tests for `lab.inspect_bridge.scorers.rag.mrr`."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import NOANSWER, Target
from inspect_ai.solver import TaskState
from lab.inspect_bridge.scorers.rag import mrr
from lab.tasks.registry import Task


def _state(
    *,
    predicate: dict[str, Any] | None,
    calls: list[list[dict[str, Any]]],
) -> TaskState:
    task = Task.model_validate(
        {
            "suite": "test",
            "slug": "mrr",
            "input": "hi",
            "success_predicate": predicate,
            "max_turns": 3,
            "tool_budget": 5,
        }
    )
    turns: list[dict[str, Any]] = []
    for i, hits in enumerate(calls):
        turns.append(
            {
                "turn": i,
                "tool_calls": [
                    {
                        "tool": "kb_query",
                        "args": {"kb_name": "bash", "question": "q"},
                        "result": {"hits": hits, "kb_status": "ok"},
                    }
                ],
            }
        )
    lab_agent = {
        "actual_turns": len(turns),
        "tool_call_count": len(turns),
        "terminated_reason": "model_finished",
        "turns": turns,
    }
    return TaskState(
        model="x",
        sample_id="s",
        epoch=0,
        input="hi",
        messages=[ChatMessageUser(content="hi")],
        metadata={"lab_task": task, "lab_agent": lab_agent},
    )


def _hit(chunk_id: str) -> dict[str, Any]:
    return {"chunk_id": chunk_id, "text": "x", "score": 0.5}


def _run(s: Any, state: TaskState) -> Any:
    return asyncio.run(s(state, Target("")))


def test_mrr_rank_1() -> None:
    pred = {"type": "retrieval_recall", "expected_chunks": ["c1"]}
    state = _state(predicate=pred, calls=[[_hit("c1"), _hit("c2")]])
    out = _run(mrr(), state)
    assert out.value == pytest.approx(1.0)


def test_mrr_rank_5() -> None:
    pred = {"type": "retrieval_recall", "expected_chunks": ["c5"]}
    state = _state(
        predicate=pred,
        calls=[[_hit("a"), _hit("b"), _hit("c"), _hit("d"), _hit("c5")]],
    )
    out = _run(mrr(), state)
    assert out.value == pytest.approx(0.2)


def test_mrr_miss_is_zero() -> None:
    pred = {"type": "retrieval_recall", "expected_chunks": ["c1"]}
    state = _state(predicate=pred, calls=[[_hit("xx"), _hit("yy")]])
    out = _run(mrr(), state)
    assert out.value == 0.0


def test_mrr_multiple_expected_averaged() -> None:
    pred = {"type": "retrieval_recall", "expected_chunks": ["c1", "c2"]}
    # c1 at rank 1 → RR=1.0; c2 at rank 2 → RR=0.5. Mean = 0.75.
    state = _state(predicate=pred, calls=[[_hit("c1"), _hit("c2")]])
    out = _run(mrr(), state)
    assert out.value == pytest.approx(0.75)


def test_mrr_picks_best_rank_across_calls() -> None:
    """If c1 shows up at rank 3 in call 1 and rank 1 in call 2, RR=1.0."""

    pred = {"type": "retrieval_recall", "expected_chunks": ["c1"]}
    state = _state(
        predicate=pred,
        calls=[
            [_hit("a"), _hit("b"), _hit("c1")],
            [_hit("c1"), _hit("a")],
        ],
    )
    out = _run(mrr(), state)
    assert out.value == pytest.approx(1.0)


def test_mrr_noanswer_on_non_retrieval_task() -> None:
    state = _state(
        predicate={"type": "workspace_file_exists", "path": "out"},
        calls=[[_hit("c1")]],
    )
    out = _run(mrr(), state)
    assert out.value == NOANSWER


def test_mrr_zero_when_no_kb_calls() -> None:
    pred = {"type": "retrieval_recall", "expected_chunks": ["c1"]}
    state = _state(predicate=pred, calls=[])
    out = _run(mrr(), state)
    assert out.value == 0.0
