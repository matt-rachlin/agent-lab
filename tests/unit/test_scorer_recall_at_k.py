"""Unit tests for `lab.inspect_bridge.scorers.rag.recall_at_k`."""

from __future__ import annotations

import asyncio
from typing import Any

from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import NOANSWER, Target
from inspect_ai.solver import TaskState
from lab.inspect_bridge.scorers.rag import recall_at_k
from lab.tasks.registry import Task


def _state(
    *,
    predicate: dict[str, Any] | None,
    calls: list[list[dict[str, Any]]],
) -> TaskState:
    """Build a TaskState with a trajectory of `kb_query` calls.

    ``calls`` is a list of "call hit lists" — each element is the list of
    hit dicts that one `kb_query` call returned.
    """

    task = Task.model_validate(
        {
            "suite": "test",
            "slug": "recall",
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
                        "args": {"kb_name": "bash", "question": "q", "k": len(hits)},
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


def _run(s: Any, state: TaskState) -> Any:
    return asyncio.run(s(state, Target("")))


def _hit(chunk_id: str, **extras: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "chunk_id": chunk_id,
        "source_url": f"https://example.com/{chunk_id}",
        "section_path": ["intro"],
        "text": "some passage",
        "score": 0.5,
    }
    base.update(extras)
    return base


def test_recall_perfect() -> None:
    pred = {"type": "retrieval_recall", "expected_chunks": ["c1", "c2"], "k": 5}
    state = _state(predicate=pred, calls=[[_hit("c1"), _hit("c2"), _hit("c3")]])
    out = _run(recall_at_k(), state)
    assert out.value == 1.0
    assert "2/2" in (out.explanation or "")


def test_recall_partial() -> None:
    pred = {"type": "retrieval_recall", "expected_chunks": ["c1", "c2", "c3", "c4"]}
    state = _state(predicate=pred, calls=[[_hit("c1"), _hit("c9"), _hit("c2")]])
    out = _run(recall_at_k(), state)
    assert out.value == 0.5
    assert out.metadata is not None
    assert out.metadata["matched_count"] == 2
    assert out.metadata["expected_count"] == 4


def test_recall_miss() -> None:
    pred = {"type": "retrieval_recall", "expected_chunks": ["c1", "c2"]}
    state = _state(predicate=pred, calls=[[_hit("xx"), _hit("yy")]])
    out = _run(recall_at_k(), state)
    assert out.value == 0.0


def test_recall_unions_across_calls() -> None:
    pred = {"type": "retrieval_recall", "expected_chunks": ["c1", "c2", "c3"], "k": 5}
    state = _state(
        predicate=pred,
        calls=[
            [_hit("c1")],
            [_hit("c2"), _hit("c3")],
        ],
    )
    out = _run(recall_at_k(), state)
    assert out.value == 1.0


def test_recall_respects_k_truncation() -> None:
    """With k=2, only the first 2 hits per call count toward recall."""

    pred = {"type": "retrieval_recall", "expected_chunks": ["c1", "c5"], "k": 2}
    state = _state(
        predicate=pred,
        calls=[
            # c5 is the 4th hit — beyond k=2 → should NOT count.
            [_hit("c1"), _hit("c2"), _hit("c3"), _hit("c5")],
        ],
    )
    out = _run(recall_at_k(), state)
    assert out.value == 0.5


def test_recall_noanswer_on_non_retrieval_task() -> None:
    state = _state(
        predicate={"type": "workspace_file_exists", "path": "out.txt"},
        calls=[[_hit("c1")]],
    )
    out = _run(recall_at_k(), state)
    assert out.value == NOANSWER


def test_recall_noanswer_when_no_predicate() -> None:
    state = _state(predicate=None, calls=[[_hit("c1")]])
    out = _run(recall_at_k(), state)
    assert out.value == NOANSWER


def test_recall_noanswer_on_empty_expected_chunks() -> None:
    pred = {"type": "retrieval_recall", "expected_chunks": []}
    state = _state(predicate=pred, calls=[[_hit("c1")]])
    out = _run(recall_at_k(), state)
    assert out.value == NOANSWER


def test_recall_no_kb_query_calls() -> None:
    """No kb_query calls in the trajectory → recall is 0, not NOANSWER."""

    pred = {"type": "retrieval_recall", "expected_chunks": ["c1"]}
    state = _state(predicate=pred, calls=[])
    out = _run(recall_at_k(), state)
    assert out.value == 0.0
