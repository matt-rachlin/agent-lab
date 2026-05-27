"""Unit tests for `lab.inspect_bridge.scorers.rag.ndcg`."""

from __future__ import annotations

import asyncio
import math
from typing import Any

import pytest
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import NOANSWER, Target
from inspect_ai.solver import TaskState
from lab.inspect_bridge.scorers.rag import ndcg
from lab.tasks.registry import Task


def _state(
    *,
    predicate: dict[str, Any] | None,
    calls: list[list[dict[str, Any]]],
) -> TaskState:
    task = Task.model_validate(
        {
            "suite": "test",
            "slug": "ndcg",
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
    lab_agent = {"turns": turns}
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


def test_ndcg_perfect() -> None:
    pred = {"type": "retrieval_recall", "expected_chunks": ["c1", "c2"], "k": 5}
    state = _state(predicate=pred, calls=[[_hit("c1"), _hit("c2"), _hit("c3")]])
    out = _run(ndcg(), state)
    assert out.value == pytest.approx(1.0)


def test_ndcg_zero_when_no_match() -> None:
    pred = {"type": "retrieval_recall", "expected_chunks": ["c1", "c2"]}
    state = _state(predicate=pred, calls=[[_hit("x"), _hit("y"), _hit("z")]])
    out = _run(ndcg(), state)
    assert out.value == 0.0


def test_ndcg_partial_with_default_grades() -> None:
    """Default relevance = 1.0 per expected chunk.

    Agent retrieves [c1, c9, c2]. Expected = {c1, c2}.
      DCG  = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5
      IDCG = 1/log2(2) + 1/log2(3) ≈ 1.0 + 0.6309
      nDCG = 1.5 / 1.6309 ≈ 0.9197
    """

    pred = {"type": "retrieval_recall", "expected_chunks": ["c1", "c2"]}
    state = _state(predicate=pred, calls=[[_hit("c1"), _hit("c9"), _hit("c2")]])
    out = _run(ndcg(), state)
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert out.value == pytest.approx(dcg / idcg, rel=1e-4)


def test_ndcg_custom_grades() -> None:
    """relevance_grades overrides the default 1.0 per expected chunk."""

    pred = {
        "type": "retrieval_recall",
        "expected_chunks": ["c1", "c2"],
        "relevance_grades": {"c1": 3.0, "c2": 1.0},
    }
    # Agent retrieves c2 then c1 — sub-ideal order: c1 should come first.
    state = _state(predicate=pred, calls=[[_hit("c2"), _hit("c1")]])
    out = _run(ndcg(), state)
    dcg = 1.0 / math.log2(2) + 3.0 / math.log2(3)
    idcg = 3.0 / math.log2(2) + 1.0 / math.log2(3)
    assert out.value == pytest.approx(dcg / idcg, rel=1e-4)
    assert 0.0 < out.value < 1.0


def test_ndcg_truncates_at_k() -> None:
    """With k=2, only the agent's top-2 hits contribute to DCG."""

    pred = {"type": "retrieval_recall", "expected_chunks": ["c1", "c2"], "k": 2}
    # c2 only appears at rank 3 — should be ignored at k=2.
    state = _state(predicate=pred, calls=[[_hit("c1"), _hit("x"), _hit("c2")]])
    out = _run(ndcg(k=2), state)
    dcg = 1.0 / math.log2(2)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert out.value == pytest.approx(dcg / idcg, rel=1e-4)


def test_ndcg_noanswer_when_not_retrieval() -> None:
    state = _state(
        predicate={"type": "workspace_file_contains", "path": "x", "substring": "y"},
        calls=[[_hit("c1")]],
    )
    out = _run(ndcg(), state)
    assert out.value == NOANSWER


def test_ndcg_zero_when_no_kb_calls() -> None:
    pred = {"type": "retrieval_recall", "expected_chunks": ["c1"]}
    state = _state(predicate=pred, calls=[])
    out = _run(ndcg(), state)
    assert out.value == 0.0
