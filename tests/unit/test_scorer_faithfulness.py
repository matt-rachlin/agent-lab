"""Unit tests for `lab.inspect_bridge.scorers.rag.faithfulness`."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_ai.scorer import NOANSWER, Target
from inspect_ai.solver import TaskState

from lab.inspect_bridge.scorers import rag as rag_mod
from lab.inspect_bridge.scorers.rag import faithfulness
from lab.tasks.registry import Task


def _state(
    *,
    calls: list[list[dict[str, Any]]],
    final_assistant: str = "The shell uses 2>&1 for stderr redirection.",
) -> TaskState:
    task = Task.model_validate(
        {
            "suite": "test",
            "slug": "faithfulness",
            "input": "Explain stderr redirection in bash.",
            "max_turns": 3,
            "tool_budget": 5,
            "success_predicate": {
                "type": "retrieval_recall",
                "expected_chunks": ["c1"],
                "include_faithfulness": True,
            },
            "tools": [{"name": "kb_query"}],
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
                        "args": {"kb_name": "bash", "question": "stderr"},
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
        input="Explain stderr redirection in bash.",
        messages=[
            ChatMessageUser(content="Explain stderr redirection in bash."),
            ChatMessageAssistant(content=final_assistant),
        ],
        metadata={"lab_task": task, "lab_agent": lab_agent},
    )


def _hit(chunk_id: str, text: str = "passage text") -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "source_url": f"https://example.com/{chunk_id}",
        "text": text,
        "score": 0.5,
    }


def _run(s: Any, state: TaskState) -> Any:
    return asyncio.run(s(state, Target("")))


def test_faithfulness_calls_judge_and_normalises(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_make_judge(*, model: str, position_swap: bool = False) -> Any:
        captured["model"] = model
        captured["position_swap"] = position_swap

        def _judge(*, prompt: str) -> tuple[float, str | None]:
            captured["prompt"] = prompt
            return 1.0, '{"score": 4, "reasoning": "claims supported"}'

        return _judge

    monkeypatch.setattr(rag_mod, "make_judge", fake_make_judge)
    state = _state(calls=[[_hit("c1", text="Use 2>&1 to redirect stderr.")]])
    out = _run(faithfulness(), state)
    assert captured["model"] == "gpt-oss-120b-cloud"
    assert captured["position_swap"] is False
    # Score 4 → 0.8
    assert out.value == pytest.approx(0.8)
    # Prompt contains the retrieved passage AND the final agent response.
    assert "Use 2>&1 to redirect stderr." in captured["prompt"]
    assert "2>&1 for stderr redirection" in captured["prompt"]


def test_faithfulness_noanswer_when_no_kb_calls() -> None:
    state = _state(calls=[])
    out = _run(faithfulness(), state)
    assert out.value == NOANSWER
    assert "no retrieval performed" in (out.explanation or "")


def test_faithfulness_noanswer_on_judge_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_make_judge(*, model: str, position_swap: bool = False) -> Any:
        def _judge(*, prompt: str) -> tuple[float, str | None]:
            raise ConnectionError("litellm exploded")

        return _judge

    monkeypatch.setattr(rag_mod, "make_judge", fake_make_judge)
    state = _state(calls=[[_hit("c1", text="Some passage.")]])
    out = _run(faithfulness(), state)
    assert out.value == NOANSWER
    assert "judge unavailable" in (out.explanation or "")


def test_faithfulness_noanswer_when_no_chunk_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calls happened but every hit had empty text — judge has nothing."""

    def fake_make_judge(*, model: str, position_swap: bool = False) -> Any:
        def _judge(*, prompt: str) -> tuple[float, str | None]:
            raise AssertionError("judge should not be called")

        return _judge

    monkeypatch.setattr(rag_mod, "make_judge", fake_make_judge)
    state = _state(calls=[[_hit("c1", text="")]])
    out = _run(faithfulness(), state)
    assert out.value == NOANSWER
    assert "no chunk text" in (out.explanation or "")


def test_faithfulness_noanswer_when_no_final_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_make_judge(*, model: str, position_swap: bool = False) -> Any:
        def _judge(*, prompt: str) -> tuple[float, str | None]:
            raise AssertionError("judge should not be called")

        return _judge

    monkeypatch.setattr(rag_mod, "make_judge", fake_make_judge)
    state = _state(calls=[[_hit("c1", text="hi")]], final_assistant="")
    out = _run(faithfulness(), state)
    assert out.value == NOANSWER


def test_faithfulness_deduplicates_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two kb_query calls returning the same chunk_id are counted once."""

    seen: dict[str, Any] = {}

    def fake_make_judge(*, model: str, position_swap: bool = False) -> Any:
        def _judge(*, prompt: str) -> tuple[float, str | None]:
            seen["prompt"] = prompt
            return 0.6, '{"score": 3, "reasoning": "ok"}'

        return _judge

    monkeypatch.setattr(rag_mod, "make_judge", fake_make_judge)
    hit = _hit("c1", text="Use redirection: cmd 2> /dev/null discards stderr.")
    state = _state(calls=[[hit], [hit]])
    out = _run(faithfulness(), state)
    assert out.value == pytest.approx(0.6)
    # The chunk header appears exactly once.
    assert seen["prompt"].count("[chunk c1") == 1
