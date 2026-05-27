"""Unit tests for `lab.inspect_bridge.scorer.trajectory_judge`.

We don't hit a real LiteLLM here — `make_judge` is monkeypatched. The
test surface is (a) the prompt the judge receives includes the
trajectory bits and (b) the score is normalised through `_normalise_1_to_5`
when the judge returns a 1-5 integer reply.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_ai.scorer import NOANSWER, Target
from inspect_ai.solver import TaskState

from lab.inspect_bridge import scorer as scorer_mod
from lab.inspect_bridge.scorer import _normalise_1_to_5, trajectory_judge
from lab.tasks.registry import Task


def _state() -> TaskState:
    task = Task.model_validate(
        {
            "suite": "test",
            "slug": "tj",
            "input": "Read 'x.txt' and tell me what it says.",
            "max_turns": 3,
            "tool_budget": 2,
        }
    )
    lab_agent = {
        "actual_turns": 2,
        "tool_call_count": 1,
        "terminated_reason": "model_finished",
        "turns": [
            {
                "turn": 0,
                "content_preview": "I'll read the file.",
                "tool_calls": [
                    {
                        "tool": "fs_read",
                        "args": {"path": "x.txt"},
                        "result": "hello",
                    }
                ],
            },
            {
                "turn": 1,
                "content_preview": "The file says: hello",
            },
        ],
    }
    return TaskState(
        model="x",
        sample_id="s",
        epoch=0,
        input="Read 'x.txt' and tell me what it says.",
        messages=[
            ChatMessageUser(content="Read 'x.txt' and tell me what it says."),
            ChatMessageAssistant(content="The file says: hello"),
        ],
        metadata={"lab_task": task, "lab_agent": lab_agent},
    )


def _run(s: Any, state: TaskState) -> Any:
    return asyncio.run(s(state, Target("")))


def test_normalise_1_to_5_from_reasoning() -> None:
    """When the judge returns score=4 the helper rescales to 0.8."""
    assert _normalise_1_to_5(1.0, '{"score": 4, "reasoning": "ok"}') == 0.8
    assert _normalise_1_to_5(1.0, "score: 5 because reasons") == 1.0
    assert _normalise_1_to_5(1.0, "score: 1 minimal effort") == 0.2


def test_normalise_passthrough_when_no_int() -> None:
    assert _normalise_1_to_5(0.5, "judge thought it was middling") == 0.5
    assert _normalise_1_to_5(0.0, None) == 0.0


def test_trajectory_judge_calls_judge_with_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_make_judge(*, model: str, position_swap: bool = False) -> Any:
        captured["model"] = model
        captured["position_swap"] = position_swap

        def _judge(*, prompt: str) -> tuple[float, str | None]:
            captured["prompt"] = prompt
            return 1.0, '{"score": 4, "reasoning": "looks good"}'

        return _judge

    monkeypatch.setattr(scorer_mod, "make_judge", fake_make_judge)
    out = _run(trajectory_judge(judge_model="gpt-oss-120b-cloud"), _state())
    assert captured["model"] == "gpt-oss-120b-cloud"
    assert captured["position_swap"] is False
    assert "Task: Read 'x.txt'" in captured["prompt"]
    assert "fs_read" in captured["prompt"]
    # Score 4 → 0.8
    assert out.value == pytest.approx(0.8)


def test_trajectory_judge_handles_transport_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_make_judge(*, model: str, position_swap: bool = False) -> Any:
        def _judge(*, prompt: str) -> tuple[float, str | None]:
            raise ConnectionError("litellm down")

        return _judge

    monkeypatch.setattr(scorer_mod, "make_judge", fake_make_judge)
    out = _run(trajectory_judge(), _state())
    assert out.value == NOANSWER
    assert "judge unavailable" in (out.explanation or "")
