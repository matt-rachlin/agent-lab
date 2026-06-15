"""Composition primitives — ADR-014 v0 (agent-as-tool + linear pipeline).

Pure unit tests: ``run_agent`` is monkeypatched; no GPU, no DB, no network.
"""

from __future__ import annotations

from typing import Any

import pytest
from lab.platform.agent_runtime import AgentResult, Tool
from lab.platform.composition import (
    PipelineResult,
    agent_as_tool,
    pipeline,
    summarize_result,
)


class _FakeSettings:
    litellm_key = "k"


def _tool(name: str, side_effect: str = "read") -> Tool:
    return Tool(
        name=name,
        description="d",
        parameters={"type": "object", "properties": {}},
        impl=lambda **_: {"ok": name},
        side_effect=side_effect,  # type: ignore[arg-type]
    )


def _agent_result(
    content: str = "done", tool_results: list[dict[str, Any]] | None = None
) -> AgentResult:
    return AgentResult(
        messages=[
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": content},
        ],
        tool_calls=len(tool_results or []),
        tool_results=tool_results or [],
        stop_reason="stop",
    )


# --- agent_as_tool ---------------------------------------------------------


def test_agent_as_tool_impl_calls_run_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run_agent(**kwargs: Any) -> AgentResult:
        captured.update(kwargs)
        return _agent_result(
            content="final answer",
            tool_results=[{"name": "search", "args": {}, "result": {"hits": 3}}],
        )

    monkeypatch.setattr("lab.platform.composition.run_agent", fake_run_agent)

    tool = agent_as_tool(
        name="worker",
        description="a worker sub-agent",
        system="you are a worker",
        tools=[_tool("search", "external_read")],
        model="m",
        settings=_FakeSettings(),  # type: ignore[arg-type]
        litellm_key="k",
    )

    out = tool.impl(input="do the thing")

    # impl actually drove run_agent with the sub-agent config + the input
    assert captured["model"] == "m"
    assert captured["system"] == "you are a worker"
    assert captured["user"] == "do the thing"
    assert [t.name for t in captured["tools"]] == ["search"]

    # compact result: final assistant content + tool_results summary + stop reason
    assert out["content"] == "final answer"
    assert out["stop_reason"] == "stop"
    assert out["tool_calls"] == 1
    assert out["tool_results"] == [{"name": "search", "result": {"hits": 3}}]


def test_agent_as_tool_side_effect_is_union_max() -> None:
    # mix of read + external_read + write_local => effective authority = write_local
    tool = agent_as_tool(
        name="worker",
        description="d",
        system="s",
        tools=[
            _tool("r", "read"),
            _tool("er", "external_read"),
            _tool("w", "write_local"),
        ],
        model="m",
        settings=_FakeSettings(),  # type: ignore[arg-type]
        litellm_key="k",
    )
    assert tool.side_effect == "write_local"


def test_agent_as_tool_side_effect_floor_is_read() -> None:
    tool = agent_as_tool(
        name="worker",
        description="d",
        system="s",
        tools=[],
        model="m",
        settings=_FakeSettings(),  # type: ignore[arg-type]
        litellm_key="k",
    )
    assert tool.side_effect == "read"


def test_agent_as_tool_side_effect_irreversible_dominates() -> None:
    tool = agent_as_tool(
        name="worker",
        description="d",
        system="s",
        tools=[_tool("w", "write_local"), _tool("rm", "irreversible"), _tool("r", "read")],
        model="m",
        settings=_FakeSettings(),  # type: ignore[arg-type]
        litellm_key="k",
    )
    assert tool.side_effect == "irreversible"


def test_summarize_result_picks_last_assistant_content() -> None:
    res = AgentResult(
        messages=[
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "last"},
        ],
        tool_calls=0,
        tool_results=[],
        stop_reason="stop",
    )
    assert summarize_result(res)["content"] == "last"


# --- pipeline --------------------------------------------------------------


def test_pipeline_threads_outputs_in_order() -> None:
    res = pipeline(
        [
            lambda x: x + 1,
            lambda x: x * 10,
            lambda x: x - 5,
        ],
        initial_input=1,
    )
    assert res.ok
    assert res.stage_outputs == [2, 20, 15]
    assert res.final_output == 15
    assert res.error_stage is None


def test_pipeline_records_per_stage_outputs_named() -> None:
    res = pipeline(
        [
            ("double", lambda x: x * 2),
            ("stringify", lambda x: f"v={x}"),
        ],
        initial_input=3,
    )
    assert res.stage_outputs == [6, "v=6"]
    assert res.final_output == "v=6"


def test_pipeline_stage_error_records_failing_stage_and_stops() -> None:
    reached: list[str] = []

    def stage0(x: Any) -> Any:
        reached.append("s0")
        return x

    def boom(_x: Any) -> Any:
        raise ValueError("bad payload")

    def stage2(x: Any) -> Any:
        reached.append("s2")  # must never run
        return x

    res = pipeline([("s0", stage0), ("boom", boom), ("s2", stage2)], initial_input=0)

    assert not res.ok
    assert res.error_stage == 1
    assert res.error_stage_name == "boom"
    assert res.error is not None
    assert "ValueError" in res.error
    assert "bad payload" in res.error
    # the rest of the pipeline is dropped
    assert reached == ["s0"]
    # only the successful stage's output was recorded
    assert res.stage_outputs == [0]
    assert res.final_output is None


def test_pipeline_agent_as_tool_impl_as_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lab.platform.composition.run_agent",
        lambda **_: _agent_result(content="synth"),
    )
    worker = agent_as_tool(
        name="synthesizer",
        description="d",
        system="s",
        tools=[_tool("read_kb", "read")],
        model="m",
        settings=_FakeSettings(),  # type: ignore[arg-type]
        litellm_key="k",
    )
    # an agent_as_tool impl can BE a pipeline stage (ADR-014 §2)
    res = pipeline(
        [
            lambda q: {"input": q},
            lambda payload: worker.impl(**payload),
            lambda summary: summary["content"],
        ],
        initial_input="research X",
    )
    assert res.ok
    assert res.final_output == "synth"


def test_pipeline_result_default_is_empty_ok() -> None:
    res = PipelineResult()
    assert res.ok
    assert res.stage_outputs == []
