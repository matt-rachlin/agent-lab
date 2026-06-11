"""plan_execute scaffold tests with mocked LiteLLM + mocked ToolPool.

Mirrors test_solver_loop.py's harness. Validates:
  * react stays the default — no planner record unless scaffold="plan_execute"
  * the planner call is tool-less and carries the planning system prompt
  * the plan is injected into the executor's system prompt (existing or new)
  * equal-budget accounting: planner + executor ≤ max_turns assistant calls,
    tool_budget untouched by the planner
  * the trajectory carries a turn-0 record with planner: true and the plan
    (truncated to 4 KB) plus the scaffold marker
  * planner failure is recorded and skips the executor
  * the adapter forwards `scaffold` into model_with_tools
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, ClassVar

import pytest

from lab.inspect_bridge import adapter as adapter_mod
from lab.inspect_bridge import solver as solver_mod
from lab.inspect_bridge import tools as tools_mod


class _StubSandbox:
    container_name = "stub-sandbox"


class _StubPool:
    """Records tool invocations and returns canned results."""

    instances: ClassVar[list[_StubPool]] = []

    def __init__(self, sandbox: Any) -> None:
        self.sandbox = sandbox
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.stopped = False
        _StubPool.instances.append(self)

    def invoke(self, module: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((module, tool_name, dict(arguments)))
        return {"ok": True}

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture(autouse=True)
def _clear_stubs() -> Any:
    _StubPool.instances.clear()
    yield
    _StubPool.instances.clear()


def _make_state(*, lab_task: Any, prompt: str = "do the thing", system: str | None = None) -> Any:
    """Build a minimal TaskState-compatible object the solver uses."""
    from inspect_ai.model import ChatMessageSystem, ChatMessageUser
    from inspect_ai.solver import TaskState

    messages: list[Any] = []
    if system is not None:
        messages.append(ChatMessageSystem(content=system))
    messages.append(ChatMessageUser(content=prompt))
    return TaskState(
        model="lab/qwen3-14b-q4",
        sample_id=lab_task.slug,
        epoch=0,
        input=prompt,
        messages=messages,
        metadata={"lab_task": lab_task},
    )


def _lab_task(*, max_turns: int = 3, tool_budget: int = 3, tools: list[Any] | None = None) -> Any:
    from lab.tasks.registry import Task

    return Task.model_validate(
        {
            "suite": "test",
            "slug": "stub",
            "input": "stub",
            "tools": tools if tools is not None else [{"name": "fs_read"}],
            "max_turns": max_turns,
            "tool_budget": tool_budget,
        }
    )


@pytest.fixture
def patch_solver_dependencies(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch LiteLLM + ToolPool + schema discovery used by the solver."""

    monkeypatch.setattr(solver_mod, "_read_litellm_key", lambda: "test-key")
    monkeypatch.setattr(solver_mod, "ToolPool", _StubPool)

    from lab.inspect_bridge.tools import ToolSchema

    schemas = {
        "fs_read": ToolSchema(
            name="fs_read",
            description="read a file",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        ),
    }
    monkeypatch.setattr(solver_mod, "discover_tool_schemas", lambda: schemas)

    from inspect_ai.tool import ToolDef, ToolParams

    def _fake_tools_for_task(
        task: Any, sandbox: Any, *, pool: Any = None, tool_names: Any = None
    ) -> list[Any]:
        async def _exec(**kwargs: Any) -> str:
            return "ok"

        return [
            ToolDef(
                tool=_exec,
                name="fs_read",
                description="read a file",
                parameters=ToolParams.model_validate(
                    {"type": "object", "properties": {"path": {"type": "string"}}}
                ),
            ).as_tool()
        ]

    monkeypatch.setattr(tools_mod, "lab_tools_for_task", _fake_tools_for_task)

    responses: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []

    def fake_call_litellm_chat(**kwargs: Any) -> tuple[dict[str, Any], int]:
        requests.append(kwargs)
        if not responses:
            raise RuntimeError("no canned response for LiteLLM call")
        return responses.pop(0), 42

    monkeypatch.setattr(solver_mod, "call_litellm_chat", fake_call_litellm_chat)
    return {"responses": responses, "requests": requests}


def _text_response(content: str, *, tokens_in: int = 10, tokens_out: int = 20) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content, "tool_calls": None}}],
        "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
    }


def _tool_call_response(i: int = 0) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"c{i}",
                            "type": "function",
                            "function": {
                                "name": "fs_read",
                                "arguments": json.dumps({"path": "x"}),
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5},
    }


def _run_solver(state: Any, solver: Any) -> Any:
    return asyncio.run(solver(state, _noop_generate))


async def _noop_generate(state: Any) -> Any:
    return state


_PLAN = "1. Read the file (artifact: file contents).\n2. Answer (artifact: final message)."


# ---------------------------------------------------------------------------
# Scaffold dispatch — react unchanged default
# ---------------------------------------------------------------------------


def test_react_default_has_no_planner_record(
    patch_solver_dependencies: dict[str, Any],
) -> None:
    responses = patch_solver_dependencies["responses"]
    requests = patch_solver_dependencies["requests"]
    responses.append(_text_response("all done"))

    task = _lab_task(max_turns=5, tool_budget=3)
    state = _make_state(lab_task=task)
    solver = solver_mod.model_with_tools(
        model="x", tool_budget=3, max_turns=5, sandbox=_StubSandbox()
    )
    out = _run_solver(state, solver)
    traj = out.metadata["lab_agent"]
    assert traj["scaffold"] == "react"
    assert traj["terminated_reason"] == "model_finished"
    assert len(requests) == 1
    # First (only) call carries the tool surface — no tool-less planner call.
    assert requests[0]["tools"] is not None
    assert all(not t.get("planner") for t in traj["turns"])


def test_plan_exec_requires_max_turns_at_least_two() -> None:
    with pytest.raises(ValueError, match="max_turns >= 2"):
        solver_mod.model_with_tools(
            model="x",
            tool_budget=1,
            max_turns=1,
            sandbox=_StubSandbox(),
            scaffold="plan_execute",
        )


# ---------------------------------------------------------------------------
# Planner call — message shape
# ---------------------------------------------------------------------------


def test_planner_call_message_shape(
    patch_solver_dependencies: dict[str, Any],
) -> None:
    responses = patch_solver_dependencies["responses"]
    requests = patch_solver_dependencies["requests"]
    responses.append(_text_response(_PLAN))
    responses.append(_text_response("all done"))

    task = _lab_task(max_turns=3, tool_budget=3)
    state = _make_state(lab_task=task, prompt="count the files")
    solver = solver_mod.model_with_tools(
        model="x", tool_budget=3, max_turns=3, sandbox=_StubSandbox(), scaffold="plan_execute"
    )
    _run_solver(state, solver)

    planner_req = requests[0]
    # Tool-less: the planner gets no tool surface.
    assert planner_req["tools"] is None
    msgs = planner_req["messages"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    # The planning prompt states the tools that WILL be available, demands
    # a numbered plan with artifacts, and forbids code blocks.
    assert "fs_read" in msgs[0]["content"]
    assert "numbered plan" in msgs[0]["content"]
    assert "artifact" in msgs[0]["content"]
    assert "code blocks" in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "count the files"}
    # Same per-call token cap as the executor turns (equal-budget).
    assert planner_req["max_tokens"] == requests[1]["max_tokens"]


# ---------------------------------------------------------------------------
# Plan injection into the executor's system prompt
# ---------------------------------------------------------------------------


def test_plan_appended_to_existing_system_prompt(
    patch_solver_dependencies: dict[str, Any],
) -> None:
    responses = patch_solver_dependencies["responses"]
    requests = patch_solver_dependencies["requests"]
    responses.append(_text_response(_PLAN))
    responses.append(_text_response("all done"))

    task = _lab_task(max_turns=3, tool_budget=3)
    state = _make_state(lab_task=task, system="You are a careful agent.")
    solver = solver_mod.model_with_tools(
        model="x", tool_budget=3, max_turns=3, sandbox=_StubSandbox(), scaffold="plan_execute"
    )
    _run_solver(state, solver)

    executor_system = requests[1]["messages"][0]
    assert executor_system["role"] == "system"
    assert executor_system["content"].startswith("You are a careful agent.")
    assert "A plan was prepared:\n" + _PLAN in executor_system["content"]
    assert "Follow it, adapting as results require." in executor_system["content"]


def test_plan_inserted_when_no_system_prompt(
    patch_solver_dependencies: dict[str, Any],
) -> None:
    responses = patch_solver_dependencies["responses"]
    requests = patch_solver_dependencies["requests"]
    responses.append(_text_response(_PLAN))
    responses.append(_text_response("all done"))

    task = _lab_task(max_turns=3, tool_budget=3)
    state = _make_state(lab_task=task)  # user message only
    solver = solver_mod.model_with_tools(
        model="x", tool_budget=3, max_turns=3, sandbox=_StubSandbox(), scaffold="plan_execute"
    )
    _run_solver(state, solver)

    executor_msgs = requests[1]["messages"]
    assert executor_msgs[0]["role"] == "system"
    assert executor_msgs[0]["content"].startswith("A plan was prepared:")
    assert executor_msgs[1]["role"] == "user"


# ---------------------------------------------------------------------------
# Budget accounting — the equal-budget guarantee
# ---------------------------------------------------------------------------


def test_total_assistant_calls_capped_at_max_turns(
    patch_solver_dependencies: dict[str, Any],
) -> None:
    responses = patch_solver_dependencies["responses"]
    requests = patch_solver_dependencies["requests"]
    responses.append(_text_response(_PLAN))
    # Executor keeps asking for tools; only max_turns - 1 = 2 calls happen.
    for i in range(5):
        responses.append(_tool_call_response(i))

    task = _lab_task(max_turns=3, tool_budget=99)
    state = _make_state(lab_task=task)
    solver = solver_mod.model_with_tools(
        model="x", tool_budget=99, max_turns=3, sandbox=_StubSandbox(), scaffold="plan_execute"
    )
    out = _run_solver(state, solver)
    traj = out.metadata["lab_agent"]
    # planner (1) + executor (max_turns - 1 = 2) == max_turns total LLM calls,
    # the same ceiling the react scaffold has.
    assert len(requests) == 3
    assert traj["actual_turns"] == 3
    assert traj["terminated_reason"] == "max_turns_reached"
    # Executor turn indices follow the planner's turn 0.
    executor_turns = [t["turn"] for t in traj["turns"] if not t.get("planner")]
    assert executor_turns == [1, 2]


def test_planner_does_not_consume_tool_budget(
    patch_solver_dependencies: dict[str, Any],
) -> None:
    responses = patch_solver_dependencies["responses"]
    responses.append(_text_response(_PLAN))
    responses.append(_tool_call_response(0))
    responses.append(_tool_call_response(1))  # budget already exhausted here

    task = _lab_task(max_turns=10, tool_budget=1)
    state = _make_state(lab_task=task)
    solver = solver_mod.model_with_tools(
        model="x", tool_budget=1, max_turns=10, sandbox=_StubSandbox(), scaffold="plan_execute"
    )
    out = _run_solver(state, solver)
    traj = out.metadata["lab_agent"]
    # The full tool_budget was available to the executor (planner used none).
    assert traj["tool_call_count"] == 1
    assert traj["terminated_reason"] == "budget_exhausted"
    assert len(_StubPool.instances[0].calls) == 1


# ---------------------------------------------------------------------------
# Trajectory — planner record
# ---------------------------------------------------------------------------


def test_trajectory_has_planner_record_with_tokens(
    patch_solver_dependencies: dict[str, Any],
) -> None:
    responses = patch_solver_dependencies["responses"]
    responses.append(_text_response(_PLAN, tokens_in=11, tokens_out=33))
    responses.append(_text_response("all done"))

    task = _lab_task(max_turns=3, tool_budget=3)
    state = _make_state(lab_task=task)
    solver = solver_mod.model_with_tools(
        model="x", tool_budget=3, max_turns=3, sandbox=_StubSandbox(), scaffold="plan_execute"
    )
    out = _run_solver(state, solver)
    traj = out.metadata["lab_agent"]
    assert traj["scaffold"] == "plan_execute"
    planner_turn = traj["turns"][0]
    assert planner_turn["turn"] == 0
    assert planner_turn["type"] == "turn"
    assert planner_turn["planner"] is True
    assert planner_turn["plan"] == _PLAN
    # Planner tokens land in the turn entry → _aggregate_tokens counts them.
    assert planner_turn["tokens_in"] == 11
    assert planner_turn["tokens_out"] == 33


def test_plan_truncated_in_trajectory_but_full_in_prompt(
    patch_solver_dependencies: dict[str, Any],
) -> None:
    responses = patch_solver_dependencies["responses"]
    requests = patch_solver_dependencies["requests"]
    long_plan = "1. " + ("x" * 6000)
    responses.append(_text_response(long_plan))
    responses.append(_text_response("all done"))

    task = _lab_task(max_turns=3, tool_budget=3)
    state = _make_state(lab_task=task)
    solver = solver_mod.model_with_tools(
        model="x", tool_budget=3, max_turns=3, sandbox=_StubSandbox(), scaffold="plan_execute"
    )
    out = _run_solver(state, solver)
    planner_turn = out.metadata["lab_agent"]["turns"][0]
    # Trajectory copy bounded at 4 KB...
    assert isinstance(planner_turn["plan"], dict)
    assert planner_turn["plan"]["_truncated"] is True
    assert len(planner_turn["plan"]["preview"]) == 4096
    # ...but the executor sees the full plan.
    assert long_plan in requests[1]["messages"][0]["content"]


def test_planner_failure_recorded_and_skips_executor(
    patch_solver_dependencies: dict[str, Any],
) -> None:
    requests = patch_solver_dependencies["requests"]
    # No canned responses → the planner call raises.
    task = _lab_task(max_turns=3, tool_budget=3)
    state = _make_state(lab_task=task)
    solver = solver_mod.model_with_tools(
        model="x", tool_budget=3, max_turns=3, sandbox=_StubSandbox(), scaffold="plan_execute"
    )
    out = _run_solver(state, solver)
    traj = out.metadata["lab_agent"]
    assert traj["terminated_reason"] == "litellm_error"
    assert traj["error"] is not None
    assert traj["error"].startswith("planner call failed")
    assert traj["turns"][0]["planner"] is True
    assert len(requests) == 1  # no executor calls after the planner died
    assert _StubPool.instances[0].stopped is True


# ---------------------------------------------------------------------------
# Logwriter — the planner flag survives turn compaction
# ---------------------------------------------------------------------------


def test_compact_turns_preserves_planner_flag() -> None:
    from lab.inspect_bridge.logwriter import _compact_turns

    lab_agent = {
        "turns": [
            {
                "turn": 0,
                "type": "turn",
                "planner": True,
                "latency_ms": 3,
                "tokens_in": 1,
                "tokens_out": 2,
                "tool_calls_requested": 0,
                "plan": "1. do the thing",
            },
            {"turn": 1, "latency_ms": 4, "tokens_in": 5, "tokens_out": 6},
        ]
    }
    compact = _compact_turns(lab_agent)
    assert compact[0]["planner"] is True
    assert "plan" not in compact[0]  # bulky text stays in MinIO / lab_agent
    assert "planner" not in compact[1]


# ---------------------------------------------------------------------------
# Adapter — scaffold forwarding
# ---------------------------------------------------------------------------


def _capture_model_with_tools(captured: dict[str, Any]) -> Any:
    from inspect_ai.solver import solver as solver_decorator

    @solver_decorator(name="plan_exec_test_stub")
    def _stub() -> Any:
        async def solve(state: Any, generate: Any) -> Any:
            return state

        return solve

    def fake(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _stub()

    return fake


def test_adapter_forwards_plan_execute_scaffold(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(adapter_mod, "model_with_tools", _capture_model_with_tools(captured))
    task = _lab_task(max_turns=3, tool_budget=2)
    adapter_mod.lab_task_to_inspect(task, model="x", scaffold="plan_execute")
    assert captured["scaffold"] == "plan_execute"


def test_adapter_scaffold_defaults_to_react(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(adapter_mod, "model_with_tools", _capture_model_with_tools(captured))
    task = _lab_task(max_turns=3, tool_budget=2)
    adapter_mod.lab_task_to_inspect(task, model="x")
    assert captured["scaffold"] == "react"
