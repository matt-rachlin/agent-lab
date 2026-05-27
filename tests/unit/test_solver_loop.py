"""Solver-loop tests with mocked LiteLLM + mocked ToolPool.

The point of these tests is to validate:
  * tool_budget exhaustion correctly terminates the loop
  * max_turns correctly terminates the loop
  * a clean model_finished termination is recorded
  * tool-call args parse from JSON-string and dict shapes
  * the per-turn trajectory lands in `state.metadata["lab_agent"]`
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, ClassVar

import pytest
from lab.inspect_bridge import solver as solver_mod
from lab.inspect_bridge import tools as tools_mod


class _StubSandbox:
    container_name = "stub-sandbox"


class _StubPool:
    """Records tool invocations and returns canned results.

    The solver constructs a `ToolPool(sandbox)` directly; we patch the
    `ToolPool` symbol in the solver module so we can plant this stub
    instead.
    """

    instances: ClassVar[list[_StubPool]] = []

    def __init__(self, sandbox: Any) -> None:
        self.sandbox = sandbox
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.results: list[Any] = []
        self.stopped = False
        _StubPool.instances.append(self)

    def queue_result(self, value: Any) -> None:
        self.results.append(value)

    def invoke(self, module: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((module, tool_name, dict(arguments)))
        if self.results:
            return self.results.pop(0)
        return {"ok": True}

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture(autouse=True)
def _clear_stubs() -> Any:
    _StubPool.instances.clear()
    yield
    _StubPool.instances.clear()


def _make_state(*, lab_task: Any, prompt: str = "do the thing") -> Any:
    """Build a minimal TaskState-compatible object the solver uses."""
    from inspect_ai.model import ChatMessageUser
    from inspect_ai.solver import TaskState

    state = TaskState(
        model="lab/qwen3-14b-q4",
        sample_id=lab_task.slug,
        epoch=0,
        input=prompt,
        messages=[ChatMessageUser(content=prompt)],
        metadata={"lab_task": lab_task},
    )
    return state


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

    # Pretend the litellm key file is on disk.
    monkeypatch.setattr(solver_mod, "_read_litellm_key", lambda: "test-key")

    # Patch the ToolPool symbol used by solver.py.
    monkeypatch.setattr(solver_mod, "ToolPool", _StubPool)

    # Discovery only needs to return name+schema; the solver uses the
    # schemas to build OpenAI tool specs (not used by these tests but the
    # call still happens).
    from lab.inspect_bridge.tools import ToolSchema

    schemas = {
        "fs_read": ToolSchema(
            name="fs_read",
            description="read",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        ),
        "fs_write": ToolSchema(
            name="fs_write",
            description="write",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            },
        ),
    }
    monkeypatch.setattr(solver_mod, "discover_tool_schemas", lambda: schemas)
    # `lab_tools_for_task` is imported inside the solve() coroutine; patch it
    # at the source module so the deferred import sees our version. Inspect
    # validates the assigned `state.tools` list, so we build real Inspect
    # ToolDef-wrapped callables here.
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
                description="read",
                parameters=ToolParams.model_validate(
                    {"type": "object", "properties": {"path": {"type": "string"}}}
                ),
            ).as_tool()
        ]

    monkeypatch.setattr(tools_mod, "lab_tools_for_task", _fake_tools_for_task)

    # Capture LiteLLM call sequences.
    responses: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []

    def fake_call_litellm_chat(**kwargs: Any) -> tuple[dict[str, Any], int]:
        requests.append(kwargs)
        if not responses:
            raise RuntimeError("no canned response for LiteLLM call")
        return responses.pop(0), 42

    monkeypatch.setattr(solver_mod, "call_litellm_chat", fake_call_litellm_chat)
    return {"responses": responses, "requests": requests}


class _FakeTool:
    """Has just enough to make _build_tool_specs happy."""

    def __init__(self, name: str) -> None:
        self.name = name


def _run_solver(state: Any, solver: Any) -> Any:
    return asyncio.run(solver(state, _noop_generate))


async def _noop_generate(state: Any) -> Any:
    return state


def test_model_finishes_without_tool_calls(
    patch_solver_dependencies: dict[str, Any],
) -> None:
    responses = patch_solver_dependencies["responses"]
    responses.append(
        {
            "choices": [{"message": {"content": "all done", "tool_calls": None}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }
    )

    task = _lab_task(max_turns=5, tool_budget=3)
    state = _make_state(lab_task=task)
    solver = solver_mod.model_with_tools(
        model="x", tool_budget=3, max_turns=5, sandbox=_StubSandbox()
    )
    out = _run_solver(state, solver)
    traj = out.metadata["lab_agent"]
    assert traj["terminated_reason"] == "model_finished"
    assert traj["actual_turns"] == 1
    assert traj["tool_call_count"] == 0
    assert _StubPool.instances[0].stopped is True


def test_tool_budget_exhaustion_terminates(
    patch_solver_dependencies: dict[str, Any],
) -> None:
    responses = patch_solver_dependencies["responses"]
    # Turn 1: model asks for a tool call.
    responses.append(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
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
    )
    # Turn 2: model asks for another tool call → budget already 0.
    responses.append(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c2",
                                "type": "function",
                                "function": {
                                    "name": "fs_read",
                                    "arguments": '{"path": "y"}',
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        }
    )

    task = _lab_task(max_turns=10, tool_budget=1)
    state = _make_state(lab_task=task)
    solver = solver_mod.model_with_tools(
        model="x", tool_budget=1, max_turns=10, sandbox=_StubSandbox()
    )
    out = _run_solver(state, solver)
    traj = out.metadata["lab_agent"]
    assert traj["terminated_reason"] == "budget_exhausted"
    assert traj["tool_call_count"] == 1
    # The pool only saw one call before the loop broke.
    assert len(_StubPool.instances[0].calls) == 1


def test_max_turns_termination(
    patch_solver_dependencies: dict[str, Any],
) -> None:
    responses = patch_solver_dependencies["responses"]
    # Two turns each calling fs_read; max_turns=2 stops us mid-loop.
    for i in range(5):
        responses.append(
            {
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
        )

    task = _lab_task(max_turns=2, tool_budget=99)
    state = _make_state(lab_task=task)
    solver = solver_mod.model_with_tools(
        model="x", tool_budget=99, max_turns=2, sandbox=_StubSandbox()
    )
    out = _run_solver(state, solver)
    traj = out.metadata["lab_agent"]
    assert traj["terminated_reason"] == "max_turns_reached"
    assert traj["actual_turns"] == 2


def test_litellm_error_recorded_in_trajectory(
    patch_solver_dependencies: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(**kwargs: Any) -> tuple[dict[str, Any], int]:
        raise RuntimeError("proxy down")

    monkeypatch.setattr(solver_mod, "call_litellm_chat", boom)
    task = _lab_task(max_turns=3, tool_budget=1)
    state = _make_state(lab_task=task)
    solver = solver_mod.model_with_tools(
        model="x", tool_budget=1, max_turns=3, sandbox=_StubSandbox()
    )
    out = _run_solver(state, solver)
    traj = out.metadata["lab_agent"]
    assert traj["terminated_reason"] == "litellm_error"
    assert "proxy down" in (traj["error"] or "")


def test_tool_call_arguments_passed_to_pool(
    patch_solver_dependencies: dict[str, Any],
) -> None:
    responses = patch_solver_dependencies["responses"]
    responses.append(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "abc",
                                "type": "function",
                                "function": {
                                    "name": "fs_read",
                                    "arguments": '{"path": "secret"}',
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        }
    )
    # Second turn: model finishes after seeing the tool result.
    responses.append(
        {
            "choices": [{"message": {"content": "done", "tool_calls": None}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }
    )

    task = _lab_task(max_turns=3, tool_budget=2)
    state = _make_state(lab_task=task)
    solver = solver_mod.model_with_tools(
        model="x", tool_budget=2, max_turns=3, sandbox=_StubSandbox()
    )
    out = _run_solver(state, solver)
    pool = _StubPool.instances[0]
    assert pool.calls == [("lab.agent.tools.fs_read", "fs_read", {"path": "secret"})]
    traj = out.metadata["lab_agent"]
    assert traj["terminated_reason"] == "model_finished"
    assert traj["tool_call_count"] == 1


def test_no_tools_when_budget_zero(
    patch_solver_dependencies: dict[str, Any],
) -> None:
    """With tool_budget=0, the solver must not even spin up a ToolPool."""

    responses = patch_solver_dependencies["responses"]
    responses.append(
        {
            "choices": [{"message": {"content": "fine", "tool_calls": None}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
    )

    task = _lab_task(max_turns=2, tool_budget=0)
    state = _make_state(lab_task=task)
    solver = solver_mod.model_with_tools(
        model="x", tool_budget=0, max_turns=2, sandbox=_StubSandbox()
    )
    out = _run_solver(state, solver)
    assert _StubPool.instances == []  # no pool created
    traj = out.metadata["lab_agent"]
    assert traj["terminated_reason"] == "model_finished"
    assert traj["tool_call_count"] == 0


def test_validation_errors_on_bad_args() -> None:
    from lab.inspect_bridge.solver import model_with_tools

    with pytest.raises(ValueError, match="max_turns"):
        model_with_tools(model="x", tool_budget=1, max_turns=0)
    with pytest.raises(ValueError, match="tool_budget"):
        model_with_tools(model="x", tool_budget=-1, max_turns=1)
