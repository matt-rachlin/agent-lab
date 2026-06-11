"""Fault-injection tests for the solver loop (mocked LiteLLM + ToolPool).

The point of these tests is to validate the `sandbox.faults` mechanism:
  * a fault fires on exactly the scheduled call_index (per-tool filter)
  * `error` / `timeout` modes replace the result WITHOUT dispatching
  * `truncate` / `wrong_result` modes dispatch the real call, then rewrite
  * a fault fires at most once — the retried call executes clean
  * no-fault passthrough leaves results and trajectory untouched
  * fired faults land in the turn entry (`fault_injected`) and the
    trajectory summary (`faults_fired`)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, ClassVar

import pytest

from lab.inspect_bridge import solver as solver_mod
from lab.inspect_bridge import tools as tools_mod
from lab.inspect_bridge.solver import FaultInjector


class _StubSandbox:
    container_name = "stub-sandbox"


class _StubPool:
    """Records tool invocations and returns canned results."""

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
    from inspect_ai.model import ChatMessageUser
    from inspect_ai.solver import TaskState

    return TaskState(
        model="lab/qwen3-14b-q4",
        sample_id=lab_task.slug,
        epoch=0,
        input=prompt,
        messages=[ChatMessageUser(content=prompt)],
        metadata={"lab_task": lab_task},
    )


def _lab_task(
    *,
    max_turns: int = 6,
    tool_budget: int = 8,
    faults: list[dict[str, Any]] | None = None,
) -> Any:
    from lab.tasks.registry import Task

    sandbox: dict[str, Any] = {"workspace_files": {"x": "stub"}}
    if faults is not None:
        sandbox["faults"] = faults
    return Task.model_validate(
        {
            "suite": "test",
            "slug": "stub",
            "input": "stub",
            "tools": [{"name": "fs_read"}, {"name": "fs_write"}],
            "max_turns": max_turns,
            "tool_budget": tool_budget,
            "sandbox": sandbox,
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

    responses: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []

    def fake_call_litellm_chat(**kwargs: Any) -> tuple[dict[str, Any], int]:
        requests.append(kwargs)
        if not responses:
            raise RuntimeError("no canned response for LiteLLM call")
        return responses.pop(0), 42

    monkeypatch.setattr(solver_mod, "call_litellm_chat", fake_call_litellm_chat)
    return {"responses": responses, "requests": requests}


def _tool_call_response(name: str, args: dict[str, Any], call_id: str = "c1") -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5},
    }


def _finish_response(text: str = "done") -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": text, "tool_calls": None}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
    }


def _run_solver(state: Any, solver: Any) -> Any:
    return asyncio.run(solver(state, _noop_generate))


async def _noop_generate(state: Any) -> Any:
    return state


def _run_with_faults(
    deps: dict[str, Any],
    *,
    faults: list[dict[str, Any]] | None,
    n_tool_turns: int = 1,
    tool: str = "fs_read",
) -> Any:
    """Drive `n_tool_turns` fs_read calls then a finish turn; return state."""

    responses = deps["responses"]
    for i in range(n_tool_turns):
        responses.append(_tool_call_response(tool, {"path": f"f{i}"}, call_id=f"c{i}"))
    responses.append(_finish_response())
    task = _lab_task(faults=faults)
    state = _make_state(lab_task=task)
    solver = solver_mod.model_with_tools(
        model="x", tool_budget=8, max_turns=8, sandbox=_StubSandbox()
    )
    return _run_solver(state, solver)


def _tool_entries(traj: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for turn in traj["turns"]:
        entries.extend(turn.get("tool_calls") or [])
    return entries


# ---------------------------------------------------------------------------
# end-to-end solver-loop tests
# ---------------------------------------------------------------------------


def test_error_fault_skips_dispatch_and_marks_turn(
    patch_solver_dependencies: dict[str, Any],
) -> None:
    out = _run_with_faults(
        patch_solver_dependencies,
        faults=[{"call_index": 1, "tool": "fs_read", "mode": "error"}],
    )
    pool = _StubPool.instances[0]
    assert pool.calls == []  # the call DID NOT execute
    traj = out.metadata["lab_agent"]
    entries = _tool_entries(traj)
    assert len(entries) == 1
    assert entries[0]["result"] == "ERROR: connection reset, retry may succeed"
    assert entries[0]["error"] is None
    assert entries[0]["fault_injected"] == {
        "mode": "error",
        "tool": "fs_read",
        "call_index": 1,
        "executed_real_call": False,
    }
    assert traj["faults_fired"] == [entries[0]["fault_injected"]]
    # The model saw the fault string as the tool message.
    tool_msgs = [m for m in out.messages if getattr(m, "role", "") == "tool"]
    assert tool_msgs
    assert "connection reset" in tool_msgs[0].text


def test_timeout_fault_skips_dispatch(
    patch_solver_dependencies: dict[str, Any],
) -> None:
    out = _run_with_faults(
        patch_solver_dependencies,
        faults=[{"call_index": 1, "tool": "*", "mode": "timeout"}],
    )
    pool = _StubPool.instances[0]
    assert pool.calls == []
    entries = _tool_entries(out.metadata["lab_agent"])
    assert entries[0]["result"] == "ERROR: tool call timed out after 30s"
    assert entries[0]["fault_injected"]["mode"] == "timeout"


def test_truncate_fault_executes_real_call_then_cuts(
    patch_solver_dependencies: dict[str, Any],
) -> None:
    responses = patch_solver_dependencies["responses"]
    responses.append(_tool_call_response("fs_read", {"path": "big"}))
    responses.append(_finish_response())
    task = _lab_task(
        faults=[
            {
                "call_index": 1,
                "tool": "fs_read",
                "mode": "truncate",
                "payload": {"keep_chars": 30},
            }
        ]
    )
    state = _make_state(lab_task=task)
    solver = solver_mod.model_with_tools(
        model="x", tool_budget=8, max_turns=8, sandbox=_StubSandbox()
    )
    pool_result = {"content": "A" * 500, "size": 500, "truncated": False, "path": "/workspace/big"}
    # The pool is instantiated inside solve(), so we can't queue results on
    # the instance up front — patch the class-level invoke for this test.
    orig_invoke = _StubPool.invoke

    def _invoke(self: _StubPool, module: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((module, tool_name, dict(arguments)))
        return pool_result

    _StubPool.invoke = _invoke  # type: ignore[method-assign]
    try:
        out = _run_solver(state, solver)
    finally:
        _StubPool.invoke = orig_invoke  # type: ignore[method-assign]

    pool = _StubPool.instances[0]
    assert len(pool.calls) == 1  # real call executed
    full = json.dumps(pool_result, default=str)
    expected = full[:30] + "...[TRUNCATED]"
    entries = _tool_entries(out.metadata["lab_agent"])
    assert entries[0]["result"] == expected
    assert entries[0]["fault_injected"]["executed_real_call"] is True
    tool_msgs = [m for m in out.messages if getattr(m, "role", "") == "tool"]
    assert tool_msgs[0].text == expected


def test_wrong_result_fault_replaces_result(
    patch_solver_dependencies: dict[str, Any],
) -> None:
    replacement = {"content": "bogus,data\n1,2\n", "size": 15, "truncated": False}
    out = _run_with_faults(
        patch_solver_dependencies,
        faults=[
            {
                "call_index": 1,
                "tool": "fs_read",
                "mode": "wrong_result",
                "payload": {"replacement": replacement},
            }
        ],
    )
    pool = _StubPool.instances[0]
    assert len(pool.calls) == 1  # real call executed (result discarded)
    entries = _tool_entries(out.metadata["lab_agent"])
    assert entries[0]["result"] == replacement
    tool_msgs = [m for m in out.messages if getattr(m, "role", "") == "tool"]
    assert json.loads(tool_msgs[0].text) == replacement


def test_fault_fires_on_correct_call_index_and_retry_succeeds(
    patch_solver_dependencies: dict[str, Any],
) -> None:
    # Three fs_read turns; the fault targets the 2nd call only.
    out = _run_with_faults(
        patch_solver_dependencies,
        faults=[{"call_index": 2, "tool": "fs_read", "mode": "error"}],
        n_tool_turns=3,
    )
    pool = _StubPool.instances[0]
    # Calls 1 and 3 dispatched; call 2 skipped.
    assert [c[2] for c in pool.calls] == [{"path": "f0"}, {"path": "f2"}]
    entries = _tool_entries(out.metadata["lab_agent"])
    assert "fault_injected" not in entries[0]
    assert entries[1]["fault_injected"]["call_index"] == 2
    assert "fault_injected" not in entries[2]  # the retry executed clean
    assert entries[2]["result"] == {"ok": True}


def test_tool_filter_does_not_count_other_tools(
    patch_solver_dependencies: dict[str, Any],
) -> None:
    responses = patch_solver_dependencies["responses"]
    responses.append(_tool_call_response("fs_read", {"path": "a"}, call_id="c0"))
    responses.append(_tool_call_response("fs_write", {"path": "b", "content": "x"}, call_id="c1"))
    responses.append(_finish_response())
    task = _lab_task(faults=[{"call_index": 1, "tool": "fs_write", "mode": "timeout"}])
    state = _make_state(lab_task=task)
    solver = solver_mod.model_with_tools(
        model="x", tool_budget=8, max_turns=8, sandbox=_StubSandbox()
    )
    out = _run_solver(state, solver)
    pool = _StubPool.instances[0]
    assert [c[1] for c in pool.calls] == ["fs_read"]  # fs_write was faulted
    entries = _tool_entries(out.metadata["lab_agent"])
    assert "fault_injected" not in entries[0]
    assert entries[1]["fault_injected"]["mode"] == "timeout"


def test_no_fault_passthrough(
    patch_solver_dependencies: dict[str, Any],
) -> None:
    # Schedule exists but targets a call index never reached.
    out = _run_with_faults(
        patch_solver_dependencies,
        faults=[{"call_index": 9, "tool": "fs_read", "mode": "error"}],
    )
    pool = _StubPool.instances[0]
    assert len(pool.calls) == 1
    traj = out.metadata["lab_agent"]
    entries = _tool_entries(traj)
    assert entries[0]["result"] == {"ok": True}
    assert "fault_injected" not in entries[0]
    assert traj["faults_fired"] == []


def test_task_without_faults_has_empty_marker(
    patch_solver_dependencies: dict[str, Any],
) -> None:
    out = _run_with_faults(patch_solver_dependencies, faults=None)
    traj = out.metadata["lab_agent"]
    assert traj["faults_fired"] == []
    assert all("fault_injected" not in e for e in _tool_entries(traj))


# ---------------------------------------------------------------------------
# FaultInjector unit tests (no solver loop)
# ---------------------------------------------------------------------------


def test_injector_skips_malformed_entries() -> None:
    injector = FaultInjector(
        [
            "not-a-dict",
            {"call_index": 0, "tool": "*", "mode": "error"},  # index < 1
            {"call_index": 1, "tool": "*", "mode": "explode"},  # unknown mode
            {"call_index": "x", "tool": "*", "mode": "error"},  # bad index
        ]
    )
    assert injector.match("fs_read") is None
    assert injector.fired_summary() == []


def test_injector_one_fault_per_call_next_one_defers() -> None:
    injector = FaultInjector(
        [
            {"call_index": 1, "tool": "fs_read", "mode": "error"},
            {"call_index": 1, "tool": "fs_read", "mode": "timeout"},
        ]
    )
    first = injector.match("fs_read")
    assert first is not None
    assert first.mode == "error"
    # The second entry's threshold was already reached; it fires on the
    # NEXT matching call instead of stacking on the same one.
    second = injector.match("fs_read")
    assert second is not None
    assert second.mode == "timeout"
    assert injector.match("fs_read") is None
    assert [f["mode"] for f in injector.fired_summary()] == ["error", "timeout"]


def test_compact_turns_keeps_fault_marker() -> None:
    """agent_logs.turns must retain `fault_injected` through compaction."""

    from lab.inspect_bridge.logwriter import _compact_turns

    marker = {"mode": "error", "tool": "fs_read", "call_index": 1, "executed_real_call": False}
    lab_agent = {
        "turns": [
            {
                "turn": 0,
                "latency_ms": 10,
                "tool_calls": [
                    {"tool": "fs_read", "latency_ms": 1, "error": None, "fault_injected": marker},
                    {"tool": "fs_read", "latency_ms": 2, "error": None},
                ],
            }
        ]
    }
    compact = _compact_turns(lab_agent)
    tools = compact[0]["tools"]
    assert tools[0]["fault_injected"] == marker
    assert "fault_injected" not in tools[1]


def test_injector_default_payloads() -> None:
    from lab.inspect_bridge.solver import _apply_truncate_fault

    assert _apply_truncate_fault("abcdef", {"keep_chars": 3}) == "abc...[TRUNCATED]"
    out = _apply_truncate_fault({"k": "v" * 300}, {})
    assert out.endswith("...[TRUNCATED]")
    assert len(out) == 200 + len("...[TRUNCATED]")
