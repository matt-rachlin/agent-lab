"""Adapter smoke test — round-trip a lab Task through Inspect.

We don't talk to LiteLLM here; the solver is replaced with a passthrough.
The point is to confirm `lab_task_to_inspect` builds an Inspect Task that
Inspect itself can run end-to-end without exploding.
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.integration


def test_adapter_round_trip_runs_under_inspect(tmp_path: Any) -> None:
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ChatMessageAssistant
    from inspect_ai.solver import Generate, Solver, TaskState, solver

    from lab.inspect_bridge.adapter import lab_task_to_inspect
    from lab.tasks.registry import Task

    @solver(name="lab_smoke_passthrough")
    def _passthrough() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            state.messages.append(ChatMessageAssistant(content="pretend answer"))
            if state.metadata is None:
                state.metadata = {}
            state.metadata["lab_agent"] = {
                "actual_turns": 1,
                "tool_call_count": 0,
                "terminated_reason": "model_finished",
                "total_latency_ms": 0,
                "error": None,
                "turns": [
                    {
                        "turn": 0,
                        "latency_ms": 0,
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "tool_calls_requested": 0,
                    }
                ],
            }
            state.completed = True
            return state

        return solve

    passthrough = _passthrough()

    task = Task.model_validate(
        {
            "suite": "smoke",
            "slug": "adapter-smoke-1",
            "input": "say hi",
            "max_turns": 1,
            "tool_budget": 0,
            "gold_answer": "hi",
        }
    )
    inspect_task = lab_task_to_inspect(
        task, model="passthrough", solver_override=passthrough
    )

    logs: list[Any] = inspect_eval(
        inspect_task, display="none", log_samples=True, log_dir=str(tmp_path / "inspect")
    )
    assert len(logs) == 1
    log = logs[0]
    assert log.samples is not None
    assert len(log.samples) == 1
    sample = log.samples[0]
    # Score is the noop 0.0 from Phase 6d; presence is what we're verifying.
    assert sample.scores
    primary = next(iter(sample.scores.values()))
    assert primary.value == 0.0
    # Trajectory landed on metadata.
    assert sample.metadata is not None
    assert sample.metadata.get("lab_agent", {}).get("actual_turns") == 1
