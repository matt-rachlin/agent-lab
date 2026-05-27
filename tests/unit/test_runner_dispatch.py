"""Unit tests for the agent-vs-single-turn dispatch in `lab.sweep.runner`.

The point of these tests is to verify that `execute_cell` routes to the
right path based on the task's `max_turns`/`tool_budget`. We don't actually
talk to LiteLLM or sandbox here — single-turn fast-path cells must continue
to work exactly as they did before 6d, and agent cells must trigger the
agent path (which we intercept).
"""

from __future__ import annotations

from typing import Any

import pytest
from lab.sweep.runner import _is_agent_cell


def test_default_task_is_single_turn() -> None:
    """Phase 1-5 tasks have max_turns=1 and tool_budget=0 implicitly."""
    assert _is_agent_cell({"input": "hi"}) is False


def test_max_turns_one_with_no_tools_is_single_turn() -> None:
    assert _is_agent_cell({"input": "hi", "max_turns": 1, "tool_budget": 0}) is False


def test_max_turns_above_one_is_agent_cell() -> None:
    assert _is_agent_cell({"input": "hi", "max_turns": 2, "tool_budget": 0}) is True


def test_tool_budget_above_zero_is_agent_cell() -> None:
    assert _is_agent_cell({"input": "hi", "max_turns": 1, "tool_budget": 3}) is True


def test_both_above_default_is_agent_cell() -> None:
    assert _is_agent_cell({"input": "hi", "max_turns": 5, "tool_budget": 10}) is True


def test_dispatch_routes_single_turn_cells_through_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lab.sweep import runner as runner_mod

    fast_calls: list[Any] = []
    agent_calls: list[Any] = []

    def fake_single(**kwargs: Any) -> Any:
        fast_calls.append(kwargs)
        return runner_mod.CellResult(
            run_id=kwargs["cell"].run_id,
            status="done",
            tokens_in=None,
            tokens_out=None,
            latency_ms=0,
            cost_usd=None,
            error=None,
            response_text=None,
            raw_response=None,
        )

    def fake_agent(**kwargs: Any) -> Any:
        agent_calls.append(kwargs)
        return runner_mod.CellResult(
            run_id=kwargs["cell"].run_id,
            status="done",
            tokens_in=None,
            tokens_out=None,
            latency_ms=0,
            cost_usd=None,
            error=None,
            response_text=None,
            raw_response=None,
        )

    monkeypatch.setattr(runner_mod, "_execute_single_turn", fake_single)
    monkeypatch.setattr(runner_mod, "_execute_agent_cell", fake_agent)
    # Stub manifest capture so we don't touch git / disk.
    monkeypatch.setattr(
        runner_mod,
        "capture_manifest",
        lambda extra: type("M", (), {"sha": "deadbeef"})(),
    )

    cell = runner_mod.Cell(
        run_id="run-1",
        experiment_id=1,
        experiment_slug="EXP",
        model_id=2,
        model_litellm_id="m",
        model_backend="ollama-local",
        task_id=3,
        task_slug="t",
        task_payload={"input": "hi"},  # single-turn default
        config=runner_mod.RunConfig(name="c"),
        config_hash="h",
        seed=0,
    )
    runner_mod.execute_cell(cell, litellm_key="k", timeout=10)
    assert len(fast_calls) == 1
    assert len(agent_calls) == 0


def test_dispatch_routes_agent_cells_through_agent_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lab.sweep import runner as runner_mod

    fast_calls: list[Any] = []
    agent_calls: list[Any] = []

    monkeypatch.setattr(
        runner_mod, "_execute_single_turn", lambda **kw: fast_calls.append(kw) or None
    )

    def fake_agent(**kwargs: Any) -> Any:
        agent_calls.append(kwargs)
        return runner_mod.CellResult(
            run_id=kwargs["cell"].run_id,
            status="done",
            tokens_in=None,
            tokens_out=None,
            latency_ms=0,
            cost_usd=None,
            error=None,
            response_text=None,
            raw_response=None,
        )

    monkeypatch.setattr(runner_mod, "_execute_agent_cell", fake_agent)
    monkeypatch.setattr(
        runner_mod,
        "capture_manifest",
        lambda extra: type("M", (), {"sha": "deadbeef"})(),
    )

    cell = runner_mod.Cell(
        run_id="run-1",
        experiment_id=1,
        experiment_slug="EXP",
        model_id=2,
        model_litellm_id="m",
        model_backend="ollama-local",
        task_id=3,
        task_slug="t",
        task_payload={
            "input": "hi",
            "max_turns": 3,
            "tool_budget": 2,
            "tools": [{"name": "fs_read"}],
        },
        config=runner_mod.RunConfig(name="c"),
        config_hash="h",
        seed=0,
    )
    runner_mod.execute_cell(cell, litellm_key="k", timeout=10)
    assert len(fast_calls) == 0
    assert len(agent_calls) == 1
