"""Solver wrapping the LiteLLM proxy with tools=[]. Body lands in 6d."""

from __future__ import annotations

from inspect_ai.solver import Solver


def model_with_tools(model: str, tool_budget: int) -> Solver:
    raise NotImplementedError("6d — not yet implemented")
