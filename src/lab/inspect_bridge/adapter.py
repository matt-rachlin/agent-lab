"""Adapt a lab `Task` into an Inspect `Task` for the agent harness.

The adapter builds a single-sample Inspect task whose solver is our
multi-turn `model_with_tools` and whose scorers are the 6e set
(`end_state`, `tool_correctness`, `budget_respected`, `trajectory_judge`),
selected per-task based on what the task actually requests. The full
lab `Task` is stashed in `Sample.metadata["lab_task"]` so the solver
and scorers can read its fields without having to re-load it.
"""

from __future__ import annotations

from typing import Any

from inspect_ai import Task as InspectTask
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver

from lab.agent.sandbox import Sandbox
from lab.inspect_bridge.scorer import (
    budget_respected,
    end_state,
    tool_correctness,
    trajectory_judge,
)
from lab.inspect_bridge.solver import model_with_tools
from lab.tasks.registry import Task as LabTask


def _select_scorers(task: LabTask) -> list[Scorer]:
    """Build the per-task scorer list from the lab `Task` shape.

    Heuristic:
      * `budget_respected` always included — it's cheap, model-agnostic,
        and the answer is meaningful for any agent run.
      * `tool_correctness` if `task.rubric.type == "tool_call"`.
      * `end_state(predicate)` if `task.success_predicate` is set.
      * `trajectory_judge` if `task.success_predicate.include_judge` is
        true (lets a task opt into the LLM-judged dimension without
        forcing the cost on every run).

    The order matters for the logwriter's primary-score preference (see
    `logwriter._select_primary_score`): we want `end_state` first when
    present, then `tool_correctness`, then `trajectory_judge`, then
    `budget_respected`. We build the list in that order; the logwriter
    re-checks by name so the order here is documentation, not load-bearing.
    """

    scorers: list[Scorer] = []
    if task.success_predicate:
        scorers.append(end_state(task.success_predicate))
    if task.rubric is not None and task.rubric.type == "tool_call":
        scorers.append(tool_correctness())
    if (
        isinstance(task.success_predicate, dict)
        and bool(task.success_predicate.get("include_judge"))
    ):
        judge_model = task.success_predicate.get("judge_model", "gpt-oss-120b-cloud")
        scorers.append(trajectory_judge(judge_model=judge_model))
    scorers.append(budget_respected())
    return scorers


def lab_task_to_inspect(
    task: LabTask,
    *,
    model: str,
    sandbox: Sandbox | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    solver_override: Solver | None = None,
) -> InspectTask:
    """Build an Inspect `Task` for one lab `Task`.

    Args:
        task: The lab task to run.
        model: LiteLLM model id (e.g. `"qwen3-14b-q4"`).
        sandbox: The Podman+gVisor sandbox the solver should run tools
            against. May be `None` for tool-less tasks (or unit tests).
        temperature, max_tokens: forwarded to the solver.
        solver_override: Inject a custom solver — used by tests to avoid
            hitting LiteLLM.
    """

    metadata: dict[str, Any] = {
        "lab_task": task,
        "lab_slug": task.slug,
        "lab_suite": task.suite,
        "lab_category": task.category,
        "lab_max_turns": task.max_turns,
        "lab_tool_budget": task.tool_budget,
    }
    target = task.gold_answer if task.gold_answer is not None else ""

    sample = Sample(
        input=task.input,
        target=target,
        metadata=metadata,
        id=task.slug,
    )

    tool_names = (
        [spec["name"] for spec in task.tools if isinstance(spec, dict) and "name" in spec]
        if task.tools
        else None
    )

    solver: Solver
    if solver_override is not None:
        solver = solver_override
    else:
        solver = model_with_tools(
            model=model,
            tool_budget=task.tool_budget,
            max_turns=task.max_turns,
            sandbox=sandbox,
            tool_names=tool_names,
            temperature=temperature,
            max_tokens=max_tokens or 1024,
        )

    scorers = _select_scorers(task)

    return InspectTask(
        dataset=[sample],
        solver=solver,
        scorer=scorers,
        name=f"lab-{task.suite}-{task.slug}",
    )


__all__ = ["_select_scorers", "lab_task_to_inspect"]
