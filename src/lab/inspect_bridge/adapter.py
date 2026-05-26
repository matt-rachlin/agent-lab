"""Adapt a lab `Task` into an Inspect `Task` for the agent harness.

The adapter builds a single-sample Inspect task whose solver is our
multi-turn `model_with_tools` and whose scorer is a no-op for Phase 6d — 6e
replaces the scorer with the real evaluators. The full lab `Task` is
stashed in `Sample.metadata["lab_task"]` so the solver and (eventually)
scorers can read its fields without having to re-load it.
"""

from __future__ import annotations

from typing import Any

from inspect_ai import Task as InspectTask
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import Solver, TaskState

from lab.agent.sandbox import Sandbox
from lab.inspect_bridge.solver import model_with_tools
from lab.tasks.registry import Task as LabTask


@scorer(metrics=[accuracy()], name="lab_noop")
def _noop_scorer() -> Scorer:
    """Placeholder scorer for Phase 6d. Always returns 0.0.

    6e replaces this with the real `end_state` / `tool_correctness` /
    `budget_respected` / `trajectory_judge` scorers. We keep a registered
    scorer here (rather than no scorer at all) so the Inspect pipeline
    runs end-to-end without modification.
    """

    async def score(state: TaskState, target: Target) -> Score:
        # The trajectory is in state.metadata['lab_agent'] — 6e scorers will
        # pull from there. For now we record it as the explanation so a
        # human reading the log gets some signal.
        traj = (state.metadata or {}).get("lab_agent") or {}
        return Score(
            value=0.0,
            answer=None,
            explanation=f"noop scorer (6d); turns={traj.get('actual_turns')} "
            f"tool_calls={traj.get('tool_call_count')} "
            f"terminated={traj.get('terminated_reason')}",
            metadata={"lab_agent": traj},
        )

    return score


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

    return InspectTask(
        dataset=[sample],
        solver=solver,
        scorer=_noop_scorer(),
        name=f"lab-{task.suite}-{task.slug}",
    )


__all__ = ["lab_task_to_inspect"]
