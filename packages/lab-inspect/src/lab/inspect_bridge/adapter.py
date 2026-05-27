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
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver

from lab.agent.sandbox import Sandbox
from lab.eval.prompts import PromptNotFoundError, PromptRegistry
from lab.inspect_bridge.scorer import (
    budget_respected,
    end_state,
    tool_correctness,
    trajectory_judge,
)
from lab.inspect_bridge.scorers.rag import (
    attribution,
    faithfulness,
    mrr,
    ndcg,
    recall_at_k,
)
from lab.inspect_bridge.solver import model_with_tools
from lab.tasks.registry import Task as LabTask


def _task_uses_kb_query(task: LabTask) -> bool:
    """True if the task's tool list includes the `kb_query` MCP tool."""

    if not task.tools:
        return False
    return any(isinstance(spec, dict) and spec.get("name") == "kb_query" for spec in task.tools)


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
      * RAG family (`recall_at_k`, `mrr`, `ndcg`, `attribution`) if
        `task.success_predicate.type == "retrieval_recall"`.
      * `faithfulness` if `success_predicate.include_faithfulness` is
        true AND the task wires up the `kb_query` tool — otherwise we'd
        be running a judge on a trajectory with nothing to ground
        against.

    The order matters for the logwriter's primary-score preference (see
    `logwriter._select_primary_score`): on RAG tasks `recall_at_k`
    should outrank `tool_correctness` and `budget_respected`. We build
    the list in that order; the logwriter re-checks by name so the
    order here is documentation, not load-bearing.
    """

    scorers: list[Scorer] = []
    pred = task.success_predicate
    pred_type = pred.get("type") if isinstance(pred, dict) else None

    if pred:
        scorers.append(end_state(pred))
    if pred_type == "retrieval_recall":
        retrieval_k = int(pred.get("k", 5)) if isinstance(pred, dict) else 5
        scorers.append(recall_at_k(k=retrieval_k))
        scorers.append(mrr())
        scorers.append(ndcg(k=retrieval_k))
        scorers.append(attribution())
    if task.rubric is not None and task.rubric.type == "tool_call":
        scorers.append(tool_correctness())
    if (
        isinstance(pred, dict)
        and bool(pred.get("include_faithfulness"))
        and _task_uses_kb_query(task)
    ):
        judge_model = pred.get("judge_model", "gpt-oss-120b-cloud")
        scorers.append(faithfulness(judge_model=judge_model))
    if isinstance(pred, dict) and bool(pred.get("include_judge")):
        judge_model = pred.get("judge_model", "gpt-oss-120b-cloud")
        scorers.append(trajectory_judge(judge_model=judge_model))
    scorers.append(budget_respected())
    return scorers


def _resolve_system_prompt(
    task: LabTask,
    *,
    registry: PromptRegistry | None = None,
) -> tuple[str | None, str | None]:
    """Return ``(system_body, prompt_id_used)`` for ``task``.

    Precedence:

    1. If ``task.system_prompt_id`` is set, resolve via ``PromptRegistry``.
       Missing prompt → raise ``PromptNotFoundError`` (build-time, NOT
       run-time) so the failure is loud and ergonomic.
    2. Otherwise, fall back to the inlined ``task.system`` string.
    3. If neither is set, return ``(None, None)``.

    Both set is rejected by the Task model_validator; this is defence in
    depth (assertion-style) in case the task came from somewhere that
    skipped validation.
    """
    if task.system is not None and task.system_prompt_id is not None:
        raise ValueError(
            f"task {task.suite}/{task.slug} sets both 'system' and "
            f"'system_prompt_id'; choose exactly one"
        )
    if task.system_prompt_id is not None:
        reg = registry if registry is not None else PromptRegistry()
        try:
            body = reg.get(task.system_prompt_id)
        except PromptNotFoundError as exc:
            raise PromptNotFoundError(
                f"task {task.suite}/{task.slug} references "
                f"system_prompt_id={task.system_prompt_id!r}, but no such "
                f"prompt exists in the registry. Original: {exc}"
            ) from exc
        return body, task.system_prompt_id
    if task.system is not None:
        return task.system, None
    return None, None


def lab_task_to_inspect(
    task: LabTask,
    *,
    model: str,
    sandbox: Sandbox | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    solver_override: Solver | None = None,
    extra: dict[str, Any] | None = None,
    prompt_registry: PromptRegistry | None = None,
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
        prompt_registry: Override the prompt registry used to resolve
            ``task.system_prompt_id`` (default: ``PromptRegistry()``).
    """

    # Resolve the effective system prompt FIRST so a missing prompt fails
    # at adapter-build time with a clear error, not in the middle of the
    # solver loop.
    system_body, prompt_id_used = _resolve_system_prompt(task, registry=prompt_registry)

    metadata: dict[str, Any] = {
        "lab_task": task,
        "lab_slug": task.slug,
        "lab_suite": task.suite,
        "lab_category": task.category,
        "lab_max_turns": task.max_turns,
        "lab_tool_budget": task.tool_budget,
        # Persisted into the trajectory so post-hoc analysis can correlate
        # outcomes with the prompt body that actually ran.
        "lab_prompt_id_used": prompt_id_used,
    }
    target = task.gold_answer if task.gold_answer is not None else ""

    # Build Sample.input as either a plain string (no system prompt) or a
    # list of ChatMessage objects when a system prompt is present. Inspect
    # accepts both shapes; the list form is what we need to inject the
    # system message into state.messages before the solver starts.
    sample_input: str | list[Any]
    if system_body is not None:
        sample_input = [
            ChatMessageSystem(content=system_body),
            ChatMessageUser(content=task.input),
        ]
    else:
        sample_input = task.input

    sample = Sample(
        input=sample_input,
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
            extra=extra,
        )

    scorers = _select_scorers(task)

    return InspectTask(
        dataset=[sample],
        solver=solver,
        scorer=scorers,
        name=f"lab-{task.suite}-{task.slug}",
    )


__all__ = ["_resolve_system_prompt", "_select_scorers", "lab_task_to_inspect"]
