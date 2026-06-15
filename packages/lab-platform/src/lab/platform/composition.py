"""Composition primitives — ADR-014 v0 (agent-as-tool + linear pipeline).

ADR-014 defines a **composition** as a topology of nodes joined by typed edges.
This module ships the two v0 primitives that ADR-014 §7 pulls first (NS-3
pipeline + orchestrator-worker) and that fall out of the ADR-012 Tool ABI with
minimal new machinery:

1. :func:`agent_as_tool` — "an agent is a tool whose implementation is another
   agent" (ADR-014 §1). Returns a :class:`~lab.core.agent_runtime.Tool` whose
   ``impl`` drives a sub-agent via ``run_agent`` and returns a compact result.
   Its ``side_effect`` is the **union (max) of the sub-agent's tools'
   side-effects** — a composition's effective authority is the union of its
   nodes (ADR-014 §5, authorization is composition-aware / gate on the UNION).

2. :func:`pipeline` — a linear typed DAG (v0 = sequential stages). Threads each
   stage's output into the next, records per-stage outputs for attribution
   (ADR-014 §5 attribution / process eval), and on a stage error drops the rest
   and records WHERE it failed.

Peer / cyclic orchestration is DEFERRED (ADR-014 §7 — do not build it here).

Pure in-process code; no GPU, no DB. ``run_agent`` is imported, not modified.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from lab.platform.agent_runtime import (
    DEFAULT_ALLOWED,
    AgentResult,
    SideEffect,
    Tool,
    run_agent,
)

from lab.core.settings import Settings

#: ADR-013 side-effect classes ordered by escalating authority. The effective
#: authority of a composition is the *maximum* over its nodes (a union semantic:
#: a write-capable node lifts the whole composition to write).
_SIDE_EFFECT_ORDER: tuple[SideEffect, ...] = (
    "read",
    "external_read",
    "write_local",
    "irreversible",
)


def _max_side_effect(tools: Sequence[Tool]) -> SideEffect:
    """Return the highest-authority side_effect across ``tools``.

    This is the composition's effective authority (ADR-014 §5): the union of its
    nodes' capabilities, collapsed onto the ADR-013 escalation ladder. An empty
    tool set is pure ``read`` (the floor).
    """
    rank = {se: i for i, se in enumerate(_SIDE_EFFECT_ORDER)}
    highest: SideEffect = "read"
    for tool in tools:
        if rank[tool.side_effect] > rank[highest]:
            highest = tool.side_effect
    return highest


def summarize_result(result: AgentResult) -> dict[str, Any]:
    """Compact, edge-friendly view of a sub-agent run (ADR-014 typed edge payload).

    Keeps the final assistant content + a per-tool-call summary + the stop
    reason; drops the full transcript so an orchestrator edge carries a bounded
    payload rather than the whole sub-agent history.
    """
    final_content = ""
    for msg in reversed(result.messages):
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str) and content:
                final_content = content
                break
    return {
        "content": final_content,
        "tool_calls": result.tool_calls,
        "stop_reason": result.stop_reason,
        "tool_results": [
            {"name": r.get("name", ""), "result": r.get("result")} for r in result.tool_results
        ],
    }


def agent_as_tool(
    *,
    name: str,
    description: str,
    system: str,
    tools: list[Tool],
    model: str,
    settings: Settings,
    litellm_key: str,
    parameters: dict[str, Any] | None = None,
    capability: str = "",
    actor: str | None = None,
    allow_side_effects: frozenset[str] | set[str] = DEFAULT_ALLOWED,
    max_turns: int = 24,
    max_tool_calls: int = 24,
    timeout: int = 90,
    num_ctx: int | None = None,
) -> Tool:
    """Wrap a sub-agent as a single :class:`Tool` (ADR-014 §1, agent-as-tool).

    The returned Tool takes one string argument (``input``, overridable via
    ``parameters``); its ``impl`` runs ``run_agent`` on the sub-agent configured
    by ``system``/``tools``/``model`` and returns :func:`summarize_result`.

    ``side_effect`` is :func:`_max_side_effect` over the sub-agent's ``tools`` —
    the composition's effective authority is the UNION of its nodes (ADR-014 §5).
    An orchestrator gating on this Tool therefore sees the worker's true reach.
    """
    schema: dict[str, Any] = parameters or {
        "type": "object",
        "properties": {"input": {"type": "string", "description": "Task for the sub-agent."}},
        "required": ["input"],
    }
    sub_actor = actor or f"agent:{name}"

    def _impl(input: str = "", **_: Any) -> dict[str, Any]:
        result = run_agent(
            settings=settings,
            litellm_key=litellm_key,
            model=model,
            system=system,
            user=input,
            tools=tools,
            actor=sub_actor,
            allow_side_effects=allow_side_effects,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            timeout=timeout,
            num_ctx=num_ctx,
        )
        return summarize_result(result)

    return Tool(
        name=name,
        description=description,
        parameters=schema,
        impl=_impl,
        side_effect=_max_side_effect(tools),
        capability=capability,
    )


#: A pipeline stage: a typed edge transform ``(prev_output) -> output``. May be
#: a plain transform or an ``agent_as_tool`` impl (ADR-014 §2).
Stage = Callable[[Any], Any]


@dataclass
class PipelineResult:
    """Outcome of a linear pipeline run (ADR-014 §2/§5 attribution hook).

    Records per-stage outputs (process eval / credit assignment), the final
    output, and — if a stage raised — which stage index/name errored and the
    error string. On error the remaining stages are dropped (error propagation
    is CAUGHT, not amplified — ADR-014 §5) and ``ok`` is ``False``.
    """

    stage_outputs: list[Any] = field(default_factory=list)
    final_output: Any = None
    error_stage: int | None = None
    error_stage_name: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True iff no stage errored."""
        return self.error_stage is None


def pipeline(
    stages: Sequence[Stage | tuple[str, Stage]],
    initial_input: Any = None,
) -> PipelineResult:
    """Run ``stages`` sequentially, threading each output into the next.

    Each stage is either a bare callable or a ``(name, callable)`` pair (the name
    is recorded for attribution). v0 is strictly sequential — no concurrency, no
    cycles (ADR-014 §7 defers the general orchestrator).

    Returns a :class:`PipelineResult` with per-stage outputs. If a stage raises,
    the run stops, the failing stage index/name + error are recorded, and the
    remaining stages are dropped.
    """
    result = PipelineResult()
    current: Any = initial_input
    for index, raw in enumerate(stages):
        if isinstance(raw, tuple):
            stage_name, fn = raw
        else:
            stage_name, fn = (getattr(raw, "__name__", f"stage_{index}"), raw)
        try:
            current = fn(current)
        except Exception as exc:
            result.error_stage = index
            result.error_stage_name = stage_name
            result.error = f"{type(exc).__name__}: {exc}"
            return result
        result.stage_outputs.append(current)
    result.final_output = current
    return result
