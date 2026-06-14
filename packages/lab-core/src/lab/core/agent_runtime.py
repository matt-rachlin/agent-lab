"""Lab Agent Runtime (LAR) — ADR-012.

ONE bounded tool-use loop for BOTH eval and deployment. Generalizes the scout's
proven loop (ADR-011) into the shared core: drives a model via call_litellm_chat,
dispatches tool calls through the Tool ABI, audits every call via record_action,
honors a turn/tool-call budget, and enforces a minimal ADR-013 side-effect gate.

v0 scope: in-process tool backend only (the sandboxed FastMCP/podman backend is
the #13 future seam). Authorization v0: a per-run allow-set over side-effect
classes — a write/irreversible tool the run did not authorize is refused before
dispatch and the denial is audited.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from lab.core.control import record_action
from lab.core.llm import call_litellm_chat
from lab.core.settings import Settings

SideEffect = Literal["read", "external_read", "write_local", "irreversible"]

#: Default authorization: read-only classes auto-execute (ADR-013 defaults).
DEFAULT_ALLOWED: frozenset[str] = frozenset({"read", "external_read"})


@dataclass(frozen=True)
class Tool:
    """A typed agent tool (ADR-012 Tool ABI). v0 = in-process callable.

    side_effect drives the ADR-013 authorization gate; capability is a free-form
    label for future per-capability grants.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    impl: Callable[..., Any]
    side_effect: SideEffect = "read"
    capability: str = ""

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class AgentResult:
    messages: list[dict[str, Any]]
    tool_calls: int
    tool_results: list[dict[str, Any]]
    stop_reason: str  # stop | max_turns | max_tool_calls | stop_predicate


def run_agent(
    *,
    settings: Settings,
    litellm_key: str,
    model: str,
    system: str,
    user: str,
    tools: list[Tool],
    actor: str = "agent",
    allow_side_effects: frozenset[str] | set[str] = DEFAULT_ALLOWED,
    max_turns: int = 24,
    max_tool_calls: int = 24,
    timeout: int = 90,
    num_ctx: int | None = None,
    extra: dict[str, Any] | None = None,
    stop_predicate: Callable[[list[dict[str, Any]]], bool] | None = None,
) -> AgentResult:
    """Drive `model` through `tools` until it stops, the budget is hit, or
    `stop_predicate(results_so_far)` is True. Every tool call (and every blocked
    call) is recorded via record_action under `actor`."""
    allowed = frozenset(allow_side_effects)
    by_name = {t.name: t for t in tools}
    schemas = [t.to_openai() for t in tools]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    call_extra: dict[str, Any] = {"think": False}
    if extra:
        call_extra.update(extra)
    if num_ctx:
        call_extra["num_ctx"] = num_ctx

    calls = 0
    results: list[dict[str, Any]] = []
    stop = "max_turns"
    for _ in range(max_turns):
        resp, _ms = call_litellm_chat(
            settings=settings,
            litellm_key=litellm_key,
            model=model,
            messages=messages,
            tools=schemas,
            tool_choice="auto",
            extra=call_extra,
            timeout=timeout,
        )
        msg = ((resp.get("choices") or [{}])[0]).get("message") or {}
        messages.append(msg)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            stop = "stop"
            break
        for tc in tool_calls:
            calls += 1
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            tool = by_name.get(name)
            result: Any
            if tool is None:
                result = {"error": f"unknown tool: {name}"}
            elif tool.side_effect not in allowed:
                record_action(actor=actor, action=f"blocked:{name}", args=args, outcome="denied")
                result = {"error": f"blocked: side_effect '{tool.side_effect}' not authorized"}
            else:
                record_action(actor=actor, action=f"tool:{name}", args=args, outcome=None)
                try:
                    result = tool.impl(**args)
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
            results.append({"name": name, "args": args, "result": result})
            messages.append(
                {"role": "tool", "tool_call_id": tc.get("id"), "content": json.dumps(result)[:4000]}
            )
        if stop_predicate is not None and stop_predicate(results):
            stop = "stop_predicate"
            break
        if calls >= max_tool_calls:
            stop = "max_tool_calls"
            break
    return AgentResult(messages=messages, tool_calls=calls, tool_results=results, stop_reason=stop)
