"""Multi-turn agent loop driving the LiteLLM proxy with MCP tools.

The Solver is the heart of Phase 6d. It:

  * spins up a `ToolPool` for the lifetime of the cell (one long-lived MCP
    server per `(sandbox, tool)` instead of per-call),
  * hands the model a tool surface translated from the lab Task,
  * runs the assistant ↔ tool ↔ assistant ↔ ... loop until the model is
    done, the budget is exhausted, the turn cap is hit, or the sandbox
    errors out,
  * dumps per-turn instrumentation into `state.metadata` so the logwriter
    can pull it out for `agent_logs.turns`.

The solver does NOT score — that's 6e's job. It does write a partial
trajectory into `state.metadata["lab_agent"]` even on error so the failure
mode is visible after the fact.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
import uuid
from typing import Any

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool

from lab.agent.sandbox import Sandbox
from lab.agent.tool_pool import ToolPool
from lab.core.llm import call_litellm_chat
from lab.core.model_pool import ModelPool, PipelineModelPlan, PipelineStep
from lab.core.settings import get_settings
from lab.inspect_bridge.tools import discover_tool_schemas
from lab.observability.log import get_logger
from lab.observability.tracing import current_span_attrs, span

log = get_logger(__name__)

# Truncation budget for tool call inputs/outputs recorded in `turns`.
_TURN_PAYLOAD_CAP = 4096

# Truncation budget for the planner's plan text recorded in the trajectory
# (`plan_execute` scaffold). The FULL plan is still injected into the
# executor's system prompt — this cap only bounds the audit record.
_PLAN_CAP = 4096


def _build_planner_system_prompt(tool_specs: list[dict[str, Any]]) -> str:
    """System prompt for the tool-less planner call (`plan_execute` Phase A).

    States the tools that WILL be available in the execution phase, demands
    a numbered plan with an expected artifact per step, and forbids code
    blocks (the planner cannot execute anything; code belongs to Phase B).
    """

    if tool_specs:
        tool_lines = "\n".join(
            f"- {spec['function']['name']}: {spec['function'].get('description') or ''}".rstrip()
            for spec in tool_specs
        )
    else:
        tool_lines = "- (no tools will be available; the executor must answer directly)"
    return (
        "You are the planning phase of a two-phase agent. You cannot call "
        "tools in this phase. In the execution phase that follows, an agent "
        "WILL have access to these tools:\n"
        f"{tool_lines}\n\n"
        "Write a concise numbered plan for completing the task below. For "
        "each step, state the expected artifact it should produce (a file "
        "written, a query result, a verified observation). Do not include "
        "code blocks. Output only the plan."
    )


def _first_user_content(chat: list[dict[str, Any]]) -> str:
    """The task input as the planner sees it: the first user message."""

    for msg in chat:
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            return str(content)
    return ""


def _extract_assistant_text(message: dict[str, Any]) -> tuple[str, str]:
    """Pull (content_text, reasoning_text) off a chat-completions assistant message.

    Most lanes return `content` as a plain string and no reasoning. The
    gemma4-12b lane (ollama_chat backend) is a thinking model: LiteLLM maps
    Ollama's `thinking` field to `reasoning_content` and returns `content`
    as an empty string for turns where the model only thinks and/or
    tool-calls — verified live against the proxy (forensic-audit follow-up;
    gemma4 trajectories showed tokens_out in the hundreds-to-thousands with
    empty content on every tool-call turn). Defensive extras: `content`
    may be None, or a list of parts (`[{"type": "text", "text": ...}, ...]`)
    on providers that return multi-part content — flatten part text in order.

    The reasoning text is returned separately and must NOT be merged into
    content: it is not assistant-visible output, and merging would pollute
    the text-tool-call recovery regex and the conversation echo.
    """

    raw = message.get("content")
    if isinstance(raw, str):
        content = raw
    elif isinstance(raw, list):
        parts: list[str] = []
        for part in raw:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        content = "\n".join(p for p in parts if p)
    elif raw is None:
        content = ""
    else:
        content = str(raw)

    reasoning = message.get("reasoning_content")
    reasoning_text = reasoning if isinstance(reasoning, str) else ""
    return content, reasoning_text


def _truncate(value: Any, cap: int = _TURN_PAYLOAD_CAP) -> Any:
    """Bound large tool I/O in the recorded trajectory.

    We keep the structural shape (JSON-serialisable) but if the string repr
    exceeds `cap`, replace it with a `{"_truncated": True, "preview": ...}`
    marker. Keeps trajectory rows in `agent_logs.turns` small enough for the
    DB; the full untruncated copy still lands in MinIO.

    Special case for kb_query-shaped results: when ``value`` is a dict
    carrying a ``hits`` list, we preserve the chunk_id / source_url /
    score / section_path fields on each hit (the RAG scorers' read keys)
    and only truncate each hit's ``text`` and ``summary`` strings. Without
    this carve-out, recall_at_k / mrr / ndcg / attribution would all
    score 0 on any RAG task whose retrieved chunks crossed the 4 KB cap
    — even when the agent correctly returned the gold chunks. F-005's
    successor finding; surfaced during the 6h-e RAG smoke.
    """

    if isinstance(value, dict) and isinstance(value.get("hits"), list):
        return _truncate_kb_query_result(value, cap)

    try:
        text = json.dumps(value, default=str)
    except Exception:
        text = str(value)
    if len(text) <= cap:
        return value
    return {"_truncated": True, "preview": text[:cap], "original_size": len(text)}


def _truncate_kb_query_result(value: dict[str, Any], cap: int) -> dict[str, Any]:
    """Truncate a kb_query result while preserving structure for RAG scorers.

    Strategy: keep all top-level fields except ``hits``; for each hit,
    keep the cheap structural fields (chunk_id, source_url, score, etc.)
    verbatim and progressively trim ``text`` + ``summary`` until the
    serialised payload fits the cap. We never drop hits — losing a hit
    silently would underreport recall in exactly the way the cap is meant
    to NOT do.
    """

    # Cheap path: maybe it already fits.
    try:
        full = json.dumps(value, default=str)
        if len(full) <= cap:
            return value
    except Exception:  # noqa: S110 - serialisation failure means we must trim; fall through
        pass

    base: dict[str, Any] = {k: v for k, v in value.items() if k != "hits"}
    pruned_hits: list[dict[str, Any]] = []
    for hit in value.get("hits") or []:
        if not isinstance(hit, dict):
            pruned_hits.append(hit)
            continue
        pruned: dict[str, Any] = {}
        for k, v in hit.items():
            if k in ("text", "summary") and isinstance(v, str) and len(v) > 240:
                pruned[k] = v[:240] + "…"
            else:
                pruned[k] = v
        pruned_hits.append(pruned)

    candidate = {**base, "hits": pruned_hits, "_hits_text_trimmed": True}
    try:
        if len(json.dumps(candidate, default=str)) <= cap:
            return candidate
    except Exception:  # noqa: S110 - same rationale as the upper try; fall through to minimal
        pass

    # If still over cap, drop text entirely from each hit but keep the
    # structural fields needed by RAG scorers (chunk_id, source_url,
    # section_path, score) and the Phase 7 rerank signal — the agent's
    # recall/attribution can still be measured even without text, and
    # rerank_score / stage1_rank are tiny but indispensable for EXP-004
    # post-hoc analysis. The full untruncated payload is in MinIO; this
    # branch keeps the DB row scorer-readable.
    minimal_hits = [
        {
            kk: vv
            for kk, vv in hit.items()
            if kk
            in {
                "chunk_id",
                "source_url",
                "section_path",
                "score",
                "title",
                "rerank_score",
                "stage1_rank",
                "dense_score",
                "sparse_score",
            }
        }
        for hit in pruned_hits
        if isinstance(hit, dict)
    ]
    minimal = {**base, "hits": minimal_hits, "_hits_text_dropped": True}
    return minimal


def _read_litellm_key() -> str:
    """Read the LiteLLM master key from the canonical on-box path.

    Settings exposes the key too, but in practice we read it from disk in
    the runner — keep the same source-of-truth here so a stale env doesn't
    silently override the active proxy creds.
    """

    settings = get_settings()
    if settings.litellm_key:
        return settings.litellm_key
    from pathlib import Path

    candidate = Path("/data/lab/services/litellm-master-key")
    if candidate.exists():
        return candidate.read_text().strip()
    return ""


def _build_tool_specs(tool_names: list[str]) -> list[dict[str, Any]]:
    """Translate MCP tool names into the OpenAI `tools=[...]` shape.

    We talk to LiteLLM directly (not through Inspect's `model_graded.generate`).
    The MCP servers are our source of truth for tool schemas; we look them up
    once at solver construction time and bake the OpenAI envelope.
    """

    schemas = discover_tool_schemas()
    specs: list[dict[str, Any]] = []
    for name in tool_names:
        if name not in schemas:
            continue
        schema = schemas[name]
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": schema.description,
                    "parameters": schema.input_schema,
                },
            }
        )
    return specs


def _coerce_arguments(raw: Any) -> dict[str, Any]:
    """Tool call `arguments` arrives as a JSON string from OpenAI-compat servers.

    Some servers (and Ollama-cloud routes) pre-parse the dict — accept both
    shapes. On parse failure, raise so we can surface it as a turn error.
    """

    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"tool arguments not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"tool arguments must be a JSON object; got {type(parsed).__name__}")
        return parsed
    raise ValueError(f"tool arguments must be a dict or JSON string; got {type(raw).__name__}")


def _serialise_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make sure outgoing messages are pure JSON (no Inspect types leak)."""

    return [dict(m) for m in messages]


def _extract_text_tool_calls(content: str, valid_names: set[str]) -> list[dict[str, Any]]:
    """Recover tool calls a model emitted as JSON *text* in its content instead of
    structured tool_calls — e.g. ``{"name": "fs_read", "arguments": {...}}`` or
    ``{"type": "function", "name": ..., "parameters": {...}}``. Several capable
    models (Llama-3.3-70B, Qwen2.5-Coder) do this and would otherwise score 0 on
    multi-step tasks. Best-effort: only objects whose ``name`` is a known tool and
    that carry arguments/parameters are recovered. Returns OpenAI-format calls.
    """
    recovered: list[dict[str, Any]] = []
    for match in re.finditer(r"\{(?:[^{}]|\{[^{}]*\})*\}", content):
        try:
            obj = json.loads(match.group(0))
        except (ValueError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        args = obj.get("arguments")
        if args is None:
            args = obj.get("parameters")
        if not isinstance(name, str) or name not in valid_names or args is None:
            continue
        recovered.append(
            {
                "id": f"recovered_{len(recovered)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args if isinstance(args, str) else json.dumps(args),
                },
            }
        )
    return recovered


# ---------------------------------------------------------------------------
# Fault injection — deterministic tool-fault schedules from `sandbox.faults`.
#
# A task may declare `sandbox: {faults: [{call_index, tool, mode, payload}]}`
# to measure agent behaviour when TOOLS misbehave (vs the brutal suite's
# broken DATA). Faults fire on the Nth dispatched tool call matching the
# entry's tool filter (per-entry counter, 1-based, across the episode) and
# fire at most once, so a retry of the same call later in the episode goes
# through clean. The model only ever sees the faulted result; every fired
# fault is recorded in the turn entry (`fault_injected`) and summarised in
# the trajectory (`faults_fired`) so analysis can verify firing post-hoc.
# ---------------------------------------------------------------------------

_FAULT_MODES = frozenset({"error", "timeout", "truncate", "wrong_result"})
_FAULT_ERROR_MESSAGE = "ERROR: connection reset, retry may succeed"
_FAULT_TIMEOUT_MESSAGE = "ERROR: tool call timed out after 30s"
_FAULT_TRUNCATE_KEEP_CHARS = 200


class _ScheduledFault:
    """One entry from `sandbox.faults`, plus its firing state."""

    __slots__ = ("call_index", "fired", "mode", "payload", "seen", "tool")

    def __init__(self, *, call_index: int, tool: str, mode: str, payload: dict[str, Any]) -> None:
        self.call_index = call_index
        self.tool = tool
        self.mode = mode
        self.payload = payload
        self.seen = 0  # dispatched calls matching `tool` so far
        self.fired = False

    @property
    def executes_real_call(self) -> bool:
        """truncate / wrong_result run the real tool; error / timeout skip it."""

        return self.mode in ("truncate", "wrong_result")

    def describe(self) -> dict[str, Any]:
        """JSON-safe marker recorded in turn entries / the trajectory."""

        return {
            "mode": self.mode,
            "tool": self.tool,
            "call_index": self.call_index,
            "executed_real_call": self.executes_real_call,
        }


class FaultInjector:
    """Deterministic per-episode scheduler over a task's `sandbox.faults` list.

    Each fault entry keeps its own counter of dispatched calls matching its
    tool filter (`"*"` matches everything). When the counter reaches
    `call_index` the fault fires — once. At most one fault applies per call
    (first entry in declaration order wins); a later-listed fault aimed at
    the same index fires on the next matching call instead.

    Malformed entries (unknown mode, non-positive call_index) are skipped
    with a warning rather than failing the cell — `sandbox` is free-form
    jsonb and a typo'd schedule should degrade to a clean episode, not an
    errored one.
    """

    def __init__(self, faults: list[Any]) -> None:
        self._faults: list[_ScheduledFault] = []
        for raw in faults:
            if not isinstance(raw, dict):
                log.warning("fault_entry_skipped_not_dict", entry=str(raw)[:200])
                continue
            mode = raw.get("mode")
            try:
                call_index = int(raw.get("call_index", 0))
            except (TypeError, ValueError):
                call_index = 0
            if mode not in _FAULT_MODES or call_index < 1:
                log.warning(
                    "fault_entry_skipped_invalid",
                    mode=str(mode),
                    call_index=raw.get("call_index"),
                )
                continue
            payload = raw.get("payload")
            self._faults.append(
                _ScheduledFault(
                    call_index=call_index,
                    tool=str(raw.get("tool", "*") or "*"),
                    mode=str(mode),
                    payload=payload if isinstance(payload, dict) else {},
                )
            )

    def match(self, tool_name: str) -> _ScheduledFault | None:
        """Register one dispatched call of `tool_name`; return the fault to apply.

        Counts the call against every entry whose filter matches, then
        returns the first not-yet-fired entry whose threshold is reached
        (marking it fired). Returns None when no fault applies.
        """

        chosen: _ScheduledFault | None = None
        for fault in self._faults:
            if fault.tool not in ("*", tool_name):
                continue
            fault.seen += 1
            if chosen is None and not fault.fired and fault.seen >= fault.call_index:
                fault.fired = True
                chosen = fault
        return chosen

    def fired_summary(self) -> list[dict[str, Any]]:
        """Markers for every fault that fired (for the trajectory record)."""

        return [f.describe() for f in self._faults if f.fired]


def _fault_skip_result(fault: _ScheduledFault) -> str:
    """Model-visible result for the skip-dispatch modes (error / timeout)."""

    if fault.mode == "timeout":
        message = fault.payload.get("message")
        return message if isinstance(message, str) else _FAULT_TIMEOUT_MESSAGE
    message = fault.payload.get("message")
    return message if isinstance(message, str) else _FAULT_ERROR_MESSAGE


def _apply_truncate_fault(result: Any, payload: dict[str, Any]) -> str:
    """Keep only the first `payload.keep_chars` of the serialised result."""

    try:
        keep = int(payload.get("keep_chars", _FAULT_TRUNCATE_KEEP_CHARS))
    except (TypeError, ValueError):
        keep = _FAULT_TRUNCATE_KEEP_CHARS
    keep = max(keep, 1)
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    return text[:keep] + "...[TRUNCATED]"


def _execute_tool_calls(
    *,
    tool_calls: list[dict[str, Any]],
    pool: ToolPool | None,
    tool_modules: dict[str, str],
    remaining_budget: int,
    injector: FaultInjector | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Run each requested tool call; return (new chat msgs, turn entries, calls done).

    `tool_modules` maps tool name → dotted MCP server module. `remaining_budget`
    is decremented for every call we actually dispatch (failed or successful);
    if we hit zero mid-batch we return early so the caller can break the loop.

    When `injector` is set, each dispatchable call is offered to the fault
    schedule first: `error`/`timeout` faults replace the result WITHOUT
    executing the tool; `truncate`/`wrong_result` execute the real call and
    rewrite what the model sees. Faulted calls still consume budget — the
    agent spent the call, same as a genuinely failed one.
    """

    new_messages: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    calls_done = 0
    for call in tool_calls:
        if remaining_budget <= 0:
            entries.append(
                {
                    "tool": call.get("function", {}).get("name"),
                    "skipped": "budget_exhausted",
                }
            )
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "name": call.get("function", {}).get("name", ""),
                    "content": "tool budget exhausted",
                }
            )
            continue

        fn = call.get("function", {}) or {}
        name = fn.get("name") or ""
        call_id = call.get("id", "")
        t0 = time.monotonic()
        fault: _ScheduledFault | None = None
        try:
            args = _coerce_arguments(fn.get("arguments", "{}"))
            if name not in tool_modules:
                raise ValueError(f"unknown tool {name!r}")
            if pool is None:
                raise RuntimeError("tool call received but no pool configured (sandbox missing)")
            # Fault schedule: only calls that would actually dispatch count
            # towards `call_index` (unknown tools / missing pool don't).
            if injector is not None:
                fault = injector.match(name)
            result: Any
            if fault is not None and not fault.executes_real_call:
                # error / timeout: the call DOES NOT execute.
                result = _fault_skip_result(fault)
            else:
                result = pool.invoke(tool_modules[name], name, args)
                if fault is not None and fault.mode == "truncate":
                    result = _apply_truncate_fault(result, fault.payload)
                elif fault is not None and fault.mode == "wrong_result":
                    result = fault.payload.get("replacement")
            if fault is not None:
                log.info(
                    "fault_injected",
                    tool=name,
                    mode=fault.mode,
                    call_index=fault.call_index,
                )
            error: str | None = None
        except Exception as exc:
            raw_args = fn.get("arguments")
            args = raw_args if isinstance(raw_args, dict) else {}
            result = {"error": f"{type(exc).__name__}: {exc}"}
            error = str(exc)
        latency_ms = int((time.monotonic() - t0) * 1000)
        calls_done += 1
        remaining_budget -= 1

        # The OpenAI spec says tool messages carry a string content. We
        # serialise whatever we got back; if the model expects structured
        # JSON, it can parse it.
        content = result if isinstance(result, str) else json.dumps(result, default=str)
        new_messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": content,
            }
        )
        entry: dict[str, Any] = {
            "tool": name,
            "args": _truncate(args),
            "result": _truncate(result),
            "latency_ms": latency_ms,
            "error": error,
        }
        if fault is not None:
            entry["fault_injected"] = fault.describe()
        entries.append(entry)
    return new_messages, entries, calls_done


@solver(name="lab_model_with_tools")
def model_with_tools(
    *,
    model: str,
    model_backend: str | None = None,
    tool_budget: int = 10,
    max_turns: int = 5,
    sandbox: Sandbox | None = None,
    tool_names: list[str] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    extra: dict[str, Any] | None = None,
    scaffold: str = "react",
) -> Solver:
    """Solver: drives `model` through up to `max_turns` of tool-using chat.

    Args:
        model: LiteLLM model name routed through the proxy at `localhost:4000`.
        tool_budget: max tool calls across all turns (≥0). When 0, the model
            is given no tools and the loop is effectively single-turn.
        max_turns: max assistant turns. Always ≥1 — the first turn is the
            model's first response to the prompt.
        sandbox: container the tool pool runs against. If None, no tools are
            attached regardless of the task's `tools` list.
        tool_names: optional restriction over `task.tools`.
        temperature, max_tokens: forwarded to the proxy.
        scaffold: `"plan_execute"` prepends a tool-less planner call whose
            plan is injected into the executor's system prompt (see the
            EQUAL-BUDGET GUARANTEE comment in solve()). Any other value
            (`"react"`, legacy `"single_turn"`) runs the plain react loop
            unchanged — react remains the default behaviour.

    Returns:
        An Inspect Solver that mutates `state.messages` and stashes a per-
        turn record at `state.metadata["lab_agent"]`.
    """

    if max_turns < 1:
        raise ValueError("max_turns must be >= 1")
    if tool_budget < 0:
        raise ValueError("tool_budget must be >= 0")
    if scaffold == "plan_execute" and max_turns < 2:
        # The planner consumes one of the max_turns assistant-call slots;
        # with max_turns=1 the executor would get zero turns. Fail loudly
        # at build time rather than producing an un-comparable cell.
        raise ValueError("plan_execute scaffold requires max_turns >= 2")
    # SGLang Phase 1 / B2: the sglang-local (-awq) arm is single-turn BFCL ONLY
    # until the multi-turn agent path is validated against it. Fail loudly here
    # rather than letting an sglang-local arm silently run the untested agent
    # loop (the ModelPool/llama-swap path below would *accept* it, which is
    # exactly the silent-success we want to prevent). Remove this guard when
    # the agent path is validated for SGLang.
    if model_backend == "sglang-local":
        raise ValueError(
            "sglang-local backend is not yet validated on the agent path "
            "(SGLang Phase 1 B2 — single-turn BFCL only)"
        )

    settings = get_settings()
    litellm_key = _read_litellm_key()

    # Capture the tool module map once so the inner loop does no extra disk
    # I/O per call. The schemas are already discovered eagerly by
    # `_build_tool_specs` further down (called once per solve()).
    from lab.agent.tools import TOOL_SERVERS

    tool_modules: dict[str, str] = dict(TOOL_SERVERS)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Establish per-cell pool + tools. We import here to keep the
        # solver decorator-time imports light (and to make unit tests that
        # monkeypatch easier to write).
        from lab.inspect_bridge.tools import lab_tools_for_task

        run_started = time.monotonic()
        task_meta = state.metadata.get("lab_task") if state.metadata else None
        # Fall back to whatever messages Inspect built; we re-use them as
        # the starting conversation.
        chat: list[dict[str, Any]] = []
        for msg in state.messages:
            chat.append(_chat_message_to_dict(msg))

        # Phase 19c — declare a per-turn pipeline plan so llama-swap can
        # pre-flight the model into the page cache and we get explicit
        # eviction at solve() end. One step per turn; predictive load
        # fires after each turn for turn+1. Best-effort: failures are
        # swallowed by ModelPool internals so the solver path never
        # gates inference on llama-swap health.
        model_pool: ModelPool | None = None
        side_models: list[str] = []
        if task_meta is not None and getattr(task_meta, "tools", None):
            for spec in task_meta.tools or []:
                if isinstance(spec, dict) and spec.get("name") == "kb_query":
                    side_models = ["qwen3-embedding", "qwen3-reranker-0.6b"]
                    break
        if model_backend == "ollama-local":
            # Served by Ollama (:11434), not llama-swap (:8080) — the pool's
            # warm/teardown calls would 400. Ollama keep_alive handles residency.
            log.debug("solver_model_pool_skipped_ollama_local", model=model)
        else:
            try:
                model_pool = ModelPool(llama_swap_url=settings.llama_swap_url)
                # llama-swap registers big models without the litellm `-local`
                # suffix (e.g. llama-3.3-70b-q4, not ...-local); using the litellm
                # id makes preflight/warm 400. Ollama-backed models are skipped
                # above via model_backend, so this only affects llama-swap models.
                pool_model = model.removesuffix("-local")
                plan_steps = [
                    PipelineStep(name=f"turn_{i}", models=[pool_model, *side_models])
                    for i in range(max(max_turns, 1))
                ]
                pipeline_id = f"agent-{uuid.uuid4().hex[:8]}"
                model_pool.declare(PipelineModelPlan(pipeline_id=pipeline_id, steps=plan_steps))
            except Exception as exc:
                log.warning(
                    "solver_model_pool_declare_failed",
                    model=model,
                    error=str(exc),
                )
                model_pool = None

        # Fault injection: the task's free-form `sandbox` dict may carry a
        # `faults` schedule (see FaultInjector). One injector per episode —
        # call counters span all turns.
        injector: FaultInjector | None = None
        sandbox_cfg = getattr(task_meta, "sandbox", None) if task_meta is not None else None
        if isinstance(sandbox_cfg, dict):
            raw_faults = sandbox_cfg.get("faults")
            if isinstance(raw_faults, list) and raw_faults:
                injector = FaultInjector(raw_faults)

        pool: ToolPool | None = None
        tools: list[Tool] = []
        if sandbox is not None and task_meta is not None and tool_budget > 0:
            pool = ToolPool(sandbox)
            try:
                tools = lab_tools_for_task(task_meta, sandbox, pool=pool, tool_names=tool_names)
            except Exception as exc:
                # Surface tool-discovery failures as a partial trajectory.
                _stash_trajectory(
                    state,
                    error=f"tool discovery failed: {exc}",
                    turns=[],
                    actual_turns=0,
                    tool_call_count=0,
                    total_latency_ms=int((time.monotonic() - run_started) * 1000),
                    scaffold=scaffold,
                )
                pool.stop()
                return state
        state.tools = tools

        # Resolve the effective tool-name list from the task. `tool_names` is the
        # solver-level filter; the task carries the canonical list.
        effective_tool_names: list[str] = []
        if task_meta is not None and getattr(task_meta, "tools", None):
            for spec in task_meta.tools or []:
                if isinstance(spec, dict) and "name" in spec:
                    effective_tool_names.append(spec["name"])
        if tool_names is not None:
            effective_tool_names = [n for n in effective_tool_names if n in tool_names]
        tool_specs = _build_tool_specs(effective_tool_names) if tools else []
        turns: list[dict[str, Any]] = []
        remaining_budget = tool_budget
        actual_turns = 0
        tool_call_count = 0
        terminated_reason = "model_finished"
        error: str | None = None
        # First loop index for the executor. Stays 0 for react; the
        # plan_execute planner phase below claims index 0 and bumps this
        # to 1 (or to max_turns on planner failure, skipping the loop).
        executor_start_turn = 0

        try:
            if scaffold == "plan_execute":
                # ---- Phase A: planner (one tool-less call) -------------
                #
                # EQUAL-BUDGET GUARANTEE (the react vs plan_execute A/B is
                # only valid because all three budget axes are identical):
                #
                #   1. Assistant calls: the planner call is one assistant
                #      LLM call against the SAME model, capped at the SAME
                #      `max_tokens`, and it CONSUMES one `max_turns` slot —
                #      the executor loop below runs turn indices
                #      1..max_turns-1, so the episode makes at most
                #      `max_turns` assistant calls total. React's ceiling
                #      is the same: max_turns calls x max_tokens each.
                #   2. Tool calls: `tool_budget` is untouched (the planner
                #      is offered no tools), so both scaffolds may dispatch
                #      at most `tool_budget` tool calls.
                #   3. Token accounting: the planner's usage is recorded in
                #      its turn entry below; the logwriter's
                #      `_aggregate_tokens` sums `lab_agent.turns`, so
                #      planner tokens COUNT toward the cell's reported
                #      tokens_in/tokens_out exactly like any other turn.
                if model_pool is not None:
                    with contextlib.suppress(Exception):
                        model_pool.step_start("turn_0")
                planner_started = time.monotonic()
                planner_messages = [
                    {
                        "role": "system",
                        "content": _build_planner_system_prompt(tool_specs),
                    },
                    {"role": "user", "content": _first_user_content(chat)},
                ]
                try:
                    with span("planner_call", **{"lab.model": model, "lab.planner": True}):
                        planner_resp, planner_latency_ms = call_litellm_chat(
                            settings=settings,
                            litellm_key=litellm_key,
                            model=model,
                            messages=planner_messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            tools=None,
                            extra=extra,
                        )
                except Exception as exc:
                    error = f"planner call failed: {exc}"
                    terminated_reason = "litellm_error"
                    log.warning("planner_call_failed", model=model, error=str(exc))
                    turns.append(
                        {
                            "turn": 0,
                            "type": "turn",
                            "planner": True,
                            "error": error,
                            "latency_ms": int((time.monotonic() - planner_started) * 1000),
                        }
                    )
                    # Skip the executor loop entirely (range below is empty);
                    # the guarded for-else preserves `litellm_error`.
                    executor_start_turn = max_turns
                else:
                    actual_turns += 1
                    planner_choice = (planner_resp.get("choices") or [{}])[0]
                    planner_msg = (planner_choice or {}).get("message") or {}
                    plan_text, planner_reasoning = _extract_assistant_text(planner_msg)
                    if not plan_text.strip():
                        plan_text = "(planner returned an empty plan)"
                    planner_usage = planner_resp.get("usage") or {}
                    # Turn-0-style record with the planner flag so audits
                    # see the call inline with the executor turns. The plan
                    # is stored truncated (_PLAN_CAP); the executor gets the
                    # full text.
                    planner_entry: dict[str, Any] = {
                        "turn": 0,
                        "type": "turn",
                        "planner": True,
                        "latency_ms": planner_latency_ms,
                        "tokens_in": planner_usage.get("prompt_tokens"),
                        "tokens_out": planner_usage.get("completion_tokens"),
                        "content_preview": _truncate(plan_text, cap=512),
                        "plan": _truncate(plan_text, cap=_PLAN_CAP),
                        "tool_calls_requested": 0,
                    }
                    if planner_reasoning:
                        # Thinking-model lanes (gemma4 via ollama_chat) put
                        # the model's reasoning here; keep it as its own
                        # field, never merged into content/plan.
                        planner_entry["reasoning_content"] = _truncate(
                            planner_reasoning, cap=_PLAN_CAP
                        )
                    turns.append(planner_entry)
                    # ---- Phase B setup: inject the plan into the system
                    # prompt the executor (the unchanged react loop) sees.
                    plan_section = (
                        "\n\nA plan was prepared:\n"
                        f"{plan_text}\n"
                        "Follow it, adapting as results require."
                    )
                    if chat and chat[0].get("role") == "system":
                        chat[0]["content"] = str(chat[0].get("content") or "") + plan_section
                    else:
                        chat.insert(0, {"role": "system", "content": plan_section.lstrip()})
                    executor_start_turn = 1
                    log.info(
                        "planner_call_done",
                        model=model,
                        latency_ms=planner_latency_ms,
                        plan_chars=len(plan_text),
                    )
                if model_pool is not None:
                    with contextlib.suppress(Exception):
                        model_pool.step_complete("turn_0")

            for turn_idx in range(executor_start_turn, max_turns):
                turn_started = time.monotonic()
                if model_pool is not None:
                    with contextlib.suppress(Exception):
                        model_pool.step_start(f"turn_{turn_idx}")
                with span(
                    "agent_turn",
                    **{"lab.turn": turn_idx, "lab.model": model},
                ):
                    try:
                        with span("litellm_call", **{"lab.model": model}):
                            resp, latency_ms = call_litellm_chat(
                                settings=settings,
                                litellm_key=litellm_key,
                                model=model,
                                messages=_serialise_messages(chat),
                                temperature=temperature,
                                max_tokens=max_tokens,
                                tools=tool_specs or None,
                                extra=extra,
                            )
                            current_span_attrs(**{"lab.latency_ms": latency_ms})
                    except Exception as exc:
                        error = f"litellm call failed at turn {turn_idx}: {exc}"
                        terminated_reason = "litellm_error"
                        log.warning(
                            "agent_turn_litellm_failed",
                            turn=turn_idx,
                            model=model,
                            error=str(exc),
                        )
                        turns.append(
                            {
                                "turn": turn_idx,
                                "error": error,
                                "latency_ms": int((time.monotonic() - turn_started) * 1000),
                            }
                        )
                        break

                    actual_turns += 1
                    choice = (resp.get("choices") or [{}])[0]
                    assistant_msg = (choice or {}).get("message") or {}
                    content_text, reasoning_text = _extract_assistant_text(assistant_msg)
                    tool_calls = assistant_msg.get("tool_calls") or []
                    if not tool_calls and content_text:
                        # Fallback: some models emit tool calls as JSON text in
                        # content rather than structured calls (Llama-3.3, Qwen2.5-
                        # Coder). Recover them so they aren't silently dropped.
                        recovered = _extract_text_tool_calls(content_text, set(tool_modules))
                        if recovered:
                            tool_calls = recovered
                            current_span_attrs(
                                **{"lab.tool_calls_recovered_from_text": len(recovered)}
                            )
                    usage = resp.get("usage") or {}
                    current_span_attrs(
                        **{
                            "lab.tokens_in": usage.get("prompt_tokens"),
                            "lab.tokens_out": usage.get("completion_tokens"),
                            "lab.tool_calls_requested": len(tool_calls),
                        }
                    )

                    # Echo the assistant message into the running conversation
                    # (with normalised shape — strip provider-specific extras).
                    chat.append(
                        {
                            "role": "assistant",
                            "content": content_text,
                            "tool_calls": tool_calls or None,
                        }
                    )

                    turn_entry: dict[str, Any] = {
                        "turn": turn_idx,
                        "latency_ms": latency_ms,
                        "tokens_in": usage.get("prompt_tokens"),
                        "tokens_out": usage.get("completion_tokens"),
                        "content_preview": _truncate(content_text, cap=512),
                        "tool_calls_requested": len(tool_calls),
                    }
                    if reasoning_text:
                        # Thinking-model lanes (gemma4-12b via ollama_chat)
                        # return empty `content` with the model's thinking in
                        # `reasoning_content`; without this field those turns
                        # logged nothing despite thousands of tokens_out.
                        # Kept separate — never merged into content.
                        turn_entry["reasoning_content"] = _truncate(reasoning_text)

                    if not tool_calls:
                        turns.append(turn_entry)
                        terminated_reason = "model_finished"
                        log.info(
                            "agent_turn_finished",
                            turn=turn_idx,
                            terminated="model_finished",
                            latency_ms=latency_ms,
                        )
                        break

                    if remaining_budget <= 0:
                        turn_entry["budget_exhausted"] = True
                        turns.append(turn_entry)
                        terminated_reason = "budget_exhausted"
                        log.info(
                            "agent_turn_finished",
                            turn=turn_idx,
                            terminated="budget_exhausted",
                            latency_ms=latency_ms,
                        )
                        break

                    tool_msgs, tool_entries, n_called = _execute_tool_calls(
                        tool_calls=tool_calls,
                        pool=pool,
                        tool_modules=tool_modules,
                        remaining_budget=remaining_budget,
                        injector=injector,
                    )
                    tool_call_count += n_called
                    remaining_budget -= n_called
                    turn_entry["tool_calls"] = tool_entries
                    turns.append(turn_entry)

                    # Append tool results to the conversation for the next turn.
                    chat.extend(tool_msgs)
                    log.debug(
                        "agent_turn_done",
                        turn=turn_idx,
                        tool_calls=n_called,
                        latency_ms=latency_ms,
                    )

                    if remaining_budget <= 0 and turn_idx + 1 < max_turns:
                        # We delivered the results but won't take another tool
                        # turn — let the model see those results and write a
                        # final assistant message. Continue the loop.
                        pass
                # End of turn — fire predictive load for turn+1 (Phase 19c).
                if model_pool is not None:
                    with contextlib.suppress(Exception):
                        model_pool.step_complete(f"turn_{turn_idx}")
            else:
                # Guarded: a planner failure empties the loop range, and the
                # for-else fires on an empty range — don't let it overwrite
                # the planner's `litellm_error`.
                if error is None:
                    terminated_reason = "max_turns_reached"
        finally:
            # Snapshot any workspace files referenced by the task's
            # success_predicate BEFORE we tear down the pool / sandbox.
            # The sandbox is gone by the time scoring runs (it's a context
            # manager around the eval call), so the scorer can't read
            # `/workspace` itself — it must read from this snapshot.
            workspace_snapshot = _snapshot_predicate_files(task_meta, sandbox)
            if pool is not None:
                pool.stop()
            # Phase 19c — evict the cell's models so the next cell starts
            # with a clean VRAM slot. Errors are swallowed by ModelPool.
            if model_pool is not None:
                with contextlib.suppress(Exception):
                    model_pool.teardown()

        # Replace state.messages with the final conversation so Inspect
        # scorers and downstream tooling see what actually happened.
        state.messages = [_dict_to_chat_message(m) for m in chat]

        _stash_trajectory(
            state,
            error=error,
            turns=turns,
            actual_turns=actual_turns,
            tool_call_count=tool_call_count,
            total_latency_ms=int((time.monotonic() - run_started) * 1000),
            terminated_reason=terminated_reason,
            workspace_snapshot=workspace_snapshot,
            faults_fired=injector.fired_summary() if injector is not None else None,
            scaffold=scaffold,
        )
        state.completed = True
        return state

    return solve


def _chat_message_to_dict(msg: Any) -> dict[str, Any]:
    """Convert any of Inspect's `ChatMessage*` (or a plain dict) to OpenAI dict.

    Inspect sometimes hands us its pydantic message types; LiteLLM wants
    plain dicts. We keep the small set of fields the wire format actually
    cares about.
    """

    if isinstance(msg, dict):
        return msg
    role = getattr(msg, "role", None) or "user"
    content = getattr(msg, "content", "") or ""
    if not isinstance(content, str):
        # Inspect supports rich content lists; we flatten to text for the
        # proxy. Tool-use content is handled via `tool_calls` directly.
        try:
            content = "\n".join(getattr(c, "text", str(c)) for c in content)
        except Exception:
            content = str(content)
    out: dict[str, Any] = {"role": role, "content": content}
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = [_tool_call_to_openai(tc) for tc in tool_calls]
    tool_call_id = getattr(msg, "tool_call_id", None)
    if tool_call_id:
        out["tool_call_id"] = tool_call_id
    name = getattr(msg, "function", None) or getattr(msg, "name", None)
    if name:
        out["name"] = name
    return out


def _tool_call_to_openai(tc: Any) -> dict[str, Any]:
    if isinstance(tc, dict):
        return tc
    function = getattr(tc, "function", "")
    arguments = getattr(tc, "arguments", {})
    if isinstance(arguments, dict):
        arguments_str = json.dumps(arguments, default=str)
    else:
        arguments_str = str(arguments)
    return {
        "id": getattr(tc, "id", "") or f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {"name": function, "arguments": arguments_str},
    }


def _dict_to_chat_message(payload: dict[str, Any]) -> Any:
    """Inverse of `_chat_message_to_dict`. Returns an Inspect chat-message."""

    role = payload.get("role", "user")
    content = payload.get("content", "")
    if role == "user":
        return ChatMessageUser(content=content)
    if role == "assistant":
        # We don't try to round-trip tool_calls into Inspect's ToolCall
        # objects — the trajectory in metadata is the authoritative record.
        return ChatMessageAssistant(content=content)
    if role == "tool":
        return ChatMessageTool(
            content=content,
            tool_call_id=payload.get("tool_call_id", ""),
            function=payload.get("name", ""),
        )
    if role == "system":
        # Phase 16.4: the adapter now injects ChatMessageSystem when a task
        # references a prompt via system_prompt_id (or an inline `system`
        # field). Preserve the role on the round-trip so post-hoc readers
        # of state.messages see the system prompt as a system message,
        # not as a fallback user message.
        return ChatMessageSystem(content=content)
    return ChatMessageUser(content=content)


def _stash_trajectory(
    state: TaskState,
    *,
    error: str | None,
    turns: list[dict[str, Any]],
    actual_turns: int,
    tool_call_count: int,
    total_latency_ms: int,
    terminated_reason: str = "unknown",
    workspace_snapshot: dict[str, bytes | None] | None = None,
    faults_fired: list[dict[str, Any]] | None = None,
    scaffold: str = "react",
) -> None:
    """Stash the agent trajectory on `state.metadata` for the logwriter."""

    if state.metadata is None:
        state.metadata = {}
    # The adapter stamps `lab_prompt_id_used` on Sample.metadata when a
    # task referenced a prompt by id. Pull it through into the trajectory
    # record so the logwriter (and human readers) can see which prompt
    # body actually ran. Stays `None` for legacy `task.system` rows.
    prompt_id_used = state.metadata.get("lab_prompt_id_used")
    state.metadata["lab_agent"] = {
        "error": error,
        "turns": turns,
        "actual_turns": actual_turns,
        "tool_call_count": tool_call_count,
        "total_latency_ms": total_latency_ms,
        "terminated_reason": terminated_reason,
        "workspace_snapshot": workspace_snapshot or {},
        "prompt_id_used": prompt_id_used,
        "faults_fired": faults_fired or [],
        # Which scaffold actually ran ("react" / "plan_execute") — lets
        # post-hoc audits verify the A/B arms without re-reading configs.
        "scaffold": scaffold,
    }


_WORKSPACE_PREDICATE_TYPES = frozenset(
    {
        "workspace_file_exists",
        "workspace_file_equals",
        "workspace_file_contains",
    }
)


def _collect_predicate_paths(predicate: dict[str, Any]) -> list[str]:
    """Walk a (possibly composite) predicate and return all workspace paths.

    Handles `workspace_file_*` directly and the `all_of` composite by
    recursing into each sub-predicate. Non-workspace types (`db_query`,
    `retrieval_recall`, etc.) contribute no paths. Unknown sub-types are
    silently skipped — the scorer reports the mismatch.
    """

    paths: list[str] = []
    ptype = predicate.get("type")
    if ptype in _WORKSPACE_PREDICATE_TYPES:
        path = predicate.get("path")
        if isinstance(path, str) and path:
            paths.append(path)
        return paths
    if ptype == "all_of":
        subs = predicate.get("predicates")
        if isinstance(subs, list):
            for sub in subs:
                if isinstance(sub, dict):
                    paths.extend(_collect_predicate_paths(sub))
    return paths


def _snapshot_predicate_files(task_meta: Any, sandbox: Sandbox | None) -> dict[str, bytes | None]:
    """Read any files referenced by `task.success_predicate` out of the sandbox.

    Returns `{path: bytes_or_None}` — empty dict if the task has no
    predicate or the sandbox is gone. Failures are non-fatal: a missing
    file shows up as `None` and the scorer reports the absence; an
    exception during snapshotting (e.g. sandbox already stopped) is
    swallowed and the affected entry is `None`.

    Only paths referenced by `workspace_file_*` sub-predicates are read;
    `db_query` predicates run at scoring time and don't need a snapshot.
    Composite `all_of` predicates are walked recursively so every nested
    workspace path is captured before sandbox teardown.
    """

    if sandbox is None or task_meta is None:
        return {}
    predicate = getattr(task_meta, "success_predicate", None)
    if not predicate or not isinstance(predicate, dict):
        return {}
    paths = _collect_predicate_paths(predicate)
    if not paths:
        return {}
    snapshot: dict[str, bytes | None] = {}
    for path in paths:
        try:
            snapshot[path] = sandbox.read_workspace_file(path)
        except Exception:
            snapshot[path] = None
    return snapshot


__all__ = ["FaultInjector", "model_with_tools"]
