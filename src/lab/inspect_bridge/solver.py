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

import json
import time
import uuid
from typing import Any

from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool

from lab.agent.sandbox import Sandbox
from lab.agent.tool_pool import ToolPool
from lab.inspect_bridge.tools import discover_tool_schemas
from lab.llm import call_litellm_chat
from lab.settings import get_settings

# Truncation budget for tool call inputs/outputs recorded in `turns`.
_TURN_PAYLOAD_CAP = 4096


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
            if kk in {
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


def _execute_tool_calls(
    *,
    tool_calls: list[dict[str, Any]],
    pool: ToolPool | None,
    tool_modules: dict[str, str],
    remaining_budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Run each requested tool call; return (new chat msgs, turn entries, calls done).

    `tool_modules` maps tool name → dotted MCP server module. `remaining_budget`
    is decremented for every call we actually dispatch (failed or successful);
    if we hit zero mid-batch we return early so the caller can break the loop.
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
        try:
            args = _coerce_arguments(fn.get("arguments", "{}"))
            if name not in tool_modules:
                raise ValueError(f"unknown tool {name!r}")
            if pool is None:
                raise RuntimeError(
                    "tool call received but no pool configured (sandbox missing)"
                )
            result = pool.invoke(tool_modules[name], name, args)
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
        entries.append(
            {
                "tool": name,
                "args": _truncate(args),
                "result": _truncate(result),
                "latency_ms": latency_ms,
                "error": error,
            }
        )
    return new_messages, entries, calls_done


@solver(name="lab_model_with_tools")
def model_with_tools(
    *,
    model: str,
    tool_budget: int = 10,
    max_turns: int = 5,
    sandbox: Sandbox | None = None,
    tool_names: list[str] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    extra: dict[str, Any] | None = None,
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

    Returns:
        An Inspect Solver that mutates `state.messages` and stashes a per-
        turn record at `state.metadata["lab_agent"]`.
    """

    if max_turns < 1:
        raise ValueError("max_turns must be >= 1")
    if tool_budget < 0:
        raise ValueError("tool_budget must be >= 0")

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

        pool: ToolPool | None = None
        tools: list[Tool] = []
        if sandbox is not None and task_meta is not None and tool_budget > 0:
            pool = ToolPool(sandbox)
            try:
                tools = lab_tools_for_task(
                    task_meta, sandbox, pool=pool, tool_names=tool_names
                )
            except Exception as exc:
                # Surface tool-discovery failures as a partial trajectory.
                _stash_trajectory(
                    state,
                    error=f"tool discovery failed: {exc}",
                    turns=[],
                    actual_turns=0,
                    tool_call_count=0,
                    total_latency_ms=int((time.monotonic() - run_started) * 1000),
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

        try:
            for turn_idx in range(max_turns):
                turn_started = time.monotonic()
                try:
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
                except Exception as exc:
                    error = f"litellm call failed at turn {turn_idx}: {exc}"
                    terminated_reason = "litellm_error"
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
                content_text = assistant_msg.get("content") or ""
                tool_calls = assistant_msg.get("tool_calls") or []
                usage = resp.get("usage") or {}

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

                if not tool_calls:
                    turns.append(turn_entry)
                    terminated_reason = "model_finished"
                    break

                if remaining_budget <= 0:
                    turn_entry["budget_exhausted"] = True
                    turns.append(turn_entry)
                    terminated_reason = "budget_exhausted"
                    break

                tool_msgs, tool_entries, n_called = _execute_tool_calls(
                    tool_calls=tool_calls,
                    pool=pool,
                    tool_modules=tool_modules,
                    remaining_budget=remaining_budget,
                )
                tool_call_count += n_called
                remaining_budget -= n_called
                turn_entry["tool_calls"] = tool_entries
                turns.append(turn_entry)

                # Append tool results to the conversation for the next turn.
                chat.extend(tool_msgs)

                if remaining_budget <= 0 and turn_idx + 1 < max_turns:
                    # We delivered the results but won't take another tool
                    # turn — let the model see those results and write a
                    # final assistant message. Continue the loop.
                    pass
            else:
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
    # System messages are kept as ChatMessageUser to avoid having to import
    # ChatMessageSystem just for the fallback; they're already in chat from
    # the initial state, not re-added by the loop.
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
) -> None:
    """Stash the agent trajectory on `state.metadata` for the logwriter."""

    if state.metadata is None:
        state.metadata = {}
    state.metadata["lab_agent"] = {
        "error": error,
        "turns": turns,
        "actual_turns": actual_turns,
        "tool_call_count": tool_call_count,
        "total_latency_ms": total_latency_ms,
        "terminated_reason": terminated_reason,
        "workspace_snapshot": workspace_snapshot or {},
    }


def _snapshot_predicate_files(
    task_meta: Any, sandbox: Sandbox | None
) -> dict[str, bytes | None]:
    """Read any files referenced by `task.success_predicate` out of the sandbox.

    Returns `{path: bytes_or_None}` — empty dict if the task has no
    predicate or the sandbox is gone. Failures are non-fatal: a missing
    file shows up as `None` and the scorer reports the absence; an
    exception during snapshotting (e.g. sandbox already stopped) is
    swallowed and the affected entry is `None`.

    Only paths referenced by `workspace_file_*` predicates are read;
    `db_query` predicates run at scoring time and don't need a snapshot.
    """

    if sandbox is None or task_meta is None:
        return {}
    predicate = getattr(task_meta, "success_predicate", None)
    if not predicate or not isinstance(predicate, dict):
        return {}
    ptype = predicate.get("type")
    if ptype not in {
        "workspace_file_exists",
        "workspace_file_equals",
        "workspace_file_contains",
    }:
        return {}
    path = predicate.get("path")
    if not isinstance(path, str) or not path:
        return {}
    snapshot: dict[str, bytes | None] = {}
    try:
        snapshot[path] = sandbox.read_workspace_file(path)
    except Exception:
        snapshot[path] = None
    return snapshot


__all__ = ["model_with_tools"]
