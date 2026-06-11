"""Harbor BaseAgent adapter — runs the lab's ReAct scaffold on Terminal-Bench.

This is a faithful, self-contained PORT of the lab's agent loop
(``packages/lab-inspect/src/lab/inspect_bridge/solver.py``) onto Harbor's
executor. Same decision layer: v2 system prompt (act-don't-narrate), the
assistant -> tool dispatch -> tool result loop, the text-tool-call fallback
parser, and per-turn trajectory instrumentation. The tool surface is the task
container via ``environment.exec`` instead of MCP sandbox tools.

Dependencies: stdlib + litellm only — both available inside Harbor's uv-tool
venv, so the lab package does NOT need to be installed there. The canonical
copy lives in the lab repo (``lab.agent.harbor_adapter``); a symlink at
``/data/lab/harbor-agents/lab_react_agent.py`` exposes it to Harbor:

    export OPENAI_API_KEY="$(cat /data/lab/services/litellm-master-key)"
    PYTHONPATH=/data/lab/harbor-agents harbor run \
        --agent-import-path lab_react_agent:LabReactAgent \
        --model openai/glm-5.1-cloud --ak api_base=http://localhost:4000/v1 ...
"""
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
# harbor runs this file inside its own uv-tool venv; the lab workspace
# doesn't depend on harbor, so its imports are unresolvable to the gates.

from __future__ import annotations

import base64
import json
import logging
import os
import posixpath
import re
import shlex
import time
from pathlib import Path
from typing import Any

import litellm
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig

# Truncation budget for tool call inputs/outputs recorded in the trajectory
# (same cap as the lab solver's `_TURN_PAYLOAD_CAP`).
_TURN_PAYLOAD_CAP = 4096

# Caps on tool output fed back to the model.
_STREAM_CAP = 8192  # per stream (stdout / stderr) for bash
_READ_CAP = 65536  # read_file content cap (bytes)
_WRITE_CAP = 512 * 1024  # write_file content cap (bytes)

_DEFAULT_BASH_TIMEOUT_SEC = 120
_MAX_BASH_TIMEOUT_SEC = 600

# Port of prompts/library/tool-use-system-v2.md — exact text, tool list
# adapted to the Harbor surface (bash / write_file / read_file).
_SYSTEM_PROMPT = """\
You are an assistant with tool access. Always call the appropriate tool
when asked to read, write, fetch, or compute — never guess. Use the
EXACT tool names provided. Run shell commands with bash; write files
with write_file; read files with read_file.

CRITICAL: act only via tool calls. Never describe or plan in text, and
never write code blocks showing what you would run — actually invoke
the tool. A reply without a tool call ends the session, so keep calling
tools until the task is fully complete."""

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a shell command in the task container. Returns stdout, stderr and return_code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run."},
                    "timeout_sec": {
                        "type": "integer",
                        "description": "Kill the command after this many seconds "
                        f"(default {_DEFAULT_BASH_TIMEOUT_SEC}, max {_MAX_BASH_TIMEOUT_SEC}).",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (overwrites; creates parent dirs).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path."},
                    "content": {"type": "string", "description": "Full file content."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file (first 64KB).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path."},
                },
                "required": ["path"],
            },
        },
    },
]

_TOOL_NAMES = {spec["function"]["name"] for spec in _TOOLS}


def _truncate(value: Any, cap: int = _TURN_PAYLOAD_CAP) -> Any:
    """Bound large tool I/O in the recorded trajectory (port of solver._truncate)."""
    try:
        text = json.dumps(value, default=str)
    except Exception:
        text = str(value)
    if len(text) <= cap:
        return value
    return {"_truncated": True, "preview": text[:cap], "original_size": len(text)}


def _cap_stream(text: str | None, cap: int = _STREAM_CAP) -> str:
    if not text:
        return ""
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n... [truncated {len(text) - cap} chars]"


def _coerce_arguments(raw: Any) -> dict[str, Any]:
    """Tool call `arguments` arrives as a JSON string from OpenAI-compat servers.

    Port of solver._coerce_arguments — accept pre-parsed dicts too; raise on
    anything else so the failure surfaces as a per-call error.
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


def _extract_text_tool_calls(content: str, valid_names: set[str]) -> list[dict[str, Any]]:
    """Recover tool calls emitted as JSON *text* instead of structured tool_calls.

    Direct port of solver._extract_text_tool_calls (same regex, same shapes):
    ``{"name": "bash", "arguments": {...}}`` or
    ``{"type": "function", "name": ..., "parameters": {...}}``.
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


def _tool_call_to_dict(tc: Any) -> dict[str, Any]:
    """Normalise a litellm tool-call object (or dict) to the OpenAI wire dict."""
    if isinstance(tc, dict):
        fn = tc.get("function") or {}
        return {
            "id": tc.get("id") or "",
            "type": "function",
            "function": {
                "name": fn.get("name") or "",
                "arguments": fn.get("arguments") or "{}",
            },
        }
    fn = getattr(tc, "function", None)
    return {
        "id": getattr(tc, "id", "") or "",
        "type": "function",
        "function": {
            "name": getattr(fn, "name", "") or "",
            "arguments": getattr(fn, "arguments", "{}") or "{}",
        },
    }


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON record (sync helper; called per turn so partial runs persist)."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


class LabReactAgent(BaseAgent):  # type: ignore[misc]
    """The lab's ReAct scaffold as a Harbor agent (decision layer port)."""

    SUPPORTS_ATIF = False

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        logger: logging.Logger | None = None,
        mcp_servers: list[MCPServerConfig] | None = None,
        skills_dir: str | None = None,
        *args: Any,
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        max_turns: int = 40,
        tool_budget: int = 60,
        **kwargs: Any,
    ):
        super().__init__(logs_dir, model_name, logger, mcp_servers, skills_dir, *args, **kwargs)
        self._api_base = api_base or os.environ.get("OPENAI_BASE_URL")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._max_turns = int(max_turns)
        self._tool_budget = int(tool_budget)

    @staticmethod
    def name() -> str:
        return "lab-react"

    def version(self) -> str | None:
        return "0.1"

    async def setup(self, environment: BaseEnvironment) -> None:
        """No agent install needed — the loop runs host-side via environment.exec."""
        return

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        run_started = time.monotonic()
        traj_path = Path(self.logs_dir) / "trajectory.jsonl"
        chat: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]

        tokens_in = 0
        tokens_out = 0
        cost_usd = 0.0
        remaining_budget = self._tool_budget
        tool_call_count = 0
        actual_turns = 0
        text_fallback_turns: list[int] = []
        terminated_reason = "max_turns_reached"
        error: str | None = None

        def _sync_context() -> None:
            context.n_input_tokens = tokens_in or None
            context.n_output_tokens = tokens_out or None
            context.cost_usd = cost_usd if cost_usd > 0 else None
            context.metadata = {
                "agent": self.name(),
                "version": self.version(),
                "actual_turns": actual_turns,
                "tool_call_count": tool_call_count,
                "terminated_reason": terminated_reason,
                "error": error,
                "text_fallback_turns": text_fallback_turns,
                "trajectory_file": traj_path.name,
            }

        _sync_context()

        for turn_idx in range(self._max_turns):
            turn_started = time.monotonic()
            try:
                resp = await litellm.acompletion(
                    model=self.model_name,
                    messages=chat,
                    tools=_TOOLS,
                    api_base=self._api_base,
                    api_key=self._api_key,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    timeout=600,
                )
            except Exception as exc:
                error = f"llm call failed at turn {turn_idx}: {exc}"
                terminated_reason = "llm_error"
                _append_jsonl(
                    traj_path,
                    {
                        "turn": turn_idx,
                        "error": error,
                        "latency_ms": int((time.monotonic() - turn_started) * 1000),
                    },
                )
                break
            latency_ms = int((time.monotonic() - turn_started) * 1000)
            actual_turns += 1

            usage = getattr(resp, "usage", None)
            tokens_in += getattr(usage, "prompt_tokens", 0) or 0
            tokens_out += getattr(usage, "completion_tokens", 0) or 0
            hidden = getattr(resp, "_hidden_params", None) or {}
            cost_usd += hidden.get("response_cost") or 0.0

            message = resp.choices[0].message
            content_text = message.content or ""
            tool_calls = [_tool_call_to_dict(tc) for tc in (message.tool_calls or [])]
            recovered_from_text = False
            if not tool_calls and content_text:
                # Fallback: some models emit tool calls as JSON text in content
                # rather than structured calls. Recover them so they aren't
                # silently dropped (port of the lab solver's fallback).
                recovered = _extract_text_tool_calls(content_text, _TOOL_NAMES)
                if recovered:
                    tool_calls = recovered
                    recovered_from_text = True
                    text_fallback_turns.append(turn_idx)

            # Echo the assistant message into the running conversation. Recovered
            # text-calls are echoed as structured tool_calls too (solver parity;
            # also keeps the following tool-role messages protocol-valid).
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": content_text}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            chat.append(assistant_msg)

            turn_entry: dict[str, Any] = {
                "turn": turn_idx,
                "latency_ms": latency_ms,
                "tokens_in": getattr(usage, "prompt_tokens", None),
                "tokens_out": getattr(usage, "completion_tokens", None),
                "content_preview": _truncate(content_text, cap=512),
                "tool_calls_requested": len(tool_calls),
                "recovered_from_text": recovered_from_text,
            }

            if not tool_calls:
                _append_jsonl(traj_path, turn_entry)
                terminated_reason = "model_finished"
                _sync_context()
                break

            if remaining_budget <= 0:
                turn_entry["budget_exhausted"] = True
                _append_jsonl(traj_path, turn_entry)
                terminated_reason = "budget_exhausted"
                _sync_context()
                break

            tool_msgs, tool_entries, n_called = await self._execute_tool_calls(
                environment=environment,
                tool_calls=tool_calls,
                remaining_budget=remaining_budget,
            )
            tool_call_count += n_called
            remaining_budget -= n_called
            turn_entry["tool_calls"] = tool_entries
            _append_jsonl(traj_path, turn_entry)
            chat.extend(tool_msgs)
            _sync_context()

        _sync_context()
        _append_jsonl(
            traj_path,
            {
                "summary": {
                    "model": self.model_name,
                    "terminated_reason": terminated_reason,
                    "error": error,
                    "actual_turns": actual_turns,
                    "tool_call_count": tool_call_count,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "cost_usd": cost_usd if cost_usd > 0 else None,
                    "text_fallback_turns": text_fallback_turns,
                    "total_latency_ms": int((time.monotonic() - run_started) * 1000),
                }
            },
        )

    async def _execute_tool_calls(
        self,
        *,
        environment: BaseEnvironment,
        tool_calls: list[dict[str, Any]],
        remaining_budget: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        """Run requested tool calls; return (tool msgs, trajectory entries, calls done).

        Port of solver._execute_tool_calls: budget decremented for every call
        actually dispatched (failed or successful); calls beyond the budget get
        a "tool budget exhausted" result instead of executing.
        """
        new_messages: list[dict[str, Any]] = []
        entries: list[dict[str, Any]] = []
        calls_done = 0
        for call in tool_calls:
            fn = call.get("function", {}) or {}
            name = fn.get("name") or ""
            call_id = call.get("id", "")
            if remaining_budget - calls_done <= 0:
                entries.append({"tool": name, "skipped": "budget_exhausted"})
                new_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": "tool budget exhausted",
                    }
                )
                continue

            t0 = time.monotonic()
            try:
                args = _coerce_arguments(fn.get("arguments", "{}"))
                result = await self._dispatch_tool(environment, name, args)
                tool_error: str | None = None
            except Exception as exc:
                raw_args = fn.get("arguments")
                args = raw_args if isinstance(raw_args, dict) else {}
                result = {"error": f"{type(exc).__name__}: {exc}"}
                tool_error = str(exc)
            latency_ms = int((time.monotonic() - t0) * 1000)
            calls_done += 1

            content = result if isinstance(result, str) else json.dumps(result, default=str)
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content,
                }
            )
            entries.append(
                {
                    "tool": name,
                    "args": _truncate(args),
                    "result": _truncate(result),
                    "latency_ms": latency_ms,
                    "error": tool_error,
                }
            )
        return new_messages, entries, calls_done

    async def _dispatch_tool(
        self,
        environment: BaseEnvironment,
        name: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        if name == "bash":
            return await self._tool_bash(environment, args)
        if name == "write_file":
            return await self._tool_write_file(environment, args)
        if name == "read_file":
            return await self._tool_read_file(environment, args)
        raise ValueError(f"unknown tool {name!r}")

    async def _tool_bash(
        self, environment: BaseEnvironment, args: dict[str, Any]
    ) -> dict[str, Any]:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("bash requires a non-empty 'command' string")
        try:
            timeout_sec = int(args.get("timeout_sec") or _DEFAULT_BASH_TIMEOUT_SEC)
        except (TypeError, ValueError):
            timeout_sec = _DEFAULT_BASH_TIMEOUT_SEC
        timeout_sec = max(1, min(timeout_sec, _MAX_BASH_TIMEOUT_SEC))
        result = await environment.exec(command, timeout_sec=timeout_sec)
        return {
            "stdout": _cap_stream(result.stdout),
            "stderr": _cap_stream(result.stderr),
            "return_code": result.return_code,
        }

    async def _tool_write_file(
        self, environment: BaseEnvironment, args: dict[str, Any]
    ) -> dict[str, Any]:
        path = args.get("path")
        content = args.get("content")
        if not isinstance(path, str) or not path:
            raise ValueError("write_file requires a 'path' string")
        if not isinstance(content, str):
            raise ValueError("write_file requires a 'content' string")
        data = content.encode("utf-8")
        if len(data) > _WRITE_CAP:
            raise ValueError(f"content too large ({len(data)} bytes > {_WRITE_CAP})")
        # base64 round-trip avoids every shell-quoting pitfall in the payload.
        b64 = base64.b64encode(data).decode("ascii")
        parent = posixpath.dirname(path)
        mkdir = f"mkdir -p {shlex.quote(parent)} && " if parent else ""
        command = f"{mkdir}printf %s {shlex.quote(b64)} | base64 -d > {shlex.quote(path)}"
        result = await environment.exec(command, timeout_sec=60)
        if result.return_code != 0:
            return {
                "ok": False,
                "return_code": result.return_code,
                "stderr": _cap_stream(result.stderr),
            }
        return {"ok": True, "path": path, "bytes_written": len(data)}

    async def _tool_read_file(
        self, environment: BaseEnvironment, args: dict[str, Any]
    ) -> dict[str, Any]:
        path = args.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("read_file requires a 'path' string")
        command = f"head -c {_READ_CAP + 1} {shlex.quote(path)}"
        result = await environment.exec(command, timeout_sec=60)
        if result.return_code != 0:
            return {
                "error": _cap_stream(result.stderr) or f"read failed (rc={result.return_code})",
            }
        data = result.stdout or ""
        truncated = len(data) > _READ_CAP
        return {"content": data[:_READ_CAP], "truncated": truncated}


__all__ = ["LabReactAgent"]
