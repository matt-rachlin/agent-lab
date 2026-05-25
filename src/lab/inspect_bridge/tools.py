"""Inspect-side wrappers for the lab MCP tool servers.

The agent solver (6d) only sees Inspect `Tool` callables. This module is the
seam: it turns the FastMCP stdio servers under `lab.agent.tools` into
Inspect tools that, when invoked, marshal the call through `podman exec` into
the sandbox, run the MCP server there for the duration of one call, and
return the parsed result.

Per-call subprocess spawn is the v0.1 default — startup cost is ~150-250 ms
under gVisor, which is acceptable for single-digit-call tasks but will need
to be pooled for multi-turn agents (flagged for 6d).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
from typing import Any

from inspect_ai.tool import Tool, ToolDef, ToolError, ToolParams, tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from lab.agent.sandbox import Sandbox
from lab.agent.tools import TOOL_SERVERS
from lab.tasks.registry import Task


class ToolSchema:
    """Cached MCP tool schema (name, description, JSON-schema parameters).

    Populated once by `discover_tool_schemas()` and reused for both the CLI
    `lab agent tools list` output and the Inspect tool registration.
    """

    __slots__ = ("description", "input_schema", "name")

    def __init__(self, name: str, description: str, input_schema: dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema


async def _list_tools_for_module(module: str) -> list[ToolSchema]:
    """Spawn `python -m <module>` locally and ask it for its MCP tool list.

    Used by the CLI and the Inspect bridge to learn each tool's schema
    without hand-duplicating it. The subprocess runs on the host (not the
    sandbox) because we just want the schema, not to execute any side
    effects.
    """

    params = StdioServerParameters(command="python", args=["-m", module])
    schemas: list[ToolSchema] = []
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        listing = await session.list_tools()
        for entry in listing.tools:
            schemas.append(
                ToolSchema(
                    name=entry.name,
                    description=entry.description or "",
                    input_schema=dict(entry.inputSchema),
                )
            )
    return schemas


def discover_tool_schemas() -> dict[str, ToolSchema]:
    """Return `{tool_name: ToolSchema}` for every server in `TOOL_SERVERS`.

    Synchronous wrapper so callers (CLI, factories) don't have to manage an
    event loop themselves.
    """

    out: dict[str, ToolSchema] = {}
    for tool_name, module in TOOL_SERVERS.items():
        try:
            schemas = asyncio.run(_list_tools_for_module(module))
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"failed to introspect {module}: {exc}") from exc
        for schema in schemas:
            # Each server should expose exactly one tool whose name matches
            # the module key. We surface a clean error otherwise so the
            # invariant is loud.
            if schema.name != tool_name:
                raise RuntimeError(
                    f"tool server {module} advertised tool {schema.name!r}, expected {tool_name!r}"
                )
            out[tool_name] = schema
    return out


def _json_schema_to_tool_params(schema: dict[str, Any]) -> ToolParams:
    """Map a FastMCP JSON-schema object to Inspect `ToolParams`.

    FastMCP emits `{type: object, properties: {...}, required: [...]}`. Inspect
    accepts the same shape natively but via its own pydantic model.
    """

    return ToolParams.model_validate(schema)


def _invoke_tool_via_sandbox_sync(
    sandbox: Sandbox,
    module: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> Any:
    """Spawn an MCP server inside the sandbox and run one tool call.

    Blocking helper used by the async tool body. We use `podman exec` directly
    (not `Sandbox.exec`) because we need duplex stdio for the MCP framing.
    """

    argv = [
        "podman",
        "exec",
        "--interactive",
        sandbox.container_name,
        "python3",
        "-m",
        module,
    ]
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        return _drive_mcp_session_sync(proc, tool_name, arguments)
    finally:
        with contextlib.suppress(Exception):
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _drive_mcp_session_sync(
    proc: subprocess.Popen[bytes],
    tool_name: str,
    arguments: dict[str, Any],
) -> Any:
    """Drive a freshly-spawned MCP stdio server through one tool call.

    We speak the JSON-RPC framing manually (one JSON message per line) rather
    than re-using `stdio_client`, which would shell out to spawn the server
    again — we've already done that via `podman exec`.
    """

    assert proc.stdin is not None
    assert proc.stdout is not None

    def _write(payload: dict[str, Any]) -> None:
        assert proc.stdin is not None
        data = (json.dumps(payload) + "\n").encode("utf-8")
        proc.stdin.write(data)
        proc.stdin.flush()

    def _read() -> dict[str, Any]:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        if not line:
            stderr = b""
            if proc.stderr is not None:
                stderr = proc.stderr.read() or b""
            raise ToolError(
                f"MCP server closed stdout unexpectedly: {stderr.decode(errors='replace')}"
            )
        return json.loads(line.decode("utf-8"))  # type: ignore[no-any-return]

    # 1. initialize
    _write(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "lab-inspect-bridge", "version": "0.1"},
            },
        }
    )
    _read()
    _write({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    # 2. tools/call
    _write(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
    )
    response = _read()
    if "error" in response:
        raise ToolError(str(response["error"]))
    result = response.get("result", {})
    if result.get("isError"):
        text_chunks = [
            c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"
        ]
        raise ToolError("; ".join(text_chunks) or "tool execution failed")
    if "structuredContent" in result:
        return result["structuredContent"]
    text_chunks = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
    joined = "\n".join(text_chunks)
    try:
        return json.loads(joined)
    except json.JSONDecodeError:
        return joined


def _make_inspect_tool(
    *,
    sandbox: Sandbox,
    schema: ToolSchema,
    module: str,
) -> Tool:
    """Build one Inspect `Tool` for a given MCP tool name."""

    tool_name = schema.name
    description = schema.description
    parameters = _json_schema_to_tool_params(schema.input_schema)

    @tool(name=tool_name)
    def _factory() -> Tool:
        async def execute(**kwargs: Any) -> Any:
            return await asyncio.to_thread(
                _invoke_tool_via_sandbox_sync,
                sandbox,
                module,
                tool_name,
                kwargs,
            )

        return ToolDef(
            tool=execute,
            name=tool_name,
            description=description,
            parameters=parameters,
        ).as_tool()

    return _factory()


def lab_tools_for_task(task: Task, sandbox: Sandbox) -> list[Tool]:
    """Return Inspect tools for the names listed in `task.tools`.

    `task.tools` is a list of `{name: str, ...}` dicts. We only honour the
    `name` field — the schema comes from the MCP server itself, not the YAML
    — and we filter to known tools. Unknown names raise so a typo doesn't
    silently fall off the floor.
    """

    if not task.tools:
        return []
    schemas = discover_tool_schemas()
    out: list[Tool] = []
    for spec in task.tools:
        name = spec.get("name") if isinstance(spec, dict) else None
        if not name:
            continue
        if name not in TOOL_SERVERS:
            raise ValueError(
                f"task {task.slug!r} references unknown tool {name!r}; "
                f"known tools: {sorted(TOOL_SERVERS)}"
            )
        schema = schemas[name]
        out.append(_make_inspect_tool(sandbox=sandbox, schema=schema, module=TOOL_SERVERS[name]))
    return out


__all__ = [
    "ToolSchema",
    "discover_tool_schemas",
    "lab_tools_for_task",
]
