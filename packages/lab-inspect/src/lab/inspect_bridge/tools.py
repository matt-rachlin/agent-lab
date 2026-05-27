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
import shutil
import subprocess
import sys
from typing import Any

from inspect_ai.tool import Tool, ToolDef, ToolError, ToolParams, tool
from lab.agent.sandbox import Sandbox
from lab.agent.tool_pool import ToolPool
from lab.agent.tools import TOOL_SERVERS
from lab.tasks.registry import Task
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


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


def _host_python() -> str:
    """Resolve the Python interpreter to use for host-side tool introspection.

    Prefer the running interpreter (`sys.executable`) so callers don't need an
    activated venv. Fall back to `python3` then `python` on PATH if
    `sys.executable` is unavailable (rare; embedded launchers etc.).
    """

    if sys.executable:
        return sys.executable
    for candidate in ("python3", "python"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return "python3"


async def _list_tools_for_module(module: str) -> list[ToolSchema]:
    """Spawn `<host-python> -m <module>` locally and ask it for its MCP tool list.

    Used by the CLI and the Inspect bridge to learn each tool's schema
    without hand-duplicating it. The subprocess runs on the host (not the
    sandbox) because we just want the schema, not to execute any side
    effects. Uses `sys.executable` so the bridge works outside an activated
    venv — flagged in 6e and fixed in 6f.
    """

    params = StdioServerParameters(command=_host_python(), args=["-m", module])
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


def _run_coro_sync(coro: Any) -> Any:
    """Run an awaitable to completion regardless of caller event-loop state.

    If we're already inside a running loop (e.g. the Inspect solver),
    `asyncio.run()` raises; spin up a dedicated thread with its own loop.
    """
    import concurrent.futures

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exe:
        return exe.submit(asyncio.run, coro).result()


def discover_tool_schemas() -> dict[str, ToolSchema]:
    """Return `{tool_name: ToolSchema}` for every server in `TOOL_SERVERS`.

    Synchronous wrapper so callers (CLI, factories) don't have to manage an
    event loop themselves. Safe to call from inside a running loop too.
    """

    out: dict[str, ToolSchema] = {}
    for tool_name, module in TOOL_SERVERS.items():
        try:
            schemas = _run_coro_sync(_list_tools_for_module(module))
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
    accepts the same shape natively but requires a non-empty `description` on
    every property; FastMCP's pydantic-derived schemas omit that. We inject
    the property's `title` (or name) as a fallback so validation passes.
    """

    patched = dict(schema)
    props = dict(patched.get("properties", {}) or {})
    for prop_name, prop_schema in list(props.items()):
        if not isinstance(prop_schema, dict):
            continue
        if not prop_schema.get("description"):
            prop = dict(prop_schema)
            prop["description"] = str(prop.get("title") or prop_name)
            props[prop_name] = prop
    if props:
        patched["properties"] = props
    return ToolParams.model_validate(patched)


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
    pool: ToolPool | None = None,
) -> Tool:
    """Build one Inspect `Tool` for a given MCP tool name.

    When `pool` is provided, calls reuse the pooled long-lived MCP server
    inside the sandbox. Without a pool, each call spawns a fresh subprocess
    — fine for unit tests and the `lab agent tools test` smoke path; far
    too expensive for multi-turn agents (which should always pass a pool).
    """

    tool_name = schema.name
    description = schema.description
    parameters = _json_schema_to_tool_params(schema.input_schema)

    @tool(name=tool_name)
    def _factory() -> Tool:
        async def execute(**kwargs: Any) -> Any:
            if pool is not None:
                # The pool's underlying stdio I/O is blocking; punt it off
                # the event loop so we don't stall the Inspect runner.
                return await asyncio.to_thread(pool.invoke, module, tool_name, kwargs)
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


def lab_tools_for_task(
    task: Task,
    sandbox: Sandbox,
    *,
    pool: ToolPool | None = None,
    tool_names: list[str] | None = None,
) -> list[Tool]:
    """Return Inspect tools for the names listed in `task.tools`.

    `task.tools` is a list of `{name: str, ...}` dicts. We only honour the
    `name` field — the schema comes from the MCP server itself, not the YAML
    — and we filter to known tools. Unknown names raise so a typo doesn't
    silently fall off the floor.

    When `pool` is provided, the returned tools route through the pool
    (one long-lived MCP server per (sandbox, module)) instead of spawning a
    fresh subprocess per call. The solver should always pass a pool.

    `tool_names`, when given, further restricts the returned tools to that
    list — useful when a sweep wants to disable a tool the task technically
    allows.
    """

    if not task.tools:
        return []
    schemas = discover_tool_schemas()
    allow: set[str] | None = set(tool_names) if tool_names is not None else None
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
        if allow is not None and name not in allow:
            continue
        schema = schemas[name]
        out.append(
            _make_inspect_tool(
                sandbox=sandbox,
                schema=schema,
                module=TOOL_SERVERS[name],
                pool=pool,
            )
        )
    return out


__all__ = [
    "ToolSchema",
    "discover_tool_schemas",
    "lab_tools_for_task",
]
