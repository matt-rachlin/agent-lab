"""Hand-built FastMCP tool servers for the lab agent harness.

Each tool is a standalone module exposing a single `FastMCP` server with
exactly one tool. Modules are spawned as subprocesses (either on the host for
unit tests, or inside the sandbox via `podman exec`) and communicate over MCP
stdio.

`TOOL_SERVERS` maps the canonical tool name to the dotted module path the
solver should launch with `python -m <module>`. Keep this dict authoritative —
the Inspect bridge and CLI both reflect it.
"""

from __future__ import annotations

TOOL_SERVERS: dict[str, str] = {
    "fs_read": "lab.agent.tools.fs_read",
    "fs_write": "lab.agent.tools.fs_write",
    "fs_grep": "lab.agent.tools.fs_grep",
    "shell_exec": "lab.agent.tools.shell_exec",
    "http_fetch": "lab.agent.tools.http_fetch",
    "python_eval": "lab.agent.tools.python_eval",
}

__all__ = ["TOOL_SERVERS"]
