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

import os
from collections.abc import Iterable
from typing import Any

TOOL_SERVERS: dict[str, str] = {
    "fs_read": "lab.agent.tools.fs_read",
    "fs_write": "lab.agent.tools.fs_write",
    "fs_grep": "lab.agent.tools.fs_grep",
    "shell_exec": "lab.agent.tools.shell_exec",
    "http_fetch": "lab.agent.tools.http_fetch",
    "python_eval": "lab.agent.tools.python_eval",
    "kb_query": "lab.agent.tools.kb_query",
}

#: Tools that need read access to the host KB root (`~/db/kb/`). The harness
#: must mount the KB root into the sandbox and set `LAB_KB_ROOT` accordingly
#: when any of these tools appear in a task's tool list. Keep this set small —
#: a tool earns membership by needing the lab's offline RAG corpora.
TOOLS_NEEDING_KB_MOUNT: frozenset[str] = frozenset({"kb_query"})


def task_needs_kb_mount(tool_specs: Iterable[Any] | None) -> bool:
    """Return True iff `tool_specs` references a tool that needs the KB mount.

    Accepts the same shape as `Task.tools` — a list of dicts with a ``name``
    key — and tolerates plain strings or `None`.
    """

    if not tool_specs:
        return False
    for spec in tool_specs:
        name = spec.get("name") if isinstance(spec, dict) else spec
        if isinstance(name, str) and name in TOOLS_NEEDING_KB_MOUNT:
            return True
    return False


#: Tools that may trigger the Phase 7 cross-encoder reranker. When any of
#: these appear in a task's tool list AND the reranker is enabled
#: (``LAB_RAG_RERANKER`` unset or != ``"none"``), the harness mounts the
#: shared HF cache so reranker weights persist across cells.
TOOLS_NEEDING_HF_CACHE: frozenset[str] = frozenset({"kb_query"})


def task_needs_hf_cache_mount(
    tool_specs: Iterable[Any] | None,
    *,
    reranker_env: str | None = None,
) -> bool:
    """Return True iff `tool_specs` may trigger the reranker AND it is enabled.

    ``reranker_env`` overrides the live ``LAB_RAG_RERANKER`` value (used by
    tests). The mount is needed iff:

    * the task uses a tool from :data:`TOOLS_NEEDING_HF_CACHE`, AND
    * the reranker is not disabled (``LAB_RAG_RERANKER`` unset, empty, or
      anything other than the case-insensitive sentinel ``"none"``).

    Returning False keeps the sandbox surface minimal — no hf-cache mount
    when the reranker is provably off.
    """

    if not tool_specs:
        return False
    value = reranker_env if reranker_env is not None else os.environ.get("LAB_RAG_RERANKER", "")
    if value.strip().lower() == "none":
        return False
    for spec in tool_specs:
        name = spec.get("name") if isinstance(spec, dict) else spec
        if isinstance(name, str) and name in TOOLS_NEEDING_HF_CACHE:
            return True
    return False


__all__ = [
    "TOOLS_NEEDING_HF_CACHE",
    "TOOLS_NEEDING_KB_MOUNT",
    "TOOL_SERVERS",
    "task_needs_hf_cache_mount",
    "task_needs_kb_mount",
]
