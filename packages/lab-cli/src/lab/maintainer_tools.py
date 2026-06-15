"""NS-2 Code Maintainer — in-process WRITE tools (Tool ABI, ADR-012 / ADR-013).

These are the maintainer's tool implementations: plain in-process callables (NOT
the sandboxed FastMCP `lab.agent.tools` MCP-subprocess backend — that is the
"#13 future" seam). Every tool is bound to a single WORKSPACE directory that the
caller guarantees is a git-tracked scratch repo, so each mutation is diff-able
and revertible — ADR-013's `write_local` ("reversible local mutation in a
git-tracked workspace").

Safety properties enforced HERE (not by the LLM):
  * PATH CONFINEMENT — every path argument is resolved and must stay inside the
    workspace root; a `..` / absolute-path escape returns an error and never
    touches the filesystem (`_resolve` raises `PathEscape`).
  * COMMAND ALLOWLIST — `mtn_run` only executes a fixed allowlist of program
    heads (pytest / ruff / python). Anything else is refused. The command runs
    with `cwd=workspace`, `shell=False` (argv, no shell metacharacters), and a
    timeout. An allowlist is strictly safer than arbitrary shell, so v0 uses one.

Side-effect classes (drive the ADR-013 gate in run_agent):
  * mtn_read  -> "read"        (no mutation)
  * mtn_write -> "write_local" (reversible file write in the git workspace)
  * mtn_run   -> "write_local" (an allowlisted command MAY mutate the workspace,
                  e.g. write caches / fixtures; classified as write_local, not
                  read, so it goes through the WRITE gate)
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lab.platform.agent_runtime import Tool

#: The actor identity the maintainer runs under (must match the authz grant key).
MAINTAINER_ACTOR = "maintainer"

#: Allowlisted program heads for mtn_run. An allowlist (not arbitrary shell) is
#: the safer v0 choice; extend deliberately. `uv` is included so the agent can
#: run `uv run pytest` against a uv-managed scratch repo.
_CMD_ALLOWLIST: frozenset[str] = frozenset({"pytest", "ruff", "python", "python3", "uv"})

#: Cap captured output so a single tool result cannot blow the context budget.
_OUTPUT_CAP = 8000


class PathEscape(ValueError):
    """Raised when a path argument resolves outside the workspace root."""


def _resolve(workspace: Path, path: str) -> Path:
    """Resolve `path` relative to `workspace` and assert it stays inside.

    Rejects absolute paths and `..` traversal that would escape the workspace —
    the path-confinement boundary. Does not require the file to exist (writes
    create new files)."""
    root = workspace.resolve()
    candidate = (root / path).resolve()
    if candidate != root and root not in candidate.parents:
        raise PathEscape(f"path escapes workspace: {path!r}")
    return candidate


def make_read(workspace: Path) -> Callable[[str], dict[str, Any]]:
    def mtn_read(path: str) -> dict[str, Any]:
        """Read a file inside the workspace. side_effect="read"."""
        try:
            target = _resolve(workspace, path)
        except PathEscape as exc:
            return {"error": str(exc)}
        if not target.is_file():
            return {"error": f"not a file: {path!r}"}
        return {"path": path, "content": target.read_text(encoding="utf-8")}

    return mtn_read


def make_write(workspace: Path) -> Callable[[str, str], dict[str, Any]]:
    def mtn_write(path: str, content: str) -> dict[str, Any]:
        """Write (overwrite) a file inside the workspace. side_effect="write_local"
        — reversible because the workspace is git-tracked."""
        try:
            target = _resolve(workspace, path)
        except PathEscape as exc:
            return {"error": str(exc)}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"path": path, "bytes_written": len(content.encode("utf-8"))}

    return mtn_write


def make_run(workspace: Path) -> Callable[[str], dict[str, Any]]:
    def mtn_run(cmd: str) -> dict[str, Any]:
        """Run an ALLOWLISTED command (pytest/ruff/python/uv) in the workspace and
        capture its output. side_effect="write_local" (the command may mutate the
        workspace). shell=False, cwd=workspace, timeout-bounded. Returns
        {returncode, stdout, stderr} — returncode==0 is the objective pass signal.
        """
        parts = cmd.split()
        if not parts:
            return {"error": "empty command"}
        head = Path(parts[0]).name  # tolerate e.g. "./python" -> "python"
        if head not in _CMD_ALLOWLIST:
            return {
                "error": (f"command not allowed: {head!r}; allowlist = {sorted(_CMD_ALLOWLIST)}")
            }
        try:
            proc = subprocess.run(
                parts,
                cwd=str(workspace.resolve()),
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"error": "command timed out", "returncode": -1}
        except FileNotFoundError as exc:
            return {"error": f"command not found: {exc}", "returncode": -1}
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout[-_OUTPUT_CAP:],
            "stderr": proc.stderr[-_OUTPUT_CAP:],
        }

    return mtn_run


# Module-level mtn_run wrapper for re-export/introspection (uses no fixed
# workspace; the real bound impls come from build_tools). Kept for the public
# `lab.maintainer.mtn_run` symbol referenced by callers/tests.
def mtn_run(cmd: str, *, workspace: str) -> dict[str, Any]:
    """Standalone allowlisted run against an explicit workspace path."""
    return make_run(Path(workspace))(cmd)


_READ_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
}
_WRITE_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
    "required": ["path", "content"],
}
_RUN_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {"cmd": {"type": "string"}},
    "required": ["cmd"],
}


def build_tools(workspace: str) -> list[Tool]:
    """The maintainer's three tools, all path-confined to `workspace`.

    side_effects: mtn_read=read, mtn_write=write_local, mtn_run=write_local.
    The workspace MUST be a git-tracked scratch repo so write_local mutations are
    reversible (ADR-013)."""
    root = Path(workspace)
    return [
        Tool(
            name="mtn_read",
            description=(
                "Read a UTF-8 text file inside the workspace (path relative to the "
                "workspace root). Read-only."
            ),
            parameters=_READ_PARAMS,
            impl=make_read(root),
            side_effect="read",
            capability="fs_read",
        ),
        Tool(
            name="mtn_write",
            description=(
                "Overwrite a file inside the workspace with the given content "
                "(creates parent dirs). Reversible: the workspace is git-tracked."
            ),
            parameters=_WRITE_PARAMS,
            impl=make_write(root),
            side_effect="write_local",
            capability="fs_write",
        ),
        Tool(
            name="mtn_run",
            description=(
                "Run an allowlisted command (pytest / ruff / python / uv) in the "
                "workspace and capture stdout/stderr/returncode. returncode==0 "
                "means the tests passed."
            ),
            parameters=_RUN_PARAMS,
            impl=make_run(root),
            side_effect="write_local",
            capability="run_cmd",
        ),
    ]


__all__ = [
    "MAINTAINER_ACTOR",
    "PathEscape",
    "build_tools",
    "make_read",
    "make_run",
    "make_write",
    "mtn_run",
]
