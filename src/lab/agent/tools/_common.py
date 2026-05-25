"""Shared helpers for the lab agent tool servers.

These tool servers are designed to run *inside* the agent sandbox: they read
and write under `/workspace`, shell out to in-container utilities, and speak
MCP over stdio to whatever solver launched them.

We keep the helper surface small on purpose: any reusable logic that's hard
to test inside the sandbox should live in `lab.agent.sandbox` instead.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

WORKSPACE_ROOT: str = "/workspace"


class PathEscapeError(ValueError):
    """Raised when a tool request resolves outside `/workspace`.

    Carries the user-facing message verbatim — tool servers surface it back
    to the model so it can self-correct.
    """


def resolve_workspace_path(path: str, *, root: str = WORKSPACE_ROOT) -> Path:
    """Resolve `path` against `/workspace`, refusing anything that escapes.

    Rules:
        * Reject absolute paths that don't start with the workspace root.
        * Reject any path whose normalised form contains `..` traversal that
          leaves the workspace.
        * Symlinks are NOT followed for the safety check (we lexically
          normalise); the underlying syscall will still fail safely if the
          symlink points outside the sandbox, but we don't pretend to defend
          against symlink attacks at the lexical layer.

    Returns a concrete `pathlib.Path` rooted inside `/workspace`.
    """

    if path == "":
        raise PathEscapeError("path must not be empty")
    p = PurePosixPath(path)
    if p.is_absolute():
        if not str(p).startswith(root + "/") and str(p) != root:
            raise PathEscapeError(f"absolute path {path!r} is outside {root}")
        rel = p.relative_to(root) if str(p) != root else PurePosixPath(".")
    else:
        rel = p
    # Lexically normalise without touching the filesystem. This catches the
    # `subdir/../../etc/passwd` family without depending on whether the file
    # exists yet.
    parts: list[str] = []
    for part in rel.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                raise PathEscapeError(f"path {path!r} escapes {root} via '..'")
            parts.pop()
        else:
            parts.append(part)
    resolved = Path(root, *parts) if parts else Path(root)
    return resolved


def is_within_workspace(path: Path, *, root: str = WORKSPACE_ROOT) -> bool:
    """Return True iff `path` is lexically inside `root`.

    Mirrors `resolve_workspace_path` semantics so callers can validate a
    `Path` they constructed themselves (e.g. after a file walk).
    """

    try:
        path.resolve(strict=False).relative_to(Path(root).resolve(strict=False))
    except ValueError:
        return False
    return True


def http_allowlist() -> frozenset[str]:
    """Parse `LAB_HTTP_ALLOWLIST` into a set of allowed hostnames.

    Empty / unset == empty set == http_fetch refuses all hosts. The format is
    a comma-separated list of bare hostnames (no scheme, no port, no path).
    The sandbox's network allow-list mirrors this same env var so the two
    layers stay in sync.
    """

    raw = os.environ.get("LAB_HTTP_ALLOWLIST", "")
    return frozenset(part.strip() for part in raw.split(",") if part.strip())
