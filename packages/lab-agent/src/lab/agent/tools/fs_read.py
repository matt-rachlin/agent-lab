"""`fs_read` — read a UTF-8 file under `/workspace`.

Run as a FastMCP stdio server:

    python -m lab.agent.tools.fs_read

The solver pipes MCP framed JSON over the subprocess's stdin/stdout.
"""

from __future__ import annotations

from typing import Any

from lab.agent.tools._common import PathEscapeError, resolve_workspace_path
from mcp.server.fastmcp import FastMCP

mcp: FastMCP = FastMCP("lab.fs_read")


@mcp.tool()
def fs_read(path: str, max_bytes: int = 65536) -> dict[str, Any]:
    """Read a UTF-8 file under `/workspace`.

    Args:
        path: Path under `/workspace`. Absolute paths must start with
            `/workspace/`; relative paths are taken as relative to it.
            Path-escape (e.g. `../etc/passwd`) is refused.
        max_bytes: Maximum number of bytes to read (default 65536, hard cap).

    Returns:
        `{content: str, size: int, truncated: bool, path: str}` where
        `size` is the full file size in bytes and `truncated` indicates
        whether `content` was capped at `max_bytes`.
    """

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    try:
        resolved = resolve_workspace_path(path)
    except PathEscapeError as exc:
        raise ValueError(str(exc)) from exc
    if not resolved.exists():
        raise FileNotFoundError(f"{path!r} does not exist")
    if not resolved.is_file():
        raise IsADirectoryError(f"{path!r} is not a regular file")
    raw = resolved.read_bytes()
    size = len(raw)
    truncated = size > max_bytes
    content_bytes = raw[:max_bytes] if truncated else raw
    try:
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path!r} is not valid UTF-8: {exc}") from exc
    return {
        "content": content,
        "size": size,
        "truncated": truncated,
        "path": str(resolved),
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
