"""`fs_write` — write/append/overwrite a file under `/workspace`.

Modes:
    * `create`    — fail if the file already exists (default).
    * `overwrite` — replace the file's contents.
    * `append`    — append to an existing file (or create if absent).

Path validation matches `fs_read`: anything that escapes `/workspace` is
refused.
"""

from __future__ import annotations

from typing import Any, Literal

from lab.agent.tools._common import PathEscapeError, resolve_workspace_path
from mcp.server.fastmcp import FastMCP

mcp: FastMCP = FastMCP("lab.fs_write")

WriteMode = Literal["create", "overwrite", "append"]


@mcp.tool()
def fs_write(
    path: str,
    content: str,
    mode: WriteMode = "create",
) -> dict[str, Any]:
    """Write `content` to a file under `/workspace`.

    Args:
        path: Path under `/workspace`. Path-escape is refused.
        content: UTF-8 text to write.
        mode: `create` (default; fail if exists), `overwrite`, or `append`.

    Returns:
        `{path: str, bytes_written: int, mode: str}`.
    """

    if mode not in ("create", "overwrite", "append"):
        raise ValueError(f"mode must be one of create|overwrite|append, got {mode!r}")
    try:
        resolved = resolve_workspace_path(path)
    except PathEscapeError as exc:
        raise ValueError(str(exc)) from exc
    if mode == "create" and resolved.exists():
        raise FileExistsError(f"{path!r} already exists (use mode='overwrite' to replace)")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode("utf-8")
    if mode == "append":
        with resolved.open("ab") as fh:
            fh.write(data)
    else:
        # `create` already verified non-existence; `overwrite` writes
        # unconditionally. Either way we open with `wb`.
        with resolved.open("wb") as fh:
            fh.write(data)
    return {
        "path": str(resolved),
        "bytes_written": len(data),
        "mode": mode,
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
