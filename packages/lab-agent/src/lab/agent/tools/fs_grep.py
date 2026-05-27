"""`fs_grep` — ripgrep-backed search under `/workspace`.

Thin wrapper around `rg --json`. We parse the line-delimited JSON stream and
emit a structured list of matches; the model gets file/line/text without
having to grok ripgrep's raw output.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

from lab.agent.tools._common import PathEscapeError, resolve_workspace_path
from mcp.server.fastmcp import FastMCP

mcp: FastMCP = FastMCP("lab.fs_grep")


@mcp.tool()
def fs_grep(
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    max_results: int = 100,
) -> dict[str, Any]:
    """Search `path` (under `/workspace`) for `pattern`.

    Args:
        pattern: Regex pattern passed to `rg`.
        path: Subpath to search; defaults to the workspace root.
        glob: Optional `rg --glob` filter (e.g. `'*.py'`).
        max_results: Maximum number of match entries to return (cap on the
            per-file `--max-count` value passed to `rg`).

    Returns:
        `{matches: [{path, line_number, text}], truncated: bool}`.
    """

    if not pattern:
        raise ValueError("pattern must not be empty")
    if max_results <= 0:
        raise ValueError("max_results must be positive")
    try:
        resolved = resolve_workspace_path(path)
    except PathEscapeError as exc:
        raise ValueError(str(exc)) from exc
    argv = [
        "rg",
        "--json",
        f"--max-count={max_results}",
        pattern,
        str(resolved),
    ]
    if glob is not None:
        argv[1:1] = ["--glob", glob]
    proc = subprocess.run(
        argv,
        capture_output=True,
        check=False,
        timeout=30,
    )
    # `rg` returns 1 when there are no matches; only treat 2+ as an error.
    if proc.returncode >= 2:
        raise RuntimeError(
            f"rg failed (exit {proc.returncode}): {proc.stderr.decode(errors='replace').strip()}"
        )
    matches: list[dict[str, Any]] = []
    truncated = False
    for line in proc.stdout.splitlines():
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("type") != "match":
            continue
        data = evt.get("data", {})
        path_text = data.get("path", {}).get("text", "")
        line_no = data.get("line_number")
        lines_obj = data.get("lines", {})
        text = lines_obj.get("text", "")
        matches.append(
            {
                "path": path_text,
                "line_number": line_no,
                "text": text.rstrip("\n"),
            }
        )
        if len(matches) >= max_results:
            truncated = True
            break
    return {"matches": matches, "truncated": truncated}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
