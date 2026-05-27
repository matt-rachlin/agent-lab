"""`python_eval` — run a Python snippet via `python3 -c "<code>"`.

cwd=/workspace; no stdin; no implicit imports beyond the stdlib. The subprocess
inherits the sandbox's locked-down environment, so the only escape hatches are
already-installed Python packages — which means the stdlib for the moment.
"""

from __future__ import annotations

import subprocess
from typing import Any

from lab.agent.tools._common import WORKSPACE_ROOT
from mcp.server.fastmcp import FastMCP

mcp: FastMCP = FastMCP("lab.python_eval")


@mcp.tool()
def python_eval(code: str, timeout_sec: int = 30) -> dict[str, Any]:
    """Run `python3 -c "<code>"` with cwd=/workspace.

    Args:
        code: Python source passed as `-c` argument.
        timeout_sec: Hard kill after this many seconds (default 30, max 300).

    Returns:
        `{stdout: str, stderr: str, exit_code: int, timed_out: bool}`.
    """

    if not code:
        raise ValueError("code must not be empty")
    if timeout_sec <= 0:
        raise ValueError("timeout_sec must be positive")
    if timeout_sec > 300:
        raise ValueError("timeout_sec must be <= 300")
    timed_out = False
    try:
        proc = subprocess.run(
            ["python3", "-c", code],
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            check=False,
            timeout=timeout_sec,
        )
        stdout_b = proc.stdout or b""
        stderr_b = proc.stderr or b""
        exit_code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout_b = exc.stdout or b""
        stderr_b = exc.stderr or b""
        exit_code = 124  # GNU `timeout` convention
    return {
        "stdout": stdout_b.decode("utf-8", errors="replace"),
        "stderr": stderr_b.decode("utf-8", errors="replace"),
        "exit_code": exit_code,
        "timed_out": timed_out,
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
