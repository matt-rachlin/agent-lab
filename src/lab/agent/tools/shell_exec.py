"""`shell_exec` — `bash -c "<command>"` cwd=/workspace with no stdin.

We deliberately don't accept arbitrary stdin: the agent's only inputs are the
function argument and the workspace filesystem. This keeps the tool surface
tight (no covert-channel via stdin) and the schema small.
"""

from __future__ import annotations

import subprocess
from typing import Any

from mcp.server.fastmcp import FastMCP

from lab.agent.tools._common import WORKSPACE_ROOT

mcp: FastMCP = FastMCP("lab.shell_exec")


@mcp.tool()
def shell_exec(command: str, timeout_sec: int = 30) -> dict[str, Any]:
    """Run `bash -c "<command>"` with cwd=/workspace.

    Args:
        command: Shell command string evaluated by `bash -c`.
        timeout_sec: Hard kill after this many seconds (default 30, max 300).

    Returns:
        `{stdout: str, stderr: str, exit_code: int, timed_out: bool}`.
        `stdout`/`stderr` are decoded as UTF-8 with replacement on errors.
    """

    if not command:
        raise ValueError("command must not be empty")
    if timeout_sec <= 0:
        raise ValueError("timeout_sec must be positive")
    if timeout_sec > 300:
        raise ValueError("timeout_sec must be <= 300")
    timed_out = False
    try:
        proc = subprocess.run(
            ["bash", "-c", command],
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
