"""`http_fetch` — HTTP GET against the task's allow-listed host set.

Allow-list comes from `LAB_HTTP_ALLOWLIST` (comma-separated hostnames). Empty
== deny everything. The agent sandbox's network layer also enforces a host
allow-list at the netns level so this is defense-in-depth, not the only line.

Hard-coded constraints:
    * 10-second timeout (cannot be raised by the model).
    * 1 MB response cap (configurable down via `max_bytes`).
    * `User-Agent: lab-agent/0.1`.
    * GET only (no method parameter exposed).
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import httpx
from mcp.server.fastmcp import FastMCP

from lab.agent.tools._common import http_allowlist

mcp: FastMCP = FastMCP("lab.http_fetch")

USER_AGENT = "lab-agent/0.1"
HARD_TIMEOUT_SEC = 10.0
HARD_MAX_BYTES = 1_048_576  # 1 MiB


@mcp.tool()
def http_fetch(url: str, max_bytes: int = HARD_MAX_BYTES) -> dict[str, Any]:
    """GET `url`, refusing anything not on the task's host allow-list.

    Args:
        url: Absolute http(s) URL. The host must appear verbatim in
            `LAB_HTTP_ALLOWLIST` (set by the solver from `task.sandbox.network`).
        max_bytes: Response body cap. Hard-capped at 1 MiB regardless of input.

    Returns:
        `{status: int, headers: dict, content: str, truncated: bool, url: str}`.
        Body is decoded as UTF-8 with replacement; binary responses end up as
        unreadable but valid strings.
    """

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    effective_max_bytes = min(max_bytes, HARD_MAX_BYTES)
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"only http(s) URLs are allowed, got scheme={parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"url {url!r} has no hostname")
    allow = http_allowlist()
    if not allow:
        raise PermissionError(
            "no hosts are allow-listed for this task; "
            "set LAB_HTTP_ALLOWLIST or task.sandbox.network"
        )
    if host not in allow:
        raise PermissionError(f"host {host!r} is not in the allow-list ({sorted(allow)})")
    headers = {"User-Agent": USER_AGENT}
    try:
        # Cap both connect + read; both share the 10 s budget.
        with httpx.Client(
            timeout=HARD_TIMEOUT_SEC,
            follow_redirects=False,
            headers=headers,
        ) as client:
            resp = client.get(url)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"http_fetch failed: {exc}") from exc
    raw = resp.content[: effective_max_bytes + 1]
    truncated = len(raw) > effective_max_bytes
    body_bytes = raw[:effective_max_bytes] if truncated else raw
    content = body_bytes.decode("utf-8", errors="replace")
    return {
        "status": resp.status_code,
        "headers": dict(resp.headers),
        "content": content,
        "truncated": truncated,
        "url": str(resp.url),
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
