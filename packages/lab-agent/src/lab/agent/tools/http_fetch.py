"""`http_fetch` — HTTP GET against the task's allow-listed host set.

Allow-list comes from `LAB_HTTP_ALLOWLIST` (comma-separated hostnames). Empty
== deny everything. The agent sandbox's network layer also enforces a host
allow-list at the netns level so this is defense-in-depth, not the only line.

Hard-coded constraints:
    * 10-second timeout (cannot be raised by the model).
    * 1 MB response cap (configurable down via `max_bytes`).
    * `User-Agent: lab-agent/0.1`.
    * GET only (no method parameter exposed).

**Offline fixture mode (added 6f)**: when `LAB_HTTP_FIXTURE_DIR` is set, any
fetch whose hostname is in the allow-list is served from
`<LAB_HTTP_FIXTURE_DIR>/<host>/<path>` on disk instead of hitting the network.
Used by the PBS-Agent v0.1 http-domain tasks so they're reproducible and
don't require live external services. Returns 200 on hit, 404 on miss; never
makes a network call when this var is set.

**Image-rebuild invariant (added 6h-e after F-005)**: the sandbox image
embeds a frozen copy of this module via `COPY src/lab/agent/...` in
``containers/Containerfile.agent-sandbox``. If you change *anything* in
this file — including fixture-mode semantics — you must rebuild the image
(``just sandbox-build``) before the change takes effect inside the
sandbox. EXP-002 was bitten by this: the sweep ran with three distinct
image hashes (mid-sweep drift caused by ``podman image prune`` reaping
layers between cells); at least one image predated the fixture-mode code
and so the http tasks hit live ``example.org`` instead. The image-hash
drift guard in ``src/lab/sweep/runner.py`` now aborts the sweep on
mid-flight hash change; this docstring is the human-facing reminder.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from lab.agent.tools._common import http_allowlist
from mcp.server.fastmcp import FastMCP

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

    # Offline fixture mode: serve from disk if LAB_HTTP_FIXTURE_DIR is set.
    fixture_dir = os.environ.get("LAB_HTTP_FIXTURE_DIR")
    if fixture_dir:
        return _serve_from_fixture(
            fixture_dir=fixture_dir,
            host=host,
            path=parsed.path or "/",
            url=url,
            max_bytes=effective_max_bytes,
        )

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


def _serve_from_fixture(
    *,
    fixture_dir: str,
    host: str,
    path: str,
    url: str,
    max_bytes: int,
) -> dict[str, Any]:
    """Serve a fixture response from `<fixture_dir>/<host>/<path>`.

    Path is normalised: leading slash stripped, trailing slash mapped to
    `index.html`. Any path-escape outside `<fixture_dir>/<host>` is treated
    as 404, not an error — we want fixture lookups to be quiet about misses.
    """

    base = Path(fixture_dir).resolve() / host
    norm = path.lstrip("/")
    if not norm or norm.endswith("/"):
        norm = (norm + "index.html") if norm else "index.html"
    candidate = (base / norm).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return {
            "status": 404,
            "headers": {"x-lab-fixture": "path-escape"},
            "content": "",
            "truncated": False,
            "url": url,
        }
    if not candidate.exists() or not candidate.is_file():
        return {
            "status": 404,
            "headers": {"x-lab-fixture": "miss"},
            "content": "",
            "truncated": False,
            "url": url,
        }
    raw = candidate.read_bytes()
    truncated = len(raw) > max_bytes
    body_bytes = raw[:max_bytes] if truncated else raw
    content = body_bytes.decode("utf-8", errors="replace")
    return {
        "status": 200,
        "headers": {
            "x-lab-fixture": "hit",
            "content-length": str(len(raw)),
        },
        "content": content,
        "truncated": truncated,
        "url": url,
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
