"""Verify the offline-fixture wiring works end-to-end inside the sandbox.

F-005 EXP-002 follow-up: `http-fetch-and-extract` and `http-fetch-and-count`
hit live `example.org` instead of the local fixture during EXP-002. Root
cause was probably mid-sweep image drift (Surprise 4 in F-005) — but the
test that locks down the contract is the same regardless: when a task
declares `sandbox.env.LAB_HTTP_FIXTURE_DIR` AND stages
`_http_fixtures/<host>/<path>` via `workspace_files`, the in-sandbox
`http_fetch` MCP tool MUST serve from disk (200 + `x-lab-fixture: hit`),
never the live network.

Skips cleanly when gVisor is not available.
"""

from __future__ import annotations

import pytest

from lab.agent.sandbox import Sandbox, gvisor_available

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _require_gvisor() -> None:
    if not gvisor_available():
        pytest.skip("gVisor not available")


def test_http_fetch_serves_from_fixture_dir_in_sandbox() -> None:
    """End-to-end: task env + workspace_files → MCP tool returns fixture."""

    from lab.inspect_bridge.tools import _invoke_tool_via_sandbox_sync

    with Sandbox(
        env={
            "LAB_HTTP_FIXTURE_DIR": "/workspace/_http_fixtures",
            "LAB_HTTP_ALLOWLIST": "example.com",
        },
        workspace_files={
            "_http_fixtures/example.com/status.json": (
                b'{"service": "lab", "uptime_minutes": 4242, "healthy": true}'
            ),
        },
        # network='none' is the strongest possible signal — if fixture mode
        # weren't working we'd see a connection error, not a live page.
        network="none",
    ) as box:
        result = _invoke_tool_via_sandbox_sync(
            box,
            "lab.agent.tools.http_fetch",
            "http_fetch",
            {"url": "http://example.com/status.json"},
        )

    assert isinstance(result, dict), f"http_fetch returned non-dict: {result!r}"
    assert result.get("status") == 200, f"expected 200, got {result.get('status')}"
    assert (
        result.get("headers", {}).get("x-lab-fixture") == "hit"
    ), f"fixture mode did not engage: headers={result.get('headers')!r}"
    assert "4242" in (
        result.get("content") or ""
    ), f"fixture body missing uptime: {result.get('content')!r}"


def test_http_fetch_fixture_miss_returns_404_no_live_traffic() -> None:
    """If a fixture path is not staged, fixture mode returns 404 — does NOT
    fall back to live network. This is the invariant that protects sweeps
    from accidentally hitting the public internet when a fixture file is
    forgotten.
    """

    from lab.inspect_bridge.tools import _invoke_tool_via_sandbox_sync

    with Sandbox(
        env={
            "LAB_HTTP_FIXTURE_DIR": "/workspace/_http_fixtures",
            "LAB_HTTP_ALLOWLIST": "example.com",
        },
        # No workspace_files — fixture dir will be empty (or absent).
        network="none",
    ) as box:
        result = _invoke_tool_via_sandbox_sync(
            box,
            "lab.agent.tools.http_fetch",
            "http_fetch",
            {"url": "http://example.com/missing.json"},
        )

    assert result.get("status") == 404
    assert result.get("headers", {}).get("x-lab-fixture") == "miss"
