"""Integration tests for the sandbox network allow-list.

Verifies the v0.1 DNS-restricted allow-list: when a host is in the list,
`http_fetch` succeeds; when it isn't, the lookup fails before the request
goes out.

Requires:
  * podman + runsc + `lab-agent-sandbox:0.1`
  * Outbound IPv4 to example.com on the host (so we can resolve at sandbox
    start time).
"""

from __future__ import annotations

import socket

import pytest

from lab.agent.sandbox import Sandbox, gvisor_available
from lab.agent.tools import TOOL_SERVERS
from lab.inspect_bridge.tools import _invoke_tool_via_sandbox_sync

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _require_gvisor_and_network() -> None:
    if not gvisor_available():
        pytest.skip("gVisor not available")
    try:
        socket.getaddrinfo("example.com", None, family=socket.AF_INET)
    except socket.gaierror:
        pytest.skip("no outbound DNS for example.com from this host")


def test_http_fetch_succeeds_for_allow_listed_host() -> None:
    with Sandbox(
        network=["example.com"],
        env={"LAB_HTTP_ALLOWLIST": "example.com"},
    ) as sb:
        out = _invoke_tool_via_sandbox_sync(
            sb,
            TOOL_SERVERS["http_fetch"],
            "http_fetch",
            {"url": "https://example.com/"},
        )
        assert out["status"] == 200  # type: ignore[index]
        # example.com returns a stable "Example Domain" body — substring
        # match is robust against minor markup changes.
        assert "Example Domain" in out["content"]  # type: ignore[index]


def test_http_fetch_rejects_non_allow_listed_host() -> None:
    # Allow-list contains a host, but the tool's own LAB_HTTP_ALLOWLIST is
    # what gates each request: leave it pointed at example.com only, then
    # ask for some other host. The tool refuses *before* it ever hits the
    # network layer (this is the layered defense).
    with (
        Sandbox(
            network=["example.com"],
            env={"LAB_HTTP_ALLOWLIST": "example.com"},
        ) as sb,
        pytest.raises(Exception, match=r"not in the allow-list|allow-list"),
    ):
        _invoke_tool_via_sandbox_sync(
            sb,
            TOOL_SERVERS["http_fetch"],
            "http_fetch",
            {"url": "https://api.github.com/"},
        )


def test_dns_blocks_unlisted_hosts() -> None:
    # Belt for the braces above: even if the tool layer was bypassed,
    # DNS for an unlisted host fails inside the sandbox.
    with Sandbox(
        network=["example.com"],
        env={"LAB_HTTP_ALLOWLIST": "example.com"},
    ) as sb:
        res = sb.exec(["getent", "hosts", "api.github.com"], timeout=10)
        assert res.exit_code != 0, "DNS unexpectedly resolved an unlisted host"


def test_dns_resolves_listed_host() -> None:
    with Sandbox(
        network=["example.com"],
        env={"LAB_HTTP_ALLOWLIST": "example.com"},
    ) as sb:
        res = sb.exec(["getent", "hosts", "example.com"], timeout=10)
        assert res.exit_code == 0, res.stderr
        assert b"example.com" in res.stdout
