"""Integration tests for `lab.agent.sandbox`.

Requires:
  * podman in PATH
  * runsc (gVisor) installed
  * `lab-agent-sandbox:0.1` image built (`just sandbox-build`)

All tests skip cleanly if the gVisor smoke probe fails — these are real
container launches and cost ~1-3s each, so they don't run in pure-CI mode.
"""

from __future__ import annotations

import pytest
from lab.agent.sandbox import Sandbox, SandboxError, gvisor_available

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _require_gvisor() -> None:
    if not gvisor_available():
        pytest.skip(
            "gVisor not available: install runsc and build lab-agent-sandbox:0.1 "
            "(see `just sandbox-build`)"
        )


def test_python_runs() -> None:
    with Sandbox() as sb:
        res = sb.exec(["python3", "-c", "print(2 + 2)"])
        assert res.exit_code == 0, res.stderr
        assert res.stdout.strip() == b"4"
        assert res.stderr == b""
        assert res.timed_out is False
        assert res.duration_ms >= 0


def test_workspace_file_roundtrip() -> None:
    files = {"in.txt": b"hello sandbox\n", "subdir/nested.txt": b"deep\n"}
    with Sandbox(workspace_files=files) as sb:
        # Files are readable from inside the container...
        res = sb.exec(["cat", "in.txt"])
        assert res.exit_code == 0
        assert res.stdout == b"hello sandbox\n"
        # ...show up in list_workspace_files...
        listing = sb.list_workspace_files()
        assert "in.txt" in listing
        assert "subdir/nested.txt" in listing
        # ...and read back to the host with the original bytes.
        assert sb.read_workspace_file("in.txt") == b"hello sandbox\n"
        assert sb.read_workspace_file("subdir/nested.txt") == b"deep\n"


def test_workspace_write_inside_then_read_out() -> None:
    with Sandbox() as sb:
        res = sb.exec(["bash", "-c", "echo crafted > out.txt"])
        assert res.exit_code == 0, res.stderr
        assert sb.read_workspace_file("out.txt") == b"crafted\n"


def test_time_limit_kills_runaway_process() -> None:
    with Sandbox(time_limit_sec=2) as sb:
        res = sb.exec(["sleep", "30"], timeout=2)
        assert res.timed_out is True
        assert res.exit_code == 124  # GNU timeout convention
        assert res.duration_ms < 10_000, "should hard-kill near the deadline"


def test_exit_code_propagates() -> None:
    with Sandbox() as sb:
        res = sb.exec(["bash", "-c", "exit 42"])
        assert res.exit_code == 42
        assert res.stderr == b""


def test_stderr_captured_separately() -> None:
    with Sandbox() as sb:
        res = sb.exec(["bash", "-c", "echo OUT; echo ERR 1>&2; exit 3"])
        assert res.exit_code == 3
        assert res.stdout == b"OUT\n"
        assert res.stderr == b"ERR\n"


def test_network_none_blocks_dns() -> None:
    # With --network=none there is no resolver and no route; DNS lookup must
    # fail. We don't care which specific failure mode (NXDOMAIN, no nameserver,
    # connection refused) — just that getent reports it.
    with Sandbox(network="none") as sb:
        res = sb.exec(["getent", "hosts", "example.com"], timeout=10)
        assert res.exit_code != 0, "DNS unexpectedly resolved under --network=none"


def test_stop_is_idempotent() -> None:
    sb = Sandbox()
    sb.start()
    sb.stop()
    sb.stop()  # should not raise


def test_exec_after_stop_raises() -> None:
    sb = Sandbox()
    sb.start()
    sb.stop()
    with pytest.raises(SandboxError, match="already stopped"):
        sb.exec(["true"])


def test_exec_before_start_raises() -> None:
    sb = Sandbox()
    with pytest.raises(SandboxError, match="must be called before exec"):
        sb.exec(["true"])
    sb.stop()


def test_runs_as_non_root_agent_user() -> None:
    """The image's USER directive should put us at uid 10001 (agent)."""
    with Sandbox() as sb:
        res = sb.exec(["id", "-u"])
        assert res.exit_code == 0
        assert res.stdout.strip() == b"10001"


def test_env_vars_propagate() -> None:
    with Sandbox(env={"LAB_CELL_ID": "abc123"}) as sb:
        res = sb.exec(["bash", "-c", "echo $LAB_CELL_ID"])
        assert res.exit_code == 0
        assert res.stdout.strip() == b"abc123"
