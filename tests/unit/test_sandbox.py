"""Unit tests for `lab.agent.sandbox` — purely command-builder shape checks.

These do NOT invoke podman. The cmd-builder is the only piece of the sandbox
worth pinning at the unit level; everything else has to be exercised against
a real container (see `tests/integration/test_sandbox.py`).
"""

from __future__ import annotations

import pytest
from lab.agent.sandbox import Sandbox, _build_run_argv


def test_build_run_argv_minimum_required_flags() -> None:
    argv = _build_run_argv(
        image="lab-agent-sandbox:0.1",
        name="lab-sandbox-deadbeef",
        runtime="runsc",
        network="none",
        mem_limit="1g",
        cpu_limit=2.0,
        env=None,
    )
    # Process basics
    assert argv[:2] == ["podman", "run"]
    assert "--detach" in argv
    assert "--rm" in argv
    assert "lab-agent-sandbox:0.1" in argv
    assert argv[-2:] == ["sleep", "infinity"], "container must idle for exec"
    # Container name flag and value are adjacent
    name_idx = argv.index("--name")
    assert argv[name_idx + 1] == "lab-sandbox-deadbeef"


def test_build_run_argv_gvisor_flags_only_for_runsc() -> None:
    runsc_argv = _build_run_argv(
        image="img",
        name="n",
        runtime="runsc",
        network="none",
        mem_limit="1g",
        cpu_limit=2.0,
        env=None,
    )
    assert "--runtime=runsc" in runsc_argv
    assert "--security-opt=label=disable" in runsc_argv
    assert "--runtime-flag=ignore-cgroups" in runsc_argv

    crun_argv = _build_run_argv(
        image="img",
        name="n",
        runtime="crun",
        network="none",
        mem_limit="1g",
        cpu_limit=2.0,
        env=None,
    )
    assert "--runtime=crun" in crun_argv
    # The gVisor workarounds must NOT leak into other runtimes.
    assert "--security-opt=label=disable" not in crun_argv
    assert "--runtime-flag=ignore-cgroups" not in crun_argv


def test_build_run_argv_passes_limits() -> None:
    argv = _build_run_argv(
        image="img",
        name="n",
        runtime="runsc",
        network="none",
        mem_limit="512m",
        cpu_limit=0.5,
        env=None,
    )
    assert "--memory=512m" in argv
    assert "--cpus=0.5" in argv


def test_build_run_argv_env_passed_sorted() -> None:
    argv = _build_run_argv(
        image="img",
        name="n",
        runtime="runsc",
        network="none",
        mem_limit="1g",
        cpu_limit=2.0,
        env={"FOO": "bar", "BAZ": "qux"},
    )
    # Expect both env entries, sorted (BAZ before FOO).
    idx_baz = argv.index("BAZ=qux")
    idx_foo = argv.index("FOO=bar")
    assert idx_baz < idx_foo
    # And each is preceded by `--env`.
    assert argv[idx_baz - 1] == "--env"
    assert argv[idx_foo - 1] == "--env"


def test_build_run_argv_network_modes() -> None:
    for mode in ("none", "host"):
        argv = _build_run_argv(
            image="img",
            name="n",
            runtime="runsc",
            network=mode,
            mem_limit="1g",
            cpu_limit=2.0,
            env=None,
        )
        assert f"--network={mode}" in argv


def test_sandbox_rejects_invalid_network() -> None:
    with pytest.raises(ValueError, match="network must be"):
        Sandbox(network="public")  # type: ignore[arg-type]


def test_sandbox_list_network_uses_bridge_with_allow_list() -> None:
    """List-mode joins the default bridge and stores the allow-list for /etc/hosts."""

    sb = Sandbox(network=["example.com"])
    assert sb._network_arg == "podman"
    assert sb._allowed_hosts == ["example.com"]


def test_sandbox_empty_list_network_falls_back_to_none() -> None:
    """An empty allow-list is unambiguously "no network", not "open bridge"."""

    sb = Sandbox(network=[])
    assert sb._network_arg == "none"
    assert sb._allowed_hosts == []


def test_sandbox_default_network_is_none() -> None:
    sb = Sandbox()
    assert sb._network_arg == "none"
    assert sb._allowed_hosts == []


def test_build_run_argv_add_hosts_and_dns() -> None:
    argv = _build_run_argv(
        image="img",
        name="n",
        runtime="runsc",
        network="podman",
        mem_limit="1g",
        cpu_limit=2.0,
        env=None,
        add_hosts=[("example.com", "1.2.3.4"), ("b.example", "5.6.7.8")],
        dns_servers=["127.0.0.1"],
    )
    assert "--add-host=example.com:1.2.3.4" in argv
    assert "--add-host=b.example:5.6.7.8" in argv
    assert "--dns=127.0.0.1" in argv


def test_sandbox_unique_container_names() -> None:
    a = Sandbox()
    b = Sandbox()
    assert a.container_name != b.container_name
    assert a.container_name.startswith("lab-sandbox-")


def test_build_run_argv_kb_mount_added_when_set() -> None:
    """kb_query needs read-only access to the host KB root inside the sandbox."""
    from pathlib import Path

    argv = _build_run_argv(
        image="img",
        name="n",
        runtime="runsc",
        network="none",
        mem_limit="1g",
        cpu_limit=2.0,
        env=None,
        kb_root_mount=Path("/home/m/db/kb"),
    )
    assert "-v=/home/m/db/kb:/kb:ro" in argv


def test_build_run_argv_no_kb_mount_by_default() -> None:
    argv = _build_run_argv(
        image="img",
        name="n",
        runtime="runsc",
        network="none",
        mem_limit="1g",
        cpu_limit=2.0,
        env=None,
    )
    assert not any(arg.startswith("-v=") for arg in argv)


def test_sandbox_accepts_kb_root_mount_as_path() -> None:
    from pathlib import Path

    sb = Sandbox(kb_root_mount=Path("/some/kb"))
    assert sb.kb_root_mount == Path("/some/kb")
    assert sb.kb_mount_target == "/kb"
