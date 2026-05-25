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


def test_sandbox_list_network_degrades_to_none() -> None:
    """v0.1: a list of allowed hosts isn't wired up yet; degrade to none."""

    sb = Sandbox(network=["example.com"])
    assert sb._network_arg == "none"


def test_sandbox_default_network_is_none() -> None:
    sb = Sandbox()
    assert sb._network_arg == "none"


def test_sandbox_unique_container_names() -> None:
    a = Sandbox()
    b = Sandbox()
    assert a.container_name != b.container_name
    assert a.container_name.startswith("lab-sandbox-")
