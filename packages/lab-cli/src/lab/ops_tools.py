"""NS-5 Homelab Ops v0 — tool implementations (ADR-012 Tool ABI + ADR-013 authz).

NS-5 is the lab's HIGHEST blast-radius vertical: its tools touch the *live*
homelab on m-box (~40 real user services). Safety gating is therefore the whole
point of this slice — see the module-level guarantee in ``lab.ops``.

The tools split cleanly into two ADR-013 side-effect classes:

    ops_disk        -> read          (df: list filesystem usage)
    ops_gpu         -> read          (nvidia-smi --query: read GPU telemetry)
    ops_services    -> read          (systemctl is-active / podman ps: status)
    ops_lease       -> read          (redis-cli GET the Valkey GPU lease)
    ops_restart_service -> irreversible  (systemctl restart: disrupts a live svc)

Only DIAGNOSTIC (read) tools auto-execute. The single MUTATING tool,
``ops_restart_service``, is ``irreversible`` — a service restart cannot be cleanly
undone and can disrupt the homelab — so under the ADR-013 default policy it
resolves to require_approval and, with the fail-closed default approver, is
DENIED. It can only fire on explicit human approval.

Safety properties enforced HERE (not by the LLM):

  * READ-ONLY COMMAND ALLOWLIST — every read tool shells out to a FIXED, inspect-
    only command (``df`` / ``nvidia-smi`` / ``systemctl is-active|status`` /
    ``podman ps`` / ``redis-cli GET``). The argv is built HERE from a constant
    template plus validated arguments; nothing the model says becomes a new
    program head. ``shell=False`` (argv list, no shell metacharacters), so there
    is no command-injection surface.
  * UNIT-NAME CONFINEMENT — ``ops_restart_service`` validates the unit name
    against a conservative pattern before it would ever build a restart argv, so
    a malformed / injection-shaped name is refused early. (It is gated to deny by
    default regardless; this is defense-in-depth.)
  * INJECTABLE RUNNERS — every tool runs its command through an injectable
    ``CommandRunner`` (default = a real, read-only ``subprocess.run``). Tests
    inject a spy runner, so NO real command — and CRITICALLY no real restart —
    ever executes under test.

The real restart subprocess lives behind ``# pragma: no cover`` and is never
reached from a test: the default authz gate denies it, and even an explicitly
approving test injects a SPY runner so the live ``systemctl restart`` body never
runs.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from lab.platform.agent_runtime import Tool

#: The actor identity the ops agent runs under (matches the authz decision key).
OPS_ACTOR = "ops"

#: Cap captured output so one tool result cannot blow the context budget.
_OUTPUT_CAP = 8000

#: Conservative systemd unit-name pattern. A unit is letters/digits/[-._@] plus
#: an optional ``.<type>`` suffix; this rejects whitespace, shell metacharacters,
#: path separators and injection-shaped input before any argv is built.
_UNIT_RE = re.compile(r"^[A-Za-z0-9_.@\\-]+(\.(service|socket|timer|target|path))?$")

#: The Valkey key the GPU lease lives under (mirrors lab.core.gpu_lease.LEASE_KEY,
#: re-declared here to avoid importing the redis-dependent module).
LEASE_KEY = "lab:gpu:lease"


# --------------------------------------------------------------------------- #
# Command runner seam (real read-only subprocess by default; spy in tests).    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CommandResult:
    """The captured outcome of a shelled-out command."""

    returncode: int
    stdout: str
    stderr: str


#: A command runner: argv -> CommandResult. The production default
#: (``subprocess_runner``) runs argv with ``shell=False`` and captures output.
#: Tests inject a spy so no real command (and no real restart) ever executes.
CommandRunner = Callable[[list[str]], CommandResult]


def subprocess_runner(argv: list[str]) -> CommandResult:
    """Production runner: execute ``argv`` with ``shell=False`` and capture output.

    Used for the READ-ONLY diagnostic commands. ``shell=False`` means there is no
    shell to interpret metacharacters — the argv is taken literally. Tests always
    inject a spy runner instead, so this never runs under test.
    """
    try:
        proc = subprocess.run(  # argv is built from constants, shell=False
            argv,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(returncode=-1, stdout="", stderr="command timed out")
    except FileNotFoundError as exc:
        return CommandResult(returncode=-1, stdout="", stderr=f"command not found: {exc}")
    return CommandResult(
        returncode=proc.returncode,
        stdout=proc.stdout[:_OUTPUT_CAP],
        stderr=proc.stderr[:_OUTPUT_CAP],
    )


def restart_subprocess_runner(argv: list[str]) -> CommandResult:  # pragma: no cover
    """LIVE restart seam — runs ``systemctl restart <unit>`` for real.

    NEVER reached from a test: ``ops_restart_service`` is ``irreversible`` so the
    ADR-013 default gate denies it (the runtime never calls the impl), and tests
    that exercise the *approved* path inject a SPY runner, so this body — the only
    code that could mutate a real service — never executes. It is fenced off with
    ``# pragma: no cover`` for exactly this reason.
    """
    return subprocess_runner(argv)


# --------------------------------------------------------------------------- #
# Tool context: holds the runners and exposes the five tool impls.             #
# --------------------------------------------------------------------------- #


@dataclass
class OpsTools:
    """Holds the ops command runners and exposes the five NS-5 tool impls.

    Two injectable seams:

      * ``read_runner``    — runs the READ-ONLY diagnostic commands.
      * ``restart_runner`` — runs the (irreversible) restart command. The default
        is the ``# pragma: no cover`` live seam; tests inject a SPY so no real
        restart can ever fire.

    ``calls`` is a spy log: each impl appends its name BEFORE doing anything, so a
    test can assert e.g. ``"ops_restart_service" not in tools.calls`` when the gate
    denies it (the runtime never reaches ``impl`` for a denied call). ``restarts``
    separately records every unit a restart was *actually attempted* on — the
    load-bearing proof that the count is zero unless a human approved.
    """

    read_runner: CommandRunner = subprocess_runner
    restart_runner: CommandRunner = restart_subprocess_runner
    calls: list[str] = field(default_factory=list)
    restarts: list[str] = field(default_factory=list)
    lease_key: str = LEASE_KEY

    # ----- read / diagnostic ------------------------------------------------
    def ops_disk(self) -> dict[str, Any]:
        """READ: filesystem usage via ``df -h`` (human-readable). Inspect-only."""
        self.calls.append("ops_disk")
        res = self.read_runner(["df", "-h"])
        return {"returncode": res.returncode, "stdout": res.stdout, "stderr": res.stderr}

    def ops_gpu(self) -> dict[str, Any]:
        """READ: GPU telemetry via ``nvidia-smi --query-gpu=... --format=csv``.

        A query-only nvidia-smi invocation — it reads utilisation / memory and
        never changes GPU state. Inspect-only.
        """
        self.calls.append("ops_gpu")
        res = self.read_runner(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader",
            ]
        )
        return {"returncode": res.returncode, "stdout": res.stdout, "stderr": res.stderr}

    def ops_services(self, unit: str) -> dict[str, Any]:
        """READ: service status via ``systemctl is-active <unit>``. Status-only.

        The unit name is validated against ``_UNIT_RE`` before the argv is built,
        so a malformed / injection-shaped name is refused without shelling out.
        ``is-active`` only reports state; it never starts/stops/restarts anything.
        """
        self.calls.append("ops_services")
        if not _UNIT_RE.match(unit):
            return {"error": f"invalid unit name: {unit!r}"}
        res = self.read_runner(["systemctl", "is-active", unit])
        return {
            "unit": unit,
            "returncode": res.returncode,
            "active": res.stdout.strip(),
            "stderr": res.stderr,
        }

    def ops_containers(self) -> dict[str, Any]:
        """READ: container status via ``podman ps`` (running containers). List-only."""
        self.calls.append("ops_containers")
        res = self.read_runner(["podman", "ps", "--format", "{{.Names}}\t{{.Status}}"])
        return {"returncode": res.returncode, "stdout": res.stdout, "stderr": res.stderr}

    def ops_lease(self) -> dict[str, Any]:
        """READ: the Valkey GPU lease via ``redis-cli GET <key>``. Get-only.

        Reads the current lease holder tag (or empty if free). ``GET`` is a pure
        read; nothing about the lease is mutated.
        """
        self.calls.append("ops_lease")
        res = self.read_runner(["redis-cli", "GET", self.lease_key])
        holder = res.stdout.strip()
        return {
            "returncode": res.returncode,
            "lease_key": self.lease_key,
            "holder": holder or None,
            "held": bool(holder),
            "stderr": res.stderr,
        }

    # ----- irreversible (gated to deny by default) --------------------------
    def ops_restart_service(self, unit: str) -> dict[str, Any]:
        """IRREVERSIBLE: ``systemctl restart <unit>`` — disrupts a LIVE service.

        Reaching this impl at all already required passing the ADR-013
        irreversible gate (require_approval -> explicit human approval). Under the
        default fail-closed approver the runtime NEVER calls this. The unit name is
        re-validated here (defense-in-depth) before any restart argv is built; the
        actual mutation goes through ``restart_runner`` (the ``# pragma: no cover``
        live seam in production, a SPY in tests).
        """
        self.calls.append("ops_restart_service")
        if not _UNIT_RE.match(unit):
            return {"restarted": False, "error": f"invalid unit name: {unit!r}"}
        self.restarts.append(unit)
        res = self.restart_runner(["systemctl", "restart", unit])
        return {
            "restarted": res.returncode == 0,
            "unit": unit,
            "returncode": res.returncode,
            "stderr": res.stderr,
        }

    # ----- Tool ABI wiring --------------------------------------------------
    def build_tools(self) -> list[Tool]:
        """The five NS-5 tools as ADR-012 Tool ABI instances, each carrying the
        ADR-013 side-effect class that drives the authorization gate.

        Four READ tools (auto-execute) + one IRREVERSIBLE mutating tool
        (``ops_restart_service``), which is gated to deny by default.
        """
        no_args: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
        unit_arg: dict[str, Any] = {
            "type": "object",
            "properties": {"unit": {"type": "string"}},
            "required": ["unit"],
        }
        return [
            Tool(
                name="ops_disk",
                description=(
                    "Report filesystem usage (df -h). Read-only diagnostic; auto-executes."
                ),
                parameters=no_args,
                impl=self.ops_disk,
                side_effect="read",
                capability="ops_inspect",
            ),
            Tool(
                name="ops_gpu",
                description=(
                    "Report GPU utilisation and memory (nvidia-smi query). "
                    "Read-only diagnostic; auto-executes."
                ),
                parameters=no_args,
                impl=self.ops_gpu,
                side_effect="read",
                capability="ops_inspect",
            ),
            Tool(
                name="ops_services",
                description=(
                    "Report whether a systemd unit is active (systemctl is-active "
                    "<unit>). Read-only diagnostic; auto-executes."
                ),
                parameters=unit_arg,
                impl=self.ops_services,
                side_effect="read",
                capability="ops_inspect",
            ),
            Tool(
                name="ops_containers",
                description=(
                    "List running containers and their status (podman ps). "
                    "Read-only diagnostic; auto-executes."
                ),
                parameters=no_args,
                impl=self.ops_containers,
                side_effect="read",
                capability="ops_inspect",
            ),
            Tool(
                name="ops_lease",
                description=(
                    "Read the current Valkey GPU lease holder (redis-cli GET). "
                    "Read-only diagnostic; auto-executes."
                ),
                parameters=no_args,
                impl=self.ops_lease,
                side_effect="read",
                capability="ops_inspect",
            ),
            Tool(
                name="ops_restart_service",
                description=(
                    "Restart a systemd unit (systemctl restart <unit>). "
                    "IRREVERSIBLE — a restart disrupts a live service and cannot be "
                    "cleanly undone; requires explicit human approval, never auto "
                    "in v0."
                ),
                parameters=unit_arg,
                impl=self.ops_restart_service,
                side_effect="irreversible",
                capability="ops_restart",
            ),
        ]


__all__ = [
    "LEASE_KEY",
    "OPS_ACTOR",
    "CommandResult",
    "CommandRunner",
    "OpsTools",
    "restart_subprocess_runner",
    "subprocess_runner",
]
