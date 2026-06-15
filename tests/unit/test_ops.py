"""NS-5 Homelab Ops v0 — gated INFRA vertical (ADR-012 LAR + ADR-013 authz).

ASSERT NOTHING MUTATES. No GPU, no live LLM, NO real command, and CRITICALLY
NO real service restart / stop / destructive action in ANY code path a test
reaches.

The whole point of NS-5 is the safety gate: the ops agent can DIAGNOSE freely
(read tools auto-execute) but cannot MUTATE infra (a service restart) without
explicit human approval — default fail-closed deny. We prove this two ways:

  1. RUNTIME-GATE proof (load-bearing): drive the REAL ``run_agent`` with
     ``call_litellm_chat`` stubbed to a scripted trajectory that calls a read tool
     (returns synthetic health) AND attempts ``ops_restart_service``. Every tool
     runs through a SPY ``CommandRunner`` (so no real command executes) and the
     restart goes through a SPY restart runner (so the live ``systemctl restart``
     seam is never touched). We assert:
       * under default authz with NO approver (fail-closed) the restart impl is
         reached ZERO times and the restart-spy records ZERO restarts;
       * even with an explicit irreversible grant set it stays denied (never auto
         in v0);
       * WITH an approving callback the SAME restart IS executed (the gate works
         both ways) — but still through the SPY, so no real service is touched.

  2. WIRING proof: mock ``lab.ops.run_agent`` and assert ``diagnose`` reports
     findings + proposed_actions and keeps ``applied`` empty by default, and that
     it surfaces an applied restart only when an approver authorizes it.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from lab.platform.agent_runtime import AgentResult, run_agent
from lab.platform.authz import AuthzPolicy, default_authorizer

from lab.ops import approve_class, diagnose
from lab.ops_tools import CommandResult, OpsTools

# --------------------------------------------------------------------------- #
# Helpers: a scripted litellm + spy command runners (never run a real command).#
# --------------------------------------------------------------------------- #


def _scripted_litellm(tool_calls: list[dict[str, Any]]) -> Any:
    """One assistant turn emitting `tool_calls`, then a stop turn."""
    turns: list[dict[str, Any]] = [
        {"choices": [{"message": {"role": "assistant", "tool_calls": tool_calls}}]},
        {"choices": [{"message": {"role": "assistant", "content": "diagnosis done"}}]},
    ]
    state = {"i": 0}

    def _call(**_kwargs: Any) -> tuple[dict[str, Any], int]:
        i = min(state["i"], len(turns) - 1)
        state["i"] += 1
        return turns[i], 1

    return _call


def _tc(name: str, args: str = "{}", cid: str = "c1") -> dict[str, Any]:
    return {"id": cid, "function": {"name": name, "arguments": args}}


class _RunnerSpy:
    """Stands in for subprocess; records argv, NEVER runs a real command.

    Returns a benign synthetic CommandResult so read tools yield fake health and a
    (hypothetically approved) restart yields a success-shaped result — without
    ever shelling out.
    """

    def __init__(self) -> None:
        self.argvs: list[list[str]] = []

    def __call__(self, argv: list[str]) -> CommandResult:
        self.argvs.append(argv)
        return CommandResult(returncode=0, stdout="SYNTHETIC", stderr="")


def _spy_tools() -> tuple[OpsTools, _RunnerSpy, _RunnerSpy]:
    """An OpsTools wired to spy runners for BOTH reads and restarts."""
    read_spy = _RunnerSpy()
    restart_spy = _RunnerSpy()
    ot = OpsTools(read_runner=read_spy, restart_runner=restart_spy)
    return ot, read_spy, restart_spy


def _drive(
    tool_calls: list[dict[str, Any]],
    ot: OpsTools,
    **agent_kwargs: Any,
) -> AgentResult:
    """Run the REAL run_agent with a scripted model over ot's tools."""
    from lab.platform import agent_runtime

    with (
        patch.object(agent_runtime, "call_litellm_chat", _scripted_litellm(tool_calls)),
        patch.object(agent_runtime, "record_action", return_value="h"),
    ):
        return run_agent(
            settings=object(),  # type: ignore[arg-type]
            litellm_key="k",
            model="m",
            system="s",
            user="u",
            tools=ot.build_tools(),
            actor="ops",
            **agent_kwargs,
        )


# --------------------------------------------------------------------------- #
# 1. RUNTIME-GATE proof — a restart cannot fire without explicit approval.      #
# --------------------------------------------------------------------------- #


def test_restart_default_authz_no_approver_does_not_fire() -> None:
    """THE GUARANTEE: default authz + no approver -> restart DENIED, impl never
    reached, ZERO real restarts recorded."""
    ot, read_spy, restart_spy = _spy_tools()
    res = _drive(
        [
            _tc("ops_disk", "{}", "c1"),
            _tc("ops_restart_service", '{"unit": "lab-litellm.service"}', "c2"),
        ],
        ot,
        authorizer=default_authorizer(),  # no approver -> fail-closed deny
    )
    # The read tool ran (diagnose freely); the restart impl was NEVER reached.
    assert "ops_disk" in ot.calls
    assert "ops_restart_service" not in ot.calls
    # PROOF nothing mutated: zero restarts attempted, restart spy never invoked.
    assert ot.restarts == []
    assert restart_spy.argvs == []
    # The read command did run through the spy (no real subprocess).
    assert ["df", "-h"] in read_spy.argvs
    by_name = {r["name"]: r["result"] for r in res.tool_results}
    assert "denied" in str(by_name["ops_restart_service"]).lower()


def test_restart_blocked_even_with_explicit_grant() -> None:
    """ADR-013: irreversible is NEVER auto, even with an explicit grant set."""
    ot, _read_spy, restart_spy = _spy_tools()
    granted = AuthzPolicy(grants={("ops", "irreversible")})
    _drive(
        [_tc("ops_restart_service", '{"unit": "mlflow.service"}')],
        ot,
        authorizer=granted,  # grant present, still require_approval -> deny
    )
    assert "ops_restart_service" not in ot.calls
    assert ot.restarts == []
    assert restart_spy.argvs == []


def test_restart_fires_only_with_approving_callback() -> None:
    """The gate works BOTH ways: an explicit approval lets the restart through —
    but still via the SPY runner, so no real service is touched."""
    ot, _read_spy, restart_spy = _spy_tools()
    res = _drive(
        [_tc("ops_restart_service", '{"unit": "lab-litellm.service"}')],
        ot,
        authorizer=default_authorizer(),
        approver=approve_class("irreversible"),  # explicit approval opens the gate
    )
    assert ot.calls == ["ops_restart_service"]
    assert ot.restarts == ["lab-litellm.service"]
    # Executed through the SPY only — the live systemctl seam never ran.
    assert restart_spy.argvs == [["systemctl", "restart", "lab-litellm.service"]]
    assert res.tool_results[0]["result"]["restarted"] is True


def test_read_tools_auto_execute_and_never_restart() -> None:
    """All four+ diagnostic tools auto-allow under default authz and touch nothing
    mutating — they only run read-only commands through the spy."""
    ot, read_spy, restart_spy = _spy_tools()
    _drive(
        [
            _tc("ops_disk", "{}", "c1"),
            _tc("ops_gpu", "{}", "c2"),
            _tc("ops_services", '{"unit": "sshd.service"}', "c3"),
            _tc("ops_containers", "{}", "c4"),
            _tc("ops_lease", "{}", "c5"),
        ],
        ot,
        authorizer=default_authorizer(),  # no approver needed for reads
    )
    assert ot.calls == ["ops_disk", "ops_gpu", "ops_services", "ops_containers", "ops_lease"]
    # Nothing mutating happened.
    assert ot.restarts == []
    assert restart_spy.argvs == []
    # Read commands are inspect-only heads.
    heads = {argv[0] for argv in read_spy.argvs}
    assert heads == {"df", "nvidia-smi", "systemctl", "podman", "redis-cli"}
    # The only systemctl read sub-verb is the status-only is-active (NOT restart).
    systemctl_argvs = [a for a in read_spy.argvs if a[0] == "systemctl"]
    assert systemctl_argvs == [["systemctl", "is-active", "sshd.service"]]


def test_services_rejects_injection_shaped_unit_without_shelling_out() -> None:
    """A malformed / injection-shaped unit name is refused before any argv runs."""
    ot, read_spy, _restart_spy = _spy_tools()
    _drive(
        [_tc("ops_services", '{"unit": "evil; rm -rf /"}')],
        ot,
        authorizer=default_authorizer(),
    )
    # No command shelled out for the rejected unit.
    assert read_spy.argvs == []


def test_restart_rejects_injection_shaped_unit_when_approved() -> None:
    """Even on the approved path, a bad unit name is refused before any restart."""
    ot, _read_spy, restart_spy = _spy_tools()
    res = _drive(
        [_tc("ops_restart_service", '{"unit": "bad name & boom"}')],
        ot,
        authorizer=default_authorizer(),
        approver=approve_class("irreversible"),
    )
    assert ot.restarts == []  # never recorded
    assert restart_spy.argvs == []  # never shelled out
    assert res.tool_results[0]["result"]["restarted"] is False


def test_tool_side_effect_classes_are_correct() -> None:
    by_name = {t.name: t for t in OpsTools().build_tools()}
    # Diagnostic tools are read (auto).
    assert by_name["ops_disk"].side_effect == "read"
    assert by_name["ops_gpu"].side_effect == "read"
    assert by_name["ops_services"].side_effect == "read"
    assert by_name["ops_containers"].side_effect == "read"
    assert by_name["ops_lease"].side_effect == "read"
    # The single mutating tool is irreversible (gated to deny by default).
    assert by_name["ops_restart_service"].side_effect == "irreversible"


def test_lease_read_parses_holder() -> None:
    """ops_lease reports the holder from a synthetic redis-cli GET, read-only."""

    class _LeaseRunner:
        def __call__(self, argv: list[str]) -> CommandResult:
            return CommandResult(returncode=0, stdout="m-box:1234:sweep:abcd\n", stderr="")

    ot = OpsTools(read_runner=_LeaseRunner())
    out = ot.ops_lease()
    assert out["held"] is True
    assert out["holder"] == "m-box:1234:sweep:abcd"


# --------------------------------------------------------------------------- #
# 2. WIRING proof — diagnose reports findings/proposals; nothing applied by     #
#    default; applied only when an approver authorizes.                          #
# --------------------------------------------------------------------------- #


def test_diagnose_default_proposes_restart_but_applies_nothing(monkeypatch: Any) -> None:
    """With default authz + no approver, diagnose replays the real runtime gate:
    the restart is denied, so it shows up in proposed_actions but applied stays
    empty and the restart spy records zero restarts."""
    ot, _read_spy, restart_spy = _spy_tools()

    def fake_run_agent(**kwargs: Any) -> AgentResult:
        return _drive(
            [
                _tc("ops_disk", "{}", "c1"),
                _tc("ops_restart_service", '{"unit": "lab-litellm.service"}', "c2"),
            ],
            ot,
            authorizer=kwargs["authorizer"],
            approver=kwargs.get("approver"),
        )

    monkeypatch.setattr("lab.ops.run_agent", fake_run_agent)
    out = diagnose(focus="all")
    assert any(f["name"] == "ops_disk" for f in out["findings"])  # diagnosed freely
    assert out["proposed_actions"] == [{"action": "restart_service", "unit": "lab-litellm.service"}]
    assert out["applied"] == []  # NOTHING applied under the default fail-closed gate
    assert ot.restarts == []
    assert restart_spy.argvs == []


def test_diagnose_applies_restart_only_with_approval(monkeypatch: Any) -> None:
    """With an irreversible-approving callback, the same restart is applied —
    through the SPY runner, so no real service is touched."""
    ot, _read_spy, restart_spy = _spy_tools()

    def fake_run_agent(**kwargs: Any) -> AgentResult:
        return _drive(
            [_tc("ops_restart_service", '{"unit": "lab-litellm.service"}')],
            ot,
            authorizer=kwargs["authorizer"],
            approver=kwargs.get("approver"),
        )

    monkeypatch.setattr("lab.ops.run_agent", fake_run_agent)
    out = diagnose(approver=approve_class("irreversible"))
    assert out["applied"] == [{"action": "restart_service", "unit": "lab-litellm.service"}]
    assert ot.restarts == ["lab-litellm.service"]
    # Still only via the spy — no live systemctl ran.
    assert restart_spy.argvs == [["systemctl", "restart", "lab-litellm.service"]]


def test_diagnose_no_proposal_when_only_reads(monkeypatch: Any) -> None:
    """A clean diagnosis with no restart proposed -> empty proposed_actions."""
    ot, _read_spy, restart_spy = _spy_tools()

    def fake_run_agent(**kwargs: Any) -> AgentResult:
        return _drive(
            [_tc("ops_disk", "{}", "c1"), _tc("ops_gpu", "{}", "c2")],
            ot,
            authorizer=kwargs["authorizer"],
            approver=kwargs.get("approver"),
        )

    monkeypatch.setattr("lab.ops.run_agent", fake_run_agent)
    out = diagnose()
    assert out["proposed_actions"] == []
    assert out["applied"] == []
    assert len(out["findings"]) == 2
    assert restart_spy.argvs == []
