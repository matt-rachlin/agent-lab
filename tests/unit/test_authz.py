"""ADR-013 action-authorization enforcement — policy tiers, ratchet, and the
backward-compatible LAR hook. Pure unit tests; everything is mocked (no DB/GPU)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import patch

from lab.core.agent_runtime import AgentResult, Tool
from lab.core.authz import (
    ApprovalCallback,
    Authorizer,
    AuthzPolicy,
    InMemoryRatchetStore,
    Ratchet,
    default_authorizer,
    deny_approver,
)

# --------------------------------------------------------------------------- #
# Tier policy (ADR-013 §2 defaults)
# --------------------------------------------------------------------------- #


def test_read_classes_auto_allow():
    p = default_authorizer()
    assert p.decide("agent", "fs_read", "read", "fs") == "allow"
    assert p.decide("agent", "web_get", "external_read", "net") == "allow"


def test_write_local_requires_approval_by_default():
    p = default_authorizer()
    assert p.decide("agent", "fs_write", "write_local", "fs") == "require_approval"


def test_write_local_auto_when_granted():
    p = AuthzPolicy(grants={("agent", "write_local")})
    assert p.decide("agent", "fs_write", "write_local", "fs") == "allow"
    # grant is per-actor: a different actor is still gated.
    assert p.decide("other", "fs_write", "write_local", "fs") == "require_approval"


def test_irreversible_requires_approval_and_never_auto():
    # Even an explicit grant cannot make irreversible auto in v0.
    p = AuthzPolicy(grants={("agent", "irreversible")})
    assert p.decide("agent", "send_email", "irreversible", "mail") == "require_approval"


def test_unknown_class_denies():
    p = default_authorizer()
    assert p.decide("agent", "weird", "teleport", "x") == "deny"


def test_force_dry_run_shadows_writes_but_not_reads():
    p = AuthzPolicy(force_dry_run=True)
    assert p.decide("agent", "fs_read", "read", "fs") == "allow"
    assert p.decide("agent", "fs_write", "write_local", "fs") == "dry_run"
    assert p.decide("agent", "send_email", "irreversible", "mail") == "dry_run"


# --------------------------------------------------------------------------- #
# Approval hook (fail-closed)
# --------------------------------------------------------------------------- #


def test_default_approver_denies():
    assert deny_approver({"tool": "fs_write"}) is False


# --------------------------------------------------------------------------- #
# Earned-autonomy ratchet (ADR-013 §3)
# --------------------------------------------------------------------------- #


def test_ratchet_flips_write_local_after_threshold():
    r = Ratchet(threshold=3, store=InMemoryRatchetStore())
    p = AuthzPolicy(ratchet=r, workflow="wf1")
    # Below threshold: still gated.
    r.record_clean("agent", "write_local", "wf1")
    r.record_clean("agent", "write_local", "wf1")
    assert p.decide("agent", "fs_write", "write_local", "fs") == "require_approval"
    # Third clean action crosses the threshold -> earned auto.
    r.record_clean("agent", "write_local", "wf1")
    assert r.is_ratcheted("agent", "write_local", "wf1")
    assert p.decide("agent", "fs_write", "write_local", "fs") == "allow"


def test_incident_resets_ratchet():
    r = Ratchet(threshold=2, store=InMemoryRatchetStore())
    p = AuthzPolicy(ratchet=r, workflow="wf1")
    r.record_clean("agent", "write_local", "wf1")
    r.record_clean("agent", "write_local", "wf1")
    assert p.decide("agent", "fs_write", "write_local", "fs") == "allow"
    # Any incident resets streak + revokes the ratchet.
    r.record_incident("agent", "write_local", "wf1")
    assert not r.is_ratcheted("agent", "write_local", "wf1")
    assert p.decide("agent", "fs_write", "write_local", "fs") == "require_approval"


def test_ratchet_is_per_workflow():
    r = Ratchet(threshold=1, store=InMemoryRatchetStore())
    r.record_clean("agent", "write_local", "wf1")
    assert r.is_ratcheted("agent", "write_local", "wf1")
    assert not r.is_ratcheted("agent", "write_local", "wf2")


def test_irreversible_never_ratchets():
    r = Ratchet(threshold=1, store=InMemoryRatchetStore())
    for _ in range(10):
        r.record_clean("agent", "irreversible", "wf1")
    assert not r.is_ratcheted("agent", "irreversible", "wf1")
    p = AuthzPolicy(ratchet=r, workflow="wf1")
    assert p.decide("agent", "send_email", "irreversible", "mail") == "require_approval"


# --------------------------------------------------------------------------- #
# LAR hook — enforcement + backward-compat
# --------------------------------------------------------------------------- #


def _stub_litellm(
    tool_calls: list[dict[str, Any]] | None,
) -> Callable[..., tuple[dict[str, Any], int]]:
    """Return one assistant turn with `tool_calls`, then an empty turn (stop)."""
    turns: list[dict[str, Any]] = [
        {"choices": [{"message": {"role": "assistant", "tool_calls": tool_calls or []}}]},
        {"choices": [{"message": {"role": "assistant", "content": "done"}}]},
    ]
    state = {"i": 0}

    def _call(**_kwargs: Any) -> tuple[dict[str, Any], int]:
        i = min(state["i"], len(turns) - 1)
        state["i"] += 1
        return turns[i], 1

    return _call


def _tool_call(name: str, args: str = "{}") -> dict[str, Any]:
    return {"id": "c1", "function": {"name": name, "arguments": args}}


def _make_tool(name: str, side_effect: str, spy: list[str]) -> Tool:
    def impl(**_kwargs: Any) -> dict[str, str]:
        spy.append(name)
        return {"ok": name}

    return Tool(
        name=name,
        description=name,
        parameters={"type": "object", "properties": {}},
        impl=impl,
        side_effect=side_effect,  # type: ignore[arg-type]
    )


def _run(
    tool: Tool,
    *,
    authorizer: Authorizer | None = None,
    allow: set[str] | None = None,
    approver: ApprovalCallback | None = None,
) -> AgentResult:
    from lab.core import agent_runtime

    kwargs: dict[str, Any] = {}
    if allow is not None:
        kwargs["allow_side_effects"] = allow
    if authorizer is not None:
        kwargs["authorizer"] = authorizer
    if approver is not None:
        kwargs["approver"] = approver
    with (
        patch.object(agent_runtime, "call_litellm_chat", _stub_litellm([_tool_call(tool.name)])),
        patch.object(agent_runtime, "record_action", return_value="h"),
    ):
        return agent_runtime.run_agent(
            settings=object(),  # type: ignore[arg-type]
            litellm_key="k",
            model="m",
            system="s",
            user="u",
            tools=[tool],
            **kwargs,
        )


def test_backward_compat_authorizer_none_uses_old_gate():
    # write_local not in the (default) allow-set -> blocked, exactly as before.
    spy: list[str] = []
    tool = _make_tool("fs_write", "write_local", spy)
    res = _run(tool)  # authorizer=None
    assert spy == []  # never executed
    assert "blocked" in str(res.tool_results[0]["result"])


def test_backward_compat_allow_set_executes_read():
    spy: list[str] = []
    tool = _make_tool("fs_read", "read", spy)
    res = _run(tool)
    assert spy == ["fs_read"]
    assert res.tool_results[0]["result"] == {"ok": "fs_read"}


def test_authorizer_allow_executes():
    spy: list[str] = []
    tool = _make_tool("fs_read", "read", spy)
    res = _run(tool, authorizer=default_authorizer())
    assert spy == ["fs_read"]
    assert res.tool_results[0]["result"] == {"ok": "fs_read"}


def test_authorizer_require_approval_approver_yes_executes():
    spy: list[str] = []
    tool = _make_tool("fs_write", "write_local", spy)
    res = _run(tool, authorizer=default_authorizer(), approver=lambda _req: True)
    assert spy == ["fs_write"]
    assert res.tool_results[0]["result"] == {"ok": "fs_write"}


def test_authorizer_require_approval_approver_no_denies():
    spy: list[str] = []
    tool = _make_tool("fs_write", "write_local", spy)
    res = _run(tool, authorizer=default_authorizer(), approver=lambda _req: False)
    assert spy == []  # never executed
    assert "denied" in str(res.tool_results[0]["result"]).lower()


def test_authorizer_require_approval_default_is_fail_closed():
    # No approver wired -> fail-closed deny.
    spy: list[str] = []
    tool = _make_tool("fs_write", "write_local", spy)
    _run(tool, authorizer=default_authorizer())
    assert spy == []


def test_authorizer_dry_run_shadows_without_executing():
    spy: list[str] = []
    tool = _make_tool("fs_write", "write_local", spy)
    res = _run(tool, authorizer=AuthzPolicy(force_dry_run=True))
    assert spy == []  # shadow: real impl never called
    assert res.tool_results[0]["result"].get("dry_run") is True


def test_authorizer_deny_refuses():
    spy: list[str] = []
    tool = _make_tool("weird", "teleport", spy)
    res = _run(tool, authorizer=default_authorizer())
    assert spy == []
    assert "denied" in str(res.tool_results[0]["result"]).lower()
