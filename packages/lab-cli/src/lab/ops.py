"""NS-5 Homelab Ops v0 (ADR-012 LAR + ADR-013 authz) — gated INFRA vertical.

A thin caller of the Lab Agent Runtime (``lab.core.agent_runtime.run_agent``),
mirroring the analyst / comms verticals, for the lab's HIGHEST blast-radius
surface: an agent that inspects the *live* homelab (m-box, ~40 real user
services) and proposes remediations such as a service restart.

THE KEY GUARANTEE (NS-5's whole reason to exist)
------------------------------------------------
**The ops agent can DIAGNOSE freely but cannot MUTATE infrastructure without
explicit human approval — the default is fail-closed deny.** This is what makes
an infra-touching agent safe, and it is enforced by the runtime, not by this
module or the LLM:

  * the DIAGNOSTIC tools (``ops_disk`` / ``ops_gpu`` / ``ops_services`` /
    ``ops_containers`` / ``ops_lease``) are ``side_effect="read"`` and so
    auto-execute under the ADR-013 default policy — the agent can look at
    everything;
  * the only MUTATING tool, ``ops_restart_service``, is ``side_effect=
    "irreversible"`` (a restart disrupts a live service and cannot be cleanly
    undone), so it resolves to ``require_approval`` and — with the default
    fail-closed approver (``lab.core.authz.deny_approver``, returns False) — is
    DENIED before the impl is ever called.

So:

    default authz + no approver         -> restart PROPOSED, NEVER executed
    default authz + approver -> True     -> restart executed (the gate opens)

``diagnose`` therefore returns the *proposed* remediations rather than firing
them: ``applied`` is empty in v0 (and in tests) because no approver authorizes a
restart. The real ``systemctl restart`` subprocess lives behind a
``# pragma: no cover`` seam in ``lab.ops_tools`` and is mocked / spied in tests,
so NO real command — and critically no real restart — ever runs.
"""

from __future__ import annotations

from typing import Any

from lab.core.agent_runtime import run_agent
from lab.core.authz import ApprovalCallback, Authorizer, default_authorizer
from lab.core.settings import get_settings
from lab.ops_tools import OPS_ACTOR, OpsTools

# --------------------------------------------------------------------------- #
# Approvers (fail-closed by default; explicit, scoped openers for the gate).   #
# --------------------------------------------------------------------------- #


def approve_class(*classes: str) -> ApprovalCallback:
    """An approver that approves ONLY the named side-effect classes and denies
    everything else (still fail-closed for anything not listed). Used to prove the
    gate opens for an explicitly-approved class while staying shut for the rest —
    e.g. ``approve_class("irreversible")`` authorizes a restart, nothing else."""
    allowed = frozenset(classes)

    def _approver(request: dict[str, Any]) -> bool:
        return request.get("side_effect") in allowed

    return _approver


_DIAGNOSE_SYSTEM = """You are the lab's homelab-ops agent for a LIVE homelab with
many real user services. INSPECT health using the read-only diagnostic tools
(disk, GPU, service status, containers, the GPU lease) — these auto-execute.

Based on what you find, PROPOSE remediations. The only action available is
ops_restart_service, which is IRREVERSIBLE: restarting a live service disrupts it
and cannot be cleanly undone, so it only fires on explicit human approval. Propose
a restart when warranted; do NOT assume it will be applied. Diagnose first, then
recommend — never restart speculatively."""


def _final_text(messages: list[dict[str, Any]]) -> str:
    """The final assistant message content (the agent's diagnosis / writeup)."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str) and content:
                return content
    return ""


def _findings(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The results of the read/diagnostic tool calls (everything the agent saw)."""
    read_tools = {"ops_disk", "ops_gpu", "ops_services", "ops_containers", "ops_lease"}
    return [r for r in results if r.get("name") in read_tools]


def _restart_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every ops_restart_service result (whether it fired or was denied)."""
    return [r for r in results if r.get("name") == "ops_restart_service"]


def _restart_applied(result: dict[str, Any]) -> bool:
    """True iff an ops_restart_service result indicates a restart ACTUALLY ran
    (``restarted: True``). A denied/blocked call returns an ``error`` instead, so
    this stays False unless a human approved the gate."""
    res = result.get("result")
    return isinstance(res, dict) and res.get("restarted") is True


def diagnose(
    *,
    focus: str = "all",
    model: str = "qwen3-4b-ft-toolcall-q4-latest",
    tools: OpsTools | None = None,
    authorizer: Authorizer | None = None,
    approver: ApprovalCallback | None = None,
    max_tool_calls: int = 12,
    timeout: int = 90,
    num_ctx: int | None = None,
) -> dict[str, Any]:
    """Inspect homelab health (read tools, auto) and PROPOSE remediations.

    With the ADR-013 defaults (``authorizer=None`` -> ``default_authorizer()``,
    ``approver=None`` -> fail-closed ``deny_approver``), any proposed
    ``ops_restart_service`` is IRREVERSIBLE and therefore resolves to
    require_approval -> deny: the restart is returned in ``proposed_actions`` and
    NEVER fires, so ``applied`` stays empty. Pass an ``approver`` that returns True
    for the ``irreversible`` class (e.g. ``approve_class("irreversible")``) to
    actually apply a restart — the gate works both ways.

    Args:
        focus: a free-form hint for what to inspect (e.g. "disk", "gpu", "all").
        model / timeout / num_ctx / max_tool_calls: passed to ``run_agent``.
        tools: an ``OpsTools`` (inject spy runners in tests so no real command
            runs); defaults to the real read-only runners.
        authorizer / approver: ADR-013 gate wiring; defaults are fail-closed.

    Returns ``{"findings", "proposed_actions", "applied", "tool_calls", "stop"}``,
    where ``findings`` are the read-tool observations, ``proposed_actions`` are the
    restarts the agent proposed, and ``applied`` lists only restarts that an
    approver actually authorized (empty under the v0 default).
    """
    settings = get_settings()
    ot = tools or OpsTools()
    authz = authorizer or default_authorizer()
    res = run_agent(
        settings=settings,
        litellm_key=settings.litellm_key,
        model=model,
        system=_DIAGNOSE_SYSTEM,
        user=(
            f"Diagnose homelab health (focus: {focus!r}). Inspect with the "
            "read-only tools, then propose any warranted service restart "
            "(ops_restart_service). Do not assume a restart is applied."
        ),
        tools=ot.build_tools(),
        actor=OPS_ACTOR,
        authorizer=authz,
        approver=approver,  # None -> runtime uses fail-closed deny_approver
        max_turns=max_tool_calls,
        max_tool_calls=max_tool_calls,
        timeout=timeout,
        num_ctx=num_ctx,
    )

    restart_results = _restart_results(res.tool_results)
    proposed_actions: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    for r in restart_results:
        unit = (r.get("args") or {}).get("unit")
        action = {"action": "restart_service", "unit": unit}
        proposed_actions.append(action)
        if _restart_applied(r):
            applied.append(action)

    return {
        "summary": _final_text(res.messages),
        "findings": _findings(res.tool_results),
        "proposed_actions": proposed_actions,
        "applied": applied,
        "tool_calls": res.tool_calls,
        "stop": res.stop_reason,
    }


__all__ = ["approve_class", "diagnose"]
