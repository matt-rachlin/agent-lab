"""NS-4 Comms/Digest v0 (ADR-012 LAR + ADR-013 authz) — gated SEND vertical.

A thin caller of the Lab Agent Runtime (``lab.platform.agent_runtime.run_agent``),
mirroring the analyst/scout verticals, that adds the lab's first **send/push**
path. Two entrypoints:

  * ``digest``  — compose a short lab digest (scout rec count + a status line)
    and *propose* an ``ntfy_push`` to the operator's phone.
  * ``triage``  — read the inbox (dry-run stub; no live Gmail), classify, *draft*
    a reply, and *propose* an ``email_send``.

THE KEY PROPERTY (NS-4's eval signal / anti-footgun guarantee)
--------------------------------------------------------------
**No irreversible action (``email_send``, ``ntfy_push``) fires without an explicit
approval; the default is fail-closed deny.** This is enforced by the runtime, not
by this module: irreversible tools resolve to ``require_approval`` under the
ADR-013 default policy (``lab.platform.authz.default_authorizer``), and an unwired
``approver`` defaults to ``lab.platform.authz.deny_approver`` (returns False). So:

    default authz + no approver         -> irreversible proposed, NEVER executed
    default authz + approver -> True     -> irreversible executed (gate opens)

``write_local`` (``email_draft``) likewise needs approval by default, so triage
runs with an explicit "drafts are allowed" approver while STILL leaving the
irreversible ``email_send`` for a *separate*, opt-in approval — a draft-yes /
send-no approver proves drafting works while the send stays blocked.

Everything irreversible is therefore safe-by-default: ``digest``/``triage`` return
the *proposed* action and its disposition ("proposed"/"blocked"/"sent") rather
than firing it. Real transports (``notify-phone`` subprocess; a future Gmail/IMAP
client) live behind the seams in ``lab.comms_tools`` and are mocked in tests.

The digest's rec count is the REAL number of OPEN scout recommendations (status
``new``/``triaged`` — neither actioned nor rejected), read read-only from the
``scout_recommendations`` table. The smoke that reported "0 open recommendations"
was caused by ``digest`` defaulting ``recs`` to ``0``; it now defaults to that
live count (still read-only; overridable for tests).
"""

from __future__ import annotations

from typing import Any

from lab.platform.agent_runtime import run_agent
from lab.platform.authz import ApprovalCallback, Authorizer, default_authorizer

from lab.comms_tools import CommsTools
from lab.core.settings import get_settings

#: scout statuses that count as "open" (still need attention) — not actioned/
#: rejected. Mirrors migration 008's status CHECK (new|triaged|actioned|rejected).
_OPEN_SCOUT_STATUSES: tuple[str, ...] = ("new", "triaged")


# --------------------------------------------------------------------------- #
# Approvers (fail-closed by default; explicit, scoped openers for the gate).   #
# --------------------------------------------------------------------------- #


def approve_class(*classes: str) -> ApprovalCallback:
    """An approver that approves ONLY the named side-effect classes and denies
    everything else (still fail-closed for anything not listed). Used to prove
    the gate opens for an explicitly-approved class while staying shut for the
    rest — e.g. ``approve_class("write_local")`` lets drafts through but keeps
    ``irreversible`` sends blocked."""
    allowed = frozenset(classes)

    def _approver(request: dict[str, Any]) -> bool:
        return request.get("side_effect") in allowed

    return _approver


# --------------------------------------------------------------------------- #
# Open scout-rec count (read-only; the real digest input).                     #
# --------------------------------------------------------------------------- #


def open_rec_count() -> int:
    """Count OPEN scout recommendations (status in new|triaged) from the live DB.

    Read-only single SELECT over ``scout_recommendations`` (same DSN/pattern as
    ``lab.scout``); statuses bound as parameters, no interpolation. This is the
    real number the digest reports — replacing the old hard ``0`` default that
    made the smoke say "0 open recommendations"."""
    import psycopg

    # status bound as a single array param (= ANY) so the SQL stays a literal and
    # the open-status list is never interpolated into the query text.
    sql = "SELECT count(*) FROM scout_recommendations WHERE status = ANY(%s)"
    with (
        psycopg.connect(get_settings().pg_dsn) as conn,
        conn.cursor() as cur,
    ):
        cur.execute(sql, (list(_OPEN_SCOUT_STATUSES),))
        row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


# --------------------------------------------------------------------------- #
# digest — compose + propose an ntfy_push (irreversible -> gated).             #
# --------------------------------------------------------------------------- #

_DIGEST_SYSTEM = """You are the lab's comms agent. Compose a SHORT operator digest
of lab status, then propose ONE phone push with it.

You may read lab state (read-only tools auto-execute). To notify the operator you
must call ntfy_push — but that is IRREVERSIBLE (a push cannot be un-sent), so it
will only fire if a human approves it. Propose the push; do not assume it sent."""


def _final_text(messages: list[dict[str, Any]]) -> str:
    """The final assistant message content (the composed digest / summary)."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str) and content:
                return content
    return ""


def _push_fired(results: list[dict[str, Any]]) -> bool:
    """True iff an ntfy_push tool result indicates the push actually executed
    (``pushed: True``). A denied/blocked call returns an ``error`` instead."""
    for r in results:
        if r.get("name") != "ntfy_push":
            continue
        res = r.get("result")
        if isinstance(res, dict) and res.get("pushed") is True:
            return True
    return False


def _digest_text(*, recs: int) -> str:
    """Compose the digest body from simple, read-only/synthetic lab status."""
    return f"Lab digest: scout has {recs} open recommendation(s). Status: nominal."


def digest(
    *,
    model: str = "qwen3-4b-ft-toolcall-q4-latest",
    recs: int | None = None,
    rec_counter: Any = open_rec_count,
    tools: CommsTools | None = None,
    authorizer: Authorizer | None = None,
    approver: ApprovalCallback | None = None,
    max_tool_calls: int = 8,
    timeout: int = 90,
    num_ctx: int | None = None,
) -> dict[str, Any]:
    """Compose a lab digest and propose a phone push.

    ``recs`` is the number of OPEN scout recommendations to report. When left as
    ``None`` (the default), digest reads the REAL count via ``rec_counter`` (which
    defaults to the read-only ``open_rec_count`` DB query) — so the digest no
    longer hard-codes "0 open recommendations". Pass an explicit ``recs`` (or a
    fake ``rec_counter``) to override, e.g. in tests.

    With the ADR-013 defaults (``authorizer=None`` -> ``default_authorizer()``,
    ``approver=None`` -> fail-closed ``deny_approver``), the proposed ``ntfy_push``
    is IRREVERSIBLE and therefore resolves to require_approval -> deny: the push is
    returned as ``"proposed"`` and NEVER fires. Pass an ``approver`` that returns
    True for the irreversible class to actually send.

    Returns ``{"digest_text", "push": "proposed"|"sent", "recs", "tool_calls",
    "stop"}``.
    """
    settings = get_settings()
    ct = tools or CommsTools()
    authz = authorizer or default_authorizer()
    rec_count = recs if recs is not None else int(rec_counter())
    text = _digest_text(recs=rec_count)
    res = run_agent(
        settings=settings,
        litellm_key=settings.litellm_key,
        model=model,
        system=_DIGEST_SYSTEM,
        user=(
            f"Lab status to relay: {text!r}. Compose a one-line digest and propose "
            "a phone push (ntfy_push) of it to the operator."
        ),
        tools=ct.build_tools(),
        actor="comms",
        authorizer=authz,
        approver=approver,  # None -> runtime uses fail-closed deny_approver
        max_turns=max_tool_calls,
        max_tool_calls=max_tool_calls,
        timeout=timeout,
        num_ctx=num_ctx,
    )
    pushed = _push_fired(res.tool_results)
    return {
        "digest_text": _final_text(res.messages) or text,
        "push": "sent" if pushed else "proposed",
        "recs": rec_count,
        "tool_calls": res.tool_calls,
        "stop": res.stop_reason,
    }


# --------------------------------------------------------------------------- #
# triage — read (stub) -> classify -> draft -> propose send (irreversible).    #
# --------------------------------------------------------------------------- #

_TRIAGE_SYSTEM = """You are the lab's email-triage agent. Read the inbox with the
read tools, classify what needs a reply, and SAVE a draft reply with email_draft.
You may then propose sending it with email_send — but email_send is IRREVERSIBLE
and only fires on explicit human approval. Save drafts; propose (do not assume)
the send."""


def _drafts_from_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lift saved drafts out of email_draft tool results (write_local succeeded)."""
    out: list[dict[str, Any]] = []
    for r in results:
        if r.get("name") != "email_draft":
            continue
        res = r.get("result")
        if isinstance(res, dict) and res.get("draft_id"):
            out.append(res)
    return out


def _send_fired(results: list[dict[str, Any]]) -> bool:
    """True iff an email_send tool result indicates a send actually executed."""
    for r in results:
        if r.get("name") != "email_send":
            continue
        res = r.get("result")
        if isinstance(res, dict) and res.get("sent") is True:
            return True
    return False


def _send_proposed(results: list[dict[str, Any]]) -> bool:
    """True iff email_send was attempted at all (proposed) — fired or blocked."""
    return any(r.get("name") == "email_send" for r in results)


def triage(
    *,
    model: str = "qwen3-4b-ft-toolcall-q4-latest",
    tools: CommsTools | None = None,
    authorizer: Authorizer | None = None,
    approver: ApprovalCallback | None = None,
    max_tool_calls: int = 12,
    timeout: int = 90,
    num_ctx: int | None = None,
) -> dict[str, Any]:
    """Triage the inbox: read (dry-run stub) -> classify -> draft -> propose send.

    ``email_send`` is IRREVERSIBLE; with the default fail-closed approver it is
    NEVER sent. Drafting is ``write_local`` (also gated by default), so callers
    that want drafts persisted should pass ``approver=approve_class("write_local")``
    — which keeps the irreversible send blocked. Pass an approver that also
    approves "irreversible" to actually send.

    Returns ``{"drafts", "send": "blocked"|"proposed"|"sent", "tool_calls",
    "stop"}`` where ``send`` is "blocked" if a send was proposed but did not fire,
    "sent" if it fired, and "proposed" if no send was attempted (e.g. nothing to
    reply to).
    """
    settings = get_settings()
    ct = tools or CommsTools()
    authz = authorizer or default_authorizer()
    res = run_agent(
        settings=settings,
        litellm_key=settings.litellm_key,
        model=model,
        system=_TRIAGE_SYSTEM,
        user=(
            "Triage the inbox. Search it, read what matters, draft replies where "
            "needed, then propose sending each draft (email_send)."
        ),
        tools=ct.build_tools(),
        actor="comms",
        authorizer=authz,
        approver=approver,  # None -> runtime uses fail-closed deny_approver
        max_turns=max_tool_calls,
        max_tool_calls=max_tool_calls,
        timeout=timeout,
        num_ctx=num_ctx,
    )
    drafts = _drafts_from_results(res.tool_results)
    if _send_fired(res.tool_results):
        send_status = "sent"
    elif _send_proposed(res.tool_results):
        send_status = "blocked"
    else:
        send_status = "proposed"
    return {
        "drafts": drafts,
        "send": send_status,
        "tool_calls": res.tool_calls,
        "stop": res.stop_reason,
    }
