"""NS-4 Comms/Digest v0 — tool implementations + the SEND/push seams (ADR-013).

NS-4 is the first real consumer of the ADR-013 SEND / approval path. The whole
point of this slice is the **anti-footgun guarantee**: no *irreversible* action
(``email_send``, ``ntfy_push``) may fire without an explicit approval, and the
runtime default is **fail-closed deny** (see ``lab.platform.authz.deny_approver`` and
``lab.platform.agent_runtime.run_agent``'s approval handling).

Side-effect classes (ADR-013 §2), as wired onto the Tool ABI:

    ntfy_push     -> irreversible    (a phone push cannot be un-sent)
    email_search  -> external_read   (read the inbox; auto-allowed)
    email_read    -> external_read   (read one message; auto-allowed)
    email_draft   -> write_local     (save a reversible local draft)
    email_send    -> irreversible    (sending mail cannot be un-sent)

LIVE vs DRY-RUN STUBS
---------------------
DRY-RUN STUBS (documented seams — the lab has NO live transports here):

  * ``email_search`` / ``email_read`` — the lab runtime has **no live Gmail
    client**. The claude.ai Gmail MCP is session-only and is NOT available to the
    lab runtime, so these return SYNTHETIC / empty inbox data. This is the read
    seam a real IMAP/Gmail client plugs into later. No network, ever.
  * ``email_draft`` — records a draft into an in-memory ``DraftBook`` (reversible,
    process-local). The real seam is a Drafts API / local maildir; here it is a
    pure in-memory store so tests are deterministic and nothing leaves the box.
  * ``email_send`` — DOES NOT SEND. Even if the gate let it through, the v0 impl
    only *marks* an in-memory draft "sent" and returns a receipt. There is no
    SMTP/Gmail client wired. The gate (irreversible -> require_approval ->
    fail-closed) is what actually guarantees no send; this impl having no real
    transport is defense-in-depth.

REAL-ish SEAM (still safe under test):

  * ``ntfy_push`` — the host has a ``notify-phone "title" "body"`` command (lab
    convention). The real impl shells out to it via subprocess. In tests this is
    ALWAYS mocked (``push_sender`` injection) so no real push fires; and it is
    irreversible, so the gate blocks it by default regardless.

Every impl is a *spy-able* callable: it records that it ran (so a test can assert
the underlying impl was reached zero times when the gate denies it).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from lab.platform.agent_runtime import Tool

# --------------------------------------------------------------------------- #
# The phone-push transport seam (real-ish, but injectable so tests never push). #
# --------------------------------------------------------------------------- #

#: A phone-push transport: (title, body) -> None. The production default shells
#: out to the host's ``notify-phone`` command; tests inject a no-op/spy so no
#: real push ever fires.
PushSender = Callable[[str, str], None]


def notify_phone_subprocess(title: str, body: str) -> None:
    """Production push seam: invoke the host's ``notify-phone "title" "body"``.

    NEVER called in a test — ``CommsTools`` is always constructed with an
    injected ``push_sender`` spy under test, and ``ntfy_push`` is irreversible so
    the default fail-closed gate blocks it before this could run anyway.
    """
    subprocess.run(["notify-phone", title, body], check=True)  # pragma: no cover


# --------------------------------------------------------------------------- #
# In-memory inbox / draft seams (dry-run stubs).                               #
# --------------------------------------------------------------------------- #


@dataclass
class DraftBook:
    """Process-local draft store (the reversible ``write_local`` seam)."""

    _drafts: dict[str, dict[str, Any]] = field(default_factory=dict)
    _seq: int = 0

    def add(self, to: str, subject: str, body: str) -> str:
        self._seq += 1
        draft_id = f"draft-{self._seq}"
        self._drafts[draft_id] = {
            "id": draft_id,
            "to": to,
            "subject": subject,
            "body": body,
            "status": "draft",
        }
        return draft_id

    def get(self, draft_id: str) -> dict[str, Any] | None:
        return self._drafts.get(draft_id)

    def all(self) -> list[dict[str, Any]]:
        return list(self._drafts.values())


#: A synthetic inbox row factory: query -> list of {id, from, subject, snippet}.
#: Default is the empty/stub seam (the lab has no live Gmail). Inject synthetic
#: rows in tests / demos.
InboxProvider = Callable[[str], list[dict[str, Any]]]


def empty_inbox(_query: str) -> list[dict[str, Any]]:
    """Default inbox seam: no live Gmail client -> empty result set."""
    return []


# --------------------------------------------------------------------------- #
# Tool implementations (spy-able), bound to a CommsTools context.              #
# --------------------------------------------------------------------------- #


@dataclass
class CommsTools:
    """Holds the comms seams (inbox provider, draft book, push sender) and
    exposes the five tool impls. Construct with injected seams in tests so no
    real email is read and no real push fires.

    ``calls`` is a spy log: every impl appends its name BEFORE doing anything, so
    a test can assert e.g. ``"email_send" not in tools.calls`` when the gate
    denies it (the runtime never reaches ``impl`` for a denied call, so the spy
    stays empty for blocked irreversible actions)."""

    inbox: InboxProvider = empty_inbox
    drafts: DraftBook = field(default_factory=DraftBook)
    push_sender: PushSender = notify_phone_subprocess
    calls: list[str] = field(default_factory=list)

    # ----- irreversible -----------------------------------------------------
    def ntfy_push(self, title: str, body: str) -> dict[str, Any]:
        """IRREVERSIBLE: push to the phone (cannot be un-sent). Gated."""
        self.calls.append("ntfy_push")
        self.push_sender(title, body)
        return {"pushed": True, "title": title}

    def email_send(self, draft_id: str) -> dict[str, Any]:
        """IRREVERSIBLE: send a saved draft (cannot be un-sent). Gated.

        DRY-RUN STUB: no SMTP/Gmail client is wired; this only marks the local
        draft "sent" and returns a receipt. Reaching this at all already required
        passing the irreversible gate."""
        self.calls.append("email_send")
        draft = self.drafts.get(draft_id)
        if draft is None:
            return {"sent": False, "error": f"unknown draft: {draft_id}"}
        draft["status"] = "sent"
        return {"sent": True, "draft_id": draft_id}

    # ----- write_local ------------------------------------------------------
    def email_draft(self, to: str, subject: str, body: str) -> dict[str, Any]:
        """WRITE_LOCAL: save a reversible draft. Returns its draft_id."""
        self.calls.append("email_draft")
        draft_id = self.drafts.add(to, subject, body)
        return {"draft_id": draft_id, "status": "draft"}

    # ----- external_read (dry-run stubs; no live Gmail) ---------------------
    def email_search(self, query: str) -> dict[str, Any]:
        """EXTERNAL_READ (DRY-RUN STUB): search the inbox. Synthetic/empty —
        the lab has no live Gmail client."""
        self.calls.append("email_search")
        return {"query": query, "messages": self.inbox(query)}

    def email_read(self, id: str) -> dict[str, Any]:
        """EXTERNAL_READ (DRY-RUN STUB): read one message by id. Looks it up in
        the synthetic inbox; returns ``found=False`` for the empty seam."""
        self.calls.append("email_read")
        for msg in self.inbox(""):
            if msg.get("id") == id:
                return {"found": True, "message": msg}
        return {"found": False, "id": id}

    # ----- Tool ABI wiring --------------------------------------------------
    def build_tools(self) -> list[Tool]:
        """The five NS-4 tools as ADR-012 Tool ABI instances, each carrying the
        ADR-013 side-effect class that drives the authorization gate."""
        return [
            Tool(
                name="email_search",
                description=(
                    "Search the inbox for messages matching a query. Read-only "
                    "(external_read). DRY-RUN STUB: synthetic/empty (no live Gmail)."
                ),
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                impl=self.email_search,
                side_effect="external_read",
                capability="email",
            ),
            Tool(
                name="email_read",
                description=(
                    "Read one inbox message by id. Read-only (external_read). "
                    "DRY-RUN STUB: synthetic (no live Gmail)."
                ),
                parameters={
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
                impl=self.email_read,
                side_effect="external_read",
                capability="email",
            ),
            Tool(
                name="email_draft",
                description=(
                    "Save a reversible email draft (to, subject, body) and return "
                    "its draft_id. write_local."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["to", "subject", "body"],
                },
                impl=self.email_draft,
                side_effect="write_local",
                capability="email",
            ),
            Tool(
                name="email_send",
                description=(
                    "Send a previously saved draft by draft_id. IRREVERSIBLE — "
                    "requires explicit approval; never auto in v0."
                ),
                parameters={
                    "type": "object",
                    "properties": {"draft_id": {"type": "string"}},
                    "required": ["draft_id"],
                },
                impl=self.email_send,
                side_effect="irreversible",
                capability="email",
            ),
            Tool(
                name="ntfy_push",
                description=(
                    "Push a notification to the operator's phone (title, body). "
                    "IRREVERSIBLE — a push cannot be un-sent; requires explicit "
                    "approval, never auto in v0."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["title", "body"],
                },
                impl=self.ntfy_push,
                side_effect="irreversible",
                capability="push",
            ),
        ]
