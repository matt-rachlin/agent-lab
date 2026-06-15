"""NS-4 Comms/Digest v0 — gated SEND vertical (ADR-012 LAR + ADR-013 authz).

ASSERT NOTHING SENDS. No GPU, no live LLM, NO real emails, NO real phone push.

Two layers of proof:

  1. RUNTIME-GATE proof (the load-bearing one): drive the REAL
     ``run_agent`` with ``call_litellm_chat`` stubbed to a scripted trajectory
     that calls the irreversible tools (email_send / ntfy_push). The tool impls
     are spies (CommsTools.calls + a push-sender spy). We assert that under the
     default authz with NO approver (fail-closed) those impls are reached ZERO
     times, and that WITH an approving callback the same call IS executed — the
     gate works both ways. The ntfy subprocess seam is replaced by a spy so no
     real push can fire.

  2. WIRING proof: mock ``lab.comms.run_agent`` and assert ``digest`` / ``triage``
     pass the authorizer + approver through and report disposition
     ("proposed"/"blocked"/"sent") correctly.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from lab.platform.agent_runtime import AgentResult, run_agent
from lab.platform.authz import AuthzPolicy, default_authorizer

from lab.comms import _digest_text, approve_class, digest, triage
from lab.comms_tools import CommsTools, DraftBook

# --------------------------------------------------------------------------- #
# Helpers: a scripted litellm + a never-firing push spy.                       #
# --------------------------------------------------------------------------- #


def _scripted_litellm(
    tool_calls: list[dict[str, Any]],
) -> Any:
    """One assistant turn emitting `tool_calls`, then a stop turn."""
    turns: list[dict[str, Any]] = [
        {"choices": [{"message": {"role": "assistant", "tool_calls": tool_calls}}]},
        {"choices": [{"message": {"role": "assistant", "content": "done"}}]},
    ]
    state = {"i": 0}

    def _call(**_kwargs: Any) -> tuple[dict[str, Any], int]:
        i = min(state["i"], len(turns) - 1)
        state["i"] += 1
        return turns[i], 1

    return _call


def _tc(name: str, args: str = "{}", cid: str = "c1") -> dict[str, Any]:
    return {"id": cid, "function": {"name": name, "arguments": args}}


class _PushSpy:
    """Stands in for the notify-phone subprocess; records calls, never pushes."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def __call__(self, title: str, body: str) -> None:
        self.sent.append((title, body))


def _draft_status(book: DraftBook, draft_id: str) -> str:
    """Fetch a draft's status, asserting the draft exists (typed, non-Optional)."""
    draft = book.get(draft_id)
    assert draft is not None
    status: str = draft["status"]
    return status


def _drive(
    tool_calls: list[dict[str, Any]],
    ct: CommsTools,
    **agent_kwargs: Any,
) -> AgentResult:
    """Run the REAL run_agent with a scripted model over ct's tools."""
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
            tools=ct.build_tools(),
            actor="comms",
            **agent_kwargs,
        )


# --------------------------------------------------------------------------- #
# 1. RUNTIME-GATE proof — irreversible cannot fire without explicit approval.  #
# --------------------------------------------------------------------------- #


def test_ntfy_push_default_authz_no_approver_does_not_fire() -> None:
    push = _PushSpy()
    ct = CommsTools(push_sender=push)
    res = _drive(
        [_tc("ntfy_push", '{"title": "hi", "body": "there"}')],
        ct,
        authorizer=default_authorizer(),  # no approver -> fail-closed deny
    )
    # The push impl was NEVER reached, and no real push fired.
    assert "ntfy_push" not in ct.calls
    assert push.sent == []
    assert "denied" in str(res.tool_results[0]["result"]).lower()


def test_email_send_default_authz_no_approver_does_not_fire() -> None:
    book = DraftBook()
    book.add("a@b.c", "s", "b")  # draft-1 exists
    ct = CommsTools(drafts=book)
    res = _drive(
        [_tc("email_send", '{"draft_id": "draft-1"}')],
        ct,
        authorizer=default_authorizer(),
    )
    assert "email_send" not in ct.calls  # impl never reached
    assert _draft_status(book, "draft-1") == "draft"  # NOT sent
    assert "denied" in str(res.tool_results[0]["result"]).lower()


def test_irreversible_blocked_even_with_explicit_grant() -> None:
    """ADR-013: irreversible is never auto, even with an explicit grant set."""
    push = _PushSpy()
    ct = CommsTools(push_sender=push)
    granted = AuthzPolicy(grants={("comms", "irreversible")})
    _drive(
        [_tc("ntfy_push", '{"title": "x", "body": "y"}')],
        ct,
        authorizer=granted,  # grant present, still require_approval -> deny
    )
    assert ct.calls == []
    assert push.sent == []


def test_ntfy_push_fires_only_with_approving_callback() -> None:
    push = _PushSpy()
    ct = CommsTools(push_sender=push)
    res = _drive(
        [_tc("ntfy_push", '{"title": "hi", "body": "there"}')],
        ct,
        authorizer=default_authorizer(),
        approver=lambda _req: True,  # explicit approval opens the gate
    )
    assert ct.calls == ["ntfy_push"]
    assert push.sent == [("hi", "there")]
    assert res.tool_results[0]["result"]["pushed"] is True


def test_email_send_fires_only_with_approving_callback() -> None:
    book = DraftBook()
    book.add("a@b.c", "s", "b")  # draft-1
    ct = CommsTools(drafts=book)
    res = _drive(
        [_tc("email_send", '{"draft_id": "draft-1"}')],
        ct,
        authorizer=default_authorizer(),
        approver=lambda _req: True,
    )
    assert ct.calls == ["email_send"]
    assert _draft_status(book, "draft-1") == "sent"
    assert res.tool_results[0]["result"]["sent"] is True


def test_approver_scoped_to_class_send_no_draft_yes() -> None:
    """A draft-yes / send-no approver: write_local executes, irreversible does
    NOT — the two gates are independent."""
    push = _PushSpy()
    ct = CommsTools(push_sender=push)
    res = _drive(
        [
            _tc("email_draft", '{"to": "a@b.c", "subject": "s", "body": "b"}', "c1"),
            _tc("email_send", '{"draft_id": "draft-1"}', "c2"),
        ],
        ct,
        authorizer=default_authorizer(),
        approver=approve_class("write_local"),  # NOT "irreversible"
    )
    assert "email_draft" in ct.calls  # draft saved
    assert "email_send" not in ct.calls  # send blocked
    by_name = {r["name"]: r["result"] for r in res.tool_results}
    assert by_name["email_draft"]["status"] == "draft"
    assert "denied" in str(by_name["email_send"]).lower()


def test_read_tools_auto_execute_and_never_push() -> None:
    """external_read (email_search/email_read) auto-allow; they touch no network
    (dry-run stub returns synthetic/empty) and never push."""
    push = _PushSpy()
    ct = CommsTools(push_sender=push)
    res = _drive(
        [_tc("email_search", '{"query": "unread"}')],
        ct,
        authorizer=default_authorizer(),  # no approver needed for reads
    )
    assert ct.calls == ["email_search"]
    assert push.sent == []
    assert res.tool_results[0]["result"]["messages"] == []  # empty stub


def test_tool_side_effect_classes_are_correct() -> None:
    by_name = {t.name: t for t in CommsTools().build_tools()}
    assert by_name["ntfy_push"].side_effect == "irreversible"
    assert by_name["email_send"].side_effect == "irreversible"
    assert by_name["email_draft"].side_effect == "write_local"
    assert by_name["email_search"].side_effect == "external_read"
    assert by_name["email_read"].side_effect == "external_read"


# --------------------------------------------------------------------------- #
# 2. WIRING proof — digest/triage report disposition; nothing fires by default. #
# --------------------------------------------------------------------------- #


class _FakeAgentResult:
    def __init__(self, messages: list[dict[str, Any]], tool_results: list[dict[str, Any]]) -> None:
        self.messages = messages
        self.tool_calls = len(tool_results)
        self.tool_results = tool_results
        self.stop_reason = "stop"


def test_digest_default_reports_proposed_not_sent(monkeypatch: Any) -> None:
    """With default authz + no approver, digest replays the real runtime gate:
    the ntfy_push is denied, so digest reports 'proposed' and nothing fires."""
    push = _PushSpy()

    def fake_run_agent(**kwargs: Any) -> AgentResult:
        # Replay the genuine gate by delegating to the real run_agent with a
        # scripted model that proposes the push.
        ct_for_run = CommsTools(push_sender=push)
        # map the schemas back to spy-able impls via a fresh CommsTools
        return _drive(
            [_tc("ntfy_push", '{"title": "Lab", "body": "digest"}')],
            ct_for_run,
            authorizer=kwargs["authorizer"],
            approver=kwargs.get("approver"),
        )

    monkeypatch.setattr("lab.comms.run_agent", fake_run_agent)
    out = digest(recs=3)
    assert out["push"] == "proposed"
    assert push.sent == []  # nothing pushed
    assert out["digest_text"]  # a digest was composed


def test_digest_text_reflects_rec_count() -> None:
    from lab.comms import _digest_text

    assert "3" in _digest_text(recs=3)
    assert "0" in _digest_text(recs=0)


def test_digest_sends_when_approver_yes(monkeypatch: Any) -> None:
    push = _PushSpy()

    def fake_run_agent(**kwargs: Any) -> AgentResult:
        ct_for_run = CommsTools(push_sender=push)
        return _drive(
            [_tc("ntfy_push", '{"title": "Lab", "body": "digest"}')],
            ct_for_run,
            authorizer=kwargs["authorizer"],
            approver=kwargs.get("approver"),
        )

    monkeypatch.setattr("lab.comms.run_agent", fake_run_agent)
    out = digest(recs=1, approver=lambda _req: True)
    assert out["push"] == "sent"
    assert push.sent == [("Lab", "digest")]


def test_triage_default_send_blocked(monkeypatch: Any) -> None:
    book = DraftBook()

    def fake_run_agent(**kwargs: Any) -> AgentResult:
        ct_for_run = CommsTools(drafts=book)
        # draft, then propose a send — under default authz the draft also needs
        # approval; here we approve only write_local so a draft exists to send.
        return _drive(
            [
                _tc("email_draft", '{"to": "a@b.c", "subject": "re", "body": "x"}', "c1"),
                _tc("email_send", '{"draft_id": "draft-1"}', "c2"),
            ],
            ct_for_run,
            authorizer=kwargs["authorizer"],
            approver=approve_class("write_local"),  # draft yes, send no
        )

    monkeypatch.setattr("lab.comms.run_agent", fake_run_agent)
    out = triage()
    assert out["send"] == "blocked"  # send proposed but never fired
    assert len(out["drafts"]) == 1
    assert _draft_status(book, "draft-1") == "draft"  # NOT sent


def test_triage_no_send_proposed_is_proposed(monkeypatch: Any) -> None:
    """Empty inbox -> nothing to send -> 'proposed' (no send attempted)."""

    def fake_run_agent(**kwargs: Any) -> AgentResult:
        ct_for_run = CommsTools()
        return _drive(
            [_tc("email_search", '{"query": "unread"}')],
            ct_for_run,
            authorizer=kwargs["authorizer"],
            approver=kwargs.get("approver"),
        )

    monkeypatch.setattr("lab.comms.run_agent", fake_run_agent)
    out = triage()
    assert out["send"] == "proposed"
    assert out["drafts"] == []


def test_triage_sends_with_full_approval(monkeypatch: Any) -> None:
    book = DraftBook()

    def fake_run_agent(**kwargs: Any) -> AgentResult:
        ct_for_run = CommsTools(drafts=book)
        return _drive(
            [
                _tc("email_draft", '{"to": "a@b.c", "subject": "re", "body": "x"}', "c1"),
                _tc("email_send", '{"draft_id": "draft-1"}', "c2"),
            ],
            ct_for_run,
            authorizer=kwargs["authorizer"],
            approver=approve_class("write_local", "irreversible"),
        )

    monkeypatch.setattr("lab.comms.run_agent", fake_run_agent)
    out = triage(approver=approve_class("write_local", "irreversible"))
    assert out["send"] == "sent"
    assert _draft_status(book, "draft-1") == "sent"


# --------------------------------------------------------------------------- #
# 3. REC-COUNT wiring — digest reports the REAL open scout-rec count.          #
# --------------------------------------------------------------------------- #


def test_digest_uses_real_open_rec_count_when_recs_omitted(monkeypatch: Any) -> None:
    """The smoke reported '0 open recommendations' though the DB had ~9. With no
    explicit ``recs``, digest must read the REAL count via ``rec_counter`` (here a
    mock standing in for the read-only DB query) and reflect it in the digest."""
    push = _PushSpy()

    def fake_run_agent(**kwargs: Any) -> AgentResult:
        ct_for_run = CommsTools(push_sender=push)
        return _drive(
            [_tc("ntfy_push", '{"title": "Lab", "body": "digest"}')],
            ct_for_run,
            authorizer=kwargs["authorizer"],
            approver=kwargs.get("approver"),
        )

    monkeypatch.setattr("lab.comms.run_agent", fake_run_agent)
    out = digest(rec_counter=lambda: 9)  # real count source mocked -> 9 open recs
    assert out["recs"] == 9  # the real count, not the old hard-coded 0
    assert "9" in _digest_text(recs=out["recs"])  # count flows into the body
    assert out["push"] == "proposed"
    assert push.sent == []  # still nothing fires


def test_digest_explicit_recs_overrides_counter(monkeypatch: Any) -> None:
    """An explicit ``recs`` wins over the counter (and the counter is not called)."""
    push = _PushSpy()
    called = {"n": 0}

    def counter() -> int:
        called["n"] += 1
        return 99

    def fake_run_agent(**kwargs: Any) -> AgentResult:
        ct_for_run = CommsTools(push_sender=push)
        return _drive(
            [_tc("ntfy_push", '{"title": "Lab", "body": "digest"}')],
            ct_for_run,
            authorizer=kwargs["authorizer"],
            approver=kwargs.get("approver"),
        )

    monkeypatch.setattr("lab.comms.run_agent", fake_run_agent)
    out = digest(recs=2, rec_counter=counter)
    assert out["recs"] == 2
    assert "2" in _digest_text(recs=out["recs"])
    assert called["n"] == 0  # explicit recs short-circuits the DB query


def test_open_rec_count_queries_open_statuses_only() -> None:
    """open_rec_count is a read-only single SELECT that counts only open statuses
    (new|triaged), binding the statuses as parameters (no interpolation)."""
    import lab.comms as comms_mod

    captured: dict[str, Any] = {}

    class _Cur:
        def execute(self, sql: str, params: Any) -> None:
            captured["sql"] = sql
            captured["params"] = params

        def fetchone(self) -> tuple[int]:
            return (9,)

        def __enter__(self) -> _Cur:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

    class _Conn:
        def cursor(self) -> _Cur:
            return _Cur()

        def __enter__(self) -> _Conn:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

    class _Psycopg:
        @staticmethod
        def connect(_dsn: str) -> _Conn:
            return _Conn()

    with patch.dict("sys.modules", {"psycopg": _Psycopg}):
        n = comms_mod.open_rec_count()

    assert n == 9
    # only open statuses are counted, bound as a single array param (never
    # actioned/rejected)
    assert captured["params"] == (["new", "triaged"],)
    assert "actioned" not in captured["sql"]
    assert "rejected" not in captured["sql"]
    assert "scout_recommendations" in captured["sql"]
    # statuses are bound (= ANY), not interpolated into the SQL text
    assert "new" not in captured["sql"]
    assert "ANY" in captured["sql"]
