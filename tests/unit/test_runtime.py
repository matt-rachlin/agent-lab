"""Lab Agent Runtime (ADR-012) — loop, audit, side-effect gate, stop conditions.
No network/DB: call_litellm_chat + record_action are monkeypatched."""

import json

from lab.platform.agent_runtime import Tool, run_agent


class _FakeSettings:
    litellm_key = "k"


def _msg(tool_calls=None, content=""):
    m = {"role": "assistant", "content": content}
    if tool_calls:
        m["tool_calls"] = tool_calls
    return {"choices": [{"message": m}]}


def _tc(name, args, cid="c1"):
    return {
        "id": cid,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _patch(monkeypatch, responses, audit):
    it = iter(responses)
    monkeypatch.setattr("lab.platform.agent_runtime.call_litellm_chat", lambda **kw: (next(it), 1))
    monkeypatch.setattr(
        "lab.platform.agent_runtime.record_action", lambda **kw: audit.append(kw) or ""
    )


def _tool(name, impl, side_effect="read"):
    return Tool(
        name=name,
        description="d",
        parameters={"type": "object", "properties": {}},
        impl=impl,
        side_effect=side_effect,
    )


def test_dispatch_records_and_stops(monkeypatch):
    seen, audit = [], []
    tool = Tool(
        name="t",
        description="d",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}},
        impl=lambda x: (seen.append(x), {"ok": x})[1],
        side_effect="read",
    )
    _patch(monkeypatch, [_msg([_tc("t", {"x": "hi"})]), _msg()], audit)
    res = run_agent(
        settings=_FakeSettings(),
        litellm_key="k",
        model="m",
        system="s",
        user="u",
        tools=[tool],
        max_turns=5,
    )
    assert seen == ["hi"]
    assert res.tool_calls == 1
    assert res.stop_reason == "stop"
    assert res.tool_results[0]["result"] == {"ok": "hi"}
    assert any(a["action"] == "tool:t" for a in audit)


def test_side_effect_gate_blocks_unauthorized(monkeypatch):
    called, audit = [], []
    tool = _tool("w", lambda: called.append(1), side_effect="write_local")
    _patch(monkeypatch, [_msg([_tc("w", {})]), _msg()], audit)
    res = run_agent(
        settings=_FakeSettings(), litellm_key="k", model="m", system="s", user="u", tools=[tool]
    )  # default allow: read/external_read
    assert called == []  # impl never executed
    assert "blocked" in res.tool_results[0]["result"]["error"]
    assert any(a["action"] == "blocked:w" for a in audit)


def test_write_allowed_when_authorized(monkeypatch):
    called, audit = [], []
    tool = _tool("w", lambda: (called.append(1), {"done": True})[1], side_effect="write_local")
    _patch(monkeypatch, [_msg([_tc("w", {})]), _msg()], audit)
    res = run_agent(
        settings=_FakeSettings(),
        litellm_key="k",
        model="m",
        system="s",
        user="u",
        tools=[tool],
        allow_side_effects=frozenset({"read", "write_local"}),
    )
    assert called == [1]
    assert res.tool_results[0]["result"] == {"done": True}


def test_stop_predicate(monkeypatch):
    audit = []
    tool = _tool("t", lambda: {"r": 1})
    _patch(monkeypatch, [_msg([_tc("t", {})]), _msg([_tc("t", {})]), _msg()], audit)
    res = run_agent(
        settings=_FakeSettings(),
        litellm_key="k",
        model="m",
        system="s",
        user="u",
        tools=[tool],
        stop_predicate=lambda results: len(results) >= 1,
    )
    assert res.stop_reason == "stop_predicate"
    assert res.tool_calls == 1


def test_max_tool_calls(monkeypatch):
    audit = []
    tool = _tool("t", lambda: {"r": 1})
    _patch(monkeypatch, [_msg([_tc("t", {})]) for _ in range(10)], audit)
    res = run_agent(
        settings=_FakeSettings(),
        litellm_key="k",
        model="m",
        system="s",
        user="u",
        tools=[tool],
        max_tool_calls=3,
        max_turns=10,
    )
    assert res.tool_calls == 3
    assert res.stop_reason == "max_tool_calls"


def test_unknown_tool(monkeypatch):
    audit = []
    _patch(monkeypatch, [_msg([_tc("nope", {})]), _msg()], audit)
    res = run_agent(
        settings=_FakeSettings(), litellm_key="k", model="m", system="s", user="u", tools=[]
    )
    assert "unknown tool" in res.tool_results[0]["result"]["error"]
    assert res.stop_reason == "stop"


def test_scout_build_tools_side_effects():
    from lab.scout_tools import build_tools

    tools = {t.name: t for t in build_tools()}
    assert tools["scout_add"].side_effect == "write_local"
    assert tools["web_search"].side_effect == "external_read"
    assert tools["fetch_url"].side_effect == "external_read"
