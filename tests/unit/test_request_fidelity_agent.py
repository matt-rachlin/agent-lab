"""Stage 0a #12 — agent-path request fidelity in the trajectory header."""

from types import SimpleNamespace

from lab.inspect_bridge.logwriter import _request_fidelity


def test_captures_config_and_invoked_tools():
    ctx = SimpleNamespace(
        config={
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 2048,
            "extra": {"tool_choice": "required"},
        }
    )
    lab_agent = {
        "turns": [{"tools": [{"tool": "read_file"}]}, {"tool_calls": [{"name": "run_sql"}]}]
    }
    r = _request_fidelity(ctx, lab_agent)  # type: ignore[arg-type]
    assert r["sampling"]["max_tokens"] == 2048
    assert r["tool_choice"] == "required"
    assert r["tools_invoked"] == ["read_file", "run_sql"]


def test_empty_agent_is_safe():
    ctx = SimpleNamespace(config={})
    r = _request_fidelity(ctx, {})  # type: ignore[arg-type]
    assert r["tools_invoked"] == []
    assert r["tool_choice"] is None
