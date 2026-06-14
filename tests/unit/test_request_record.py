"""Stage 0a D1 — request-fidelity helper round-trips the effective request."""

from lab.sweep.config import RunConfig
from lab.sweep.runner import _request_record


def test_captures_tools_choice_and_effective_sampling():
    cfg = RunConfig(
        name="greedy",
        temperature=0.0,
        top_p=1.0,
        max_tokens=4096,
        extra={"tool_choice": "required", "think": False},
    )
    rec = _request_record(
        cfg, tools=[{"type": "function", "function": {"name": "f"}}], tool_choice="required"
    )
    assert rec["request_tools"][0]["function"]["name"] == "f"
    assert rec["tool_choice"] == "required"
    assert rec["request_sampling"] == {"temperature": 0.0, "top_p": 1.0, "max_tokens": 4096}
    assert rec["request_extra"]["think"] is False


def test_single_turn_no_tools():
    rec = _request_record(RunConfig(name="greedy"), tools=None, tool_choice=None)
    assert rec["request_tools"] is None
    assert rec["tool_choice"] is None
    assert set(rec) == {"request_tools", "tool_choice", "request_sampling", "request_extra"}
