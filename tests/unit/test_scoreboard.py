"""Stage 1 #15 — scoreboard multi-axis gate + safety veto (ADR-009)."""

from lab.analyze.scoreboard import Entry, TierConfig, evaluate_tier

T = TierConfig("t", capability_floor=0.5, reliability_floor=0.5, safety_completion_floor=0.5)


def _entry(**over: object) -> Entry:
    base: dict[str, object] = {
        "model": "m",
        "config_hash": "h",
        "capability": {"bfcl-v3-ast": 0.8},
        "reliability": 0.8,
        "safety_violations": 0,
        "safety_completion": 0.9,
        "cost_tokens_out": 100,
    }
    base.update(over)
    return Entry(**base)  # type: ignore[arg-type]


def test_clean_entry_passes():
    assert evaluate_tier(_entry(), T).status == "pass"


def test_safety_violation_vetoes():
    v = evaluate_tier(_entry(safety_violations=1), T)
    assert v.status == "fail"
    assert any(a.axis == "safety" and "VETO" in a.detail for a in v.axes)


def test_no_violation_data_is_incomplete():
    assert evaluate_tier(_entry(safety_violations=None), T).status == "incomplete"


def test_over_refusal_fails_safety():
    v = evaluate_tier(_entry(safety_completion=0.1), T)
    assert v.status == "fail"
    assert any(a.axis == "safety" and "over-refusal" in a.detail for a in v.axes)


def test_low_capability_fails():
    assert evaluate_tier(_entry(capability={"bfcl-v3-ast": 0.2}), T).status == "fail"


def test_no_capability_is_incomplete():
    assert evaluate_tier(_entry(capability={}, reliability=None), T).status == "incomplete"
