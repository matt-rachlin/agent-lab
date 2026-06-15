"""Stage 0b #8 — trust transitions + BFCL validity gate (ADR-008)."""

from lab.platform.trust import (
    _row_hash,
    baseline_sanity,
    bfcl_validity,
    contamination_signal,
    decode_integrity,
    judge_integrity,
    single_turn_validity,
)


def test_bfcl_validity_passes_with_tools_and_choice():
    r = bfcl_validity(
        request_tools=[{"x": 1}], tool_choice="required", bfcl_error_type=None, passed=True
    )
    assert r.passed
    assert r.emitted is True
    assert r.correct is True


def test_bfcl_validity_flags_missing_tools():
    r = bfcl_validity(
        request_tools=None,
        tool_choice="required",
        bfcl_error_type="model_output:no_tool_call",
        passed=False,
    )
    assert not r.passed
    assert any("expects tools" in v for v in r.violations)
    assert r.emitted is False


def test_row_hash_is_deterministic_and_chains():
    h1 = _row_hash(None, "r1", "raw", "validity_passed", "sys", False, {"a": 1}, None)
    h1b = _row_hash(None, "r1", "raw", "validity_passed", "sys", False, {"a": 1}, None)
    h2 = _row_hash(h1, "r1", "raw", "validity_passed", "sys", False, {"a": 1}, None)
    assert h1 == h1b
    assert h1 != h2
    assert len(h1) == 64


def test_single_turn_validity_passes_with_output():
    r = single_turn_validity(
        request_sampling={"temperature": 0.0}, response_text="hi", raw_response=None
    )
    assert r.passed
    assert r.emitted is True


def test_single_turn_validity_flags_empty_output():
    r = single_turn_validity(
        request_sampling={"temperature": 0.0}, response_text="", raw_response={}
    )
    assert not r.passed
    assert any("no model output" in v for v in r.violations)


def test_decode_integrity_flags_truncation_and_empty():
    assert decode_integrity({"choices": [{"finish_reason": "length", "message": {"content": "x"}}]})
    assert decode_integrity({"choices": [{"finish_reason": "stop", "message": {}}]})
    assert decode_integrity(None) == ["decode: no choices in response"]


def test_decode_integrity_clean_response_ok():
    ok = {"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]}
    assert decode_integrity(ok) == []


def test_baseline_sanity_both_directions():
    assert baseline_sanity(0.1, 0.4, 0.95)  # below floor
    assert baseline_sanity(0.99, 0.4, 0.95)  # above ceiling
    assert baseline_sanity(0.7, 0.4, 0.95) == []
    assert baseline_sanity(0.7, None, None) == []


def test_contamination_signal_flags_big_drop():
    assert contamination_signal(0.9, 0.5)
    assert contamination_signal(0.9, 0.85) == []


def test_judge_integrity_requires_calibration():
    assert judge_integrity(None)
    assert judge_integrity(0.6)
    assert judge_integrity(0.9) == []
