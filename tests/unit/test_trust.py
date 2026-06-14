"""Stage 0b #8 — trust transitions + BFCL validity gate (ADR-008)."""

from lab.core.trust import _row_hash, bfcl_validity


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
