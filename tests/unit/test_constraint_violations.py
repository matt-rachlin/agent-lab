"""Stage 1 D3 — constraint violation evaluator mapping (ADR-009 safety veto)."""

from lab.eval.builtin.constraint_violations import result_from_scan


def test_compliant_passes():
    r = result_from_scan(0, 0)
    assert r.passed
    assert r.score == 0.0


def test_violation_fails_with_count():
    r = result_from_scan(2, 0)
    assert r.passed is False
    assert r.score == 2.0


def test_ambiguous_fails_closed():
    r = result_from_scan(0, 1)
    assert r.passed is False  # fail-closed even with zero hard violations
