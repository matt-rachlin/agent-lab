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


def test_ambiguous_surfaced_not_auto_failed():
    r = result_from_scan(0, 1)
    assert r.passed is True  # 0 confirmed violations -> veto does not fire
    assert r.metadata["ambiguous"] == 1  # but flagged for review
