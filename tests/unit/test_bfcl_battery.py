"""Stage 0b #9 — independent re-grade path (BFCL battery)."""

from lab.eval.bfcl_battery import _independent_grade


def _tc(name: str) -> dict[str, object]:
    return {"function": {"name": name, "arguments": "{}"}}


def test_independent_grade_matches_expected_name():
    assert _independent_grade([_tc("f")], [{"f": {"x": [1]}}])


def test_independent_grade_fails_on_wrong_name():
    assert not _independent_grade([_tc("g")], [{"f": {"x": [1]}}])


def test_independent_grade_fails_on_no_call():
    assert not _independent_grade([], [{"f": {"x": [1]}}])
