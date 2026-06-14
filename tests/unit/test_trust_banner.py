"""Stage 0a — report trust banner labels sub-finding results (ADR-008)."""

from lab.analyze.report import _trust_banner


def test_sub_finding_warns_and_reports_top_level():
    out = _trust_banner([{"trust_level": "raw", "n": 100, "legacy_n": 100}])
    assert "`raw`" in out
    assert "Sub-finding" in out
    assert "100" in out


def test_verified_top_has_no_warning():
    out = _trust_banner(
        [
            {"trust_level": "raw", "n": 2, "legacy_n": 0},
            {"trust_level": "verified", "n": 5, "legacy_n": 0},
        ]
    )
    assert "`verified`" in out
    assert "Sub-finding" not in out
