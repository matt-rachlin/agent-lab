"""Unit tests for ADR-008 finding-doc trust ladder (P2.R5)."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import get_args

import pytest

from lab.finding import (
    TRUST_RUNGS,
    TrustLevel,
    backfill_trust,
    parse_finding,
    promote_finding,
)

# ---------------------------------------------------------------------------
# 1. TrustLevel literal includes all expected rungs
# ---------------------------------------------------------------------------


def test_trust_level_literal_rungs() -> None:
    expected = {"unverified", "verified", "reliability_confirmed", "deployable", "retracted"}
    actual = set(get_args(TrustLevel))
    assert actual == expected, f"TrustLevel args mismatch: {actual}"


def test_trust_rungs_tuple_order() -> None:
    assert TRUST_RUNGS[0] == "unverified"
    assert TRUST_RUNGS[-1] == "retracted"
    assert "verified" in TRUST_RUNGS
    assert "reliability_confirmed" in TRUST_RUNGS
    assert "deployable" in TRUST_RUNGS


# ---------------------------------------------------------------------------
# Helpers for writing temp finding docs
# ---------------------------------------------------------------------------


def _write_finding(
    tmp_path: Path,
    slug: str = "F-099",
    trust_level: str = "unverified",
    has_depends_on: bool = False,
    source: str = "EXP-TEST-001",
) -> Path:
    dep_field = f"depends_on: {source}" if has_depends_on else ""
    text = textwrap.dedent(f"""\
        ---
        doc_id: {slug.lower()}-test
        title: 'test finding'
        {dep_field}
        ---
        # {slug}: Test claim for trust tests

        Date: 2026-06-14
        Confidence: high
        Source: {source}
        trust_level: {trust_level}

        ## Claim

        A test claim.
    """)
    p = tmp_path / f"{slug}-test-claim.md"
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 2. promote happy path: unverified -> verified
# ---------------------------------------------------------------------------


def test_promote_unverified_to_verified(tmp_path: Path) -> None:
    _write_finding(tmp_path, slug="F-099", trust_level="unverified", has_depends_on=True)
    result = promote_finding("F-099", "verified", findings_dir=tmp_path)
    assert result.exists()
    text = result.read_text(encoding="utf-8")
    assert "trust_level: verified" in text
    assert "## Promotion history" in text
    assert "unverified -> verified" in text


def test_promote_verified_to_reliability_confirmed(tmp_path: Path) -> None:
    _write_finding(tmp_path, slug="F-099", trust_level="verified", has_depends_on=True)
    result = promote_finding("F-099", "reliability_confirmed", findings_dir=tmp_path)
    text = result.read_text(encoding="utf-8")
    assert "trust_level: reliability_confirmed" in text
    assert "verified -> reliability_confirmed" in text


def test_promote_to_retracted_from_any_rung(tmp_path: Path) -> None:
    _write_finding(tmp_path, slug="F-099", trust_level="unverified", has_depends_on=False)
    result = promote_finding("F-099", "retracted", findings_dir=tmp_path)
    text = result.read_text(encoding="utf-8")
    assert "trust_level: retracted" in text
    assert "-> retracted" in text


# ---------------------------------------------------------------------------
# 3. promote refuses rung skip without --force
# ---------------------------------------------------------------------------


def test_promote_refuses_rung_skip(tmp_path: Path) -> None:
    _write_finding(tmp_path, slug="F-099", trust_level="unverified", has_depends_on=True)
    with pytest.raises(ValueError, match="rung skip"):
        promote_finding("F-099", "reliability_confirmed", findings_dir=tmp_path, force=False)


def test_promote_allows_rung_skip_with_force(tmp_path: Path) -> None:
    _write_finding(tmp_path, slug="F-099", trust_level="unverified", has_depends_on=True)
    result = promote_finding("F-099", "reliability_confirmed", findings_dir=tmp_path, force=True)
    text = result.read_text(encoding="utf-8")
    assert "trust_level: reliability_confirmed" in text


def test_promote_retracted_is_terminal(tmp_path: Path) -> None:
    _write_finding(tmp_path, slug="F-099", trust_level="retracted", has_depends_on=True)
    with pytest.raises(ValueError, match="retracted"):
        promote_finding("F-099", "deployable", findings_dir=tmp_path)


# ---------------------------------------------------------------------------
# 4. promote refuses when depends_on is missing (for non-unverified targets)
# ---------------------------------------------------------------------------


def test_promote_refuses_without_depends_on(tmp_path: Path) -> None:
    _write_finding(tmp_path, slug="F-099", trust_level="unverified", has_depends_on=False)
    with pytest.raises(ValueError, match="depends_on"):
        promote_finding("F-099", "verified", findings_dir=tmp_path)


def test_promote_retraction_does_not_require_depends_on(tmp_path: Path) -> None:
    _write_finding(tmp_path, slug="F-099", trust_level="unverified", has_depends_on=False)
    result = promote_finding("F-099", "retracted", findings_dir=tmp_path)
    assert result.exists()


# ---------------------------------------------------------------------------
# 5. backfill is idempotent
# ---------------------------------------------------------------------------


def _write_finding_no_trust(tmp_path: Path, slug: str) -> Path:
    text = textwrap.dedent(f"""\
        # {slug}: A finding with no trust_level

        Date: 2026-06-14
        Confidence: high
        Source: EXP-TEST-001

        ## Claim

        Body.
    """)
    p = tmp_path / f"{slug}-test.md"
    p.write_text(text, encoding="utf-8")
    return p


def test_backfill_adds_trust_level(tmp_path: Path) -> None:
    _write_finding_no_trust(tmp_path, "F-001")
    _write_finding_no_trust(tmp_path, "F-002")

    updated, already_set = backfill_trust(tmp_path)
    assert updated == 2
    assert already_set == 0

    for slug in ("F-001", "F-002"):
        p = tmp_path / f"{slug}-test.md"
        assert "trust_level: unverified" in p.read_text(encoding="utf-8")


def test_backfill_is_idempotent(tmp_path: Path) -> None:
    _write_finding_no_trust(tmp_path, "F-001")
    backfill_trust(tmp_path)
    updated2, already_set2 = backfill_trust(tmp_path)
    assert updated2 == 0
    assert already_set2 == 1


def test_backfill_skips_files_with_existing_trust(tmp_path: Path) -> None:
    _write_finding(tmp_path, slug="F-001", trust_level="verified", has_depends_on=True)
    _write_finding_no_trust(tmp_path, "F-002")

    updated, already_set = backfill_trust(tmp_path)
    assert updated == 1
    assert already_set == 1

    # F-001 stays verified
    f1 = tmp_path / "F-001-test-claim.md"
    assert "trust_level: verified" in f1.read_text(encoding="utf-8")
    assert "trust_level: unverified" not in f1.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 6. parse_finding reads trust_level correctly
# ---------------------------------------------------------------------------


def test_parse_finding_reads_trust_level(tmp_path: Path) -> None:
    p = _write_finding(tmp_path, trust_level="reliability_confirmed")
    pf = parse_finding(p)
    assert pf is not None
    assert pf.trust_level == "reliability_confirmed"


def test_parse_finding_defaults_to_unverified(tmp_path: Path) -> None:
    p = _write_finding_no_trust(tmp_path, "F-001")
    pf = parse_finding(p)
    assert pf is not None
    assert pf.trust_level == "unverified"


def test_parse_finding_reads_depends_on(tmp_path: Path) -> None:
    p = _write_finding(tmp_path, has_depends_on=True, source="EXP-007-reliable-baseline")
    pf = parse_finding(p)
    assert pf is not None
    assert pf.depends_on == "EXP-007-reliable-baseline"
