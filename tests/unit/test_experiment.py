"""Experiment plan validation tests (no DB)."""

from __future__ import annotations

from pathlib import Path

from lab.experiment import _section_headings, _slug_from_text, validate_plan

GOOD_PLAN = """# EXP-042: example experiment

Date: 2026-05-25
Status: planned

## Hypothesis
foo

## Method
bar

## Success / failure criteria (defined before running)
baz

## Kill criteria
qux
"""


MISSING_SECTIONS = """# EXP-099: incomplete

## Hypothesis
foo

## Method
bar
"""


def test_section_headings_tolerates_parentheticals() -> None:
    headings = _section_headings(GOOD_PLAN)
    assert "success / failure criteria" in headings
    assert "hypothesis" in headings


def test_section_headings_tolerates_em_dash_subtitle() -> None:
    text = "## Method — what we'll do\n\nbody\n"
    headings = _section_headings(text)
    assert "method" in headings


def test_slug_from_h1() -> None:
    assert _slug_from_text("EXP-042: foo", "x.md") == "EXP-042"


def test_slug_from_filename_fallback() -> None:
    assert _slug_from_text(None, "EXP-123-something.md") == "EXP-123"


def test_validate_plan_good(tmp_path: Path) -> None:
    p = tmp_path / "EXP-042-x.md"
    p.write_text(GOOD_PLAN, encoding="utf-8")
    v = validate_plan(p)
    assert v.slug == "EXP-042"
    assert not v.missing_sections


def test_validate_plan_missing_sections(tmp_path: Path) -> None:
    p = tmp_path / "EXP-099-x.md"
    p.write_text(MISSING_SECTIONS, encoding="utf-8")
    v = validate_plan(p)
    assert set(v.missing_sections) == {"success / failure criteria", "kill criteria"}
    assert not v.ok
