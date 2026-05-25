"""Finding markdown parser tests (no DB)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from lab.finding import new_finding, parse_finding

GOOD_FINDING = """# F-042: a useful claim

Date: 2026-05-25
Confidence: high
Source: EXP-FOO-001

## Claim
The thing.
"""


def test_parse_finding(tmp_path: Path) -> None:
    p = tmp_path / "F-042-x.md"
    p.write_text(GOOD_FINDING, encoding="utf-8")
    parsed = parse_finding(p)
    assert parsed is not None
    assert parsed.slug == "F-042"
    assert parsed.claim == "a useful claim"
    assert parsed.confidence == "high"
    assert parsed.date == date(2026, 5, 25)
    assert parsed.source_exp_slug is not None
    assert "FOO-001" in parsed.source_exp_slug


def test_parse_finding_missing_h1(tmp_path: Path) -> None:
    p = tmp_path / "no-h1.md"
    p.write_text("just text\n", encoding="utf-8")
    assert parse_finding(p) is None


def test_new_finding_creates_file(tmp_path: Path) -> None:
    out = new_finding("F-077", "my-test-claim", dir_=tmp_path)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "F-077" in text
    assert "my-test-claim" in text


def test_new_finding_rejects_bad_slug(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="F-NNN"):
        new_finding("not-a-slug", dir_=tmp_path)


def test_new_finding_refuses_overwrite(tmp_path: Path) -> None:
    new_finding("F-077", "claim", dir_=tmp_path)
    with pytest.raises(FileExistsError):
        new_finding("F-077", "claim", dir_=tmp_path)
