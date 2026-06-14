"""Scout scaffolding (ADR-010) — context assembly + title parsing (pure)."""

from lab.scout import _frontmatter_title, context_bundle


def test_frontmatter_title_from_yaml():
    md = "---\ntitle: 'ADR-009: the thing'\nzone: lab\n---\n# heading\nbody"
    assert _frontmatter_title(md) == "ADR-009: the thing"


def test_frontmatter_title_falls_back_to_heading():
    assert _frontmatter_title("# My Doc\ntext") == "My Doc"


def test_context_bundle_includes_charter_and_dedup_hint():
    out = context_bundle()
    assert "agent factory" in out  # charter mission present
    assert "DEDUP" in out  # dedup section present (recs exist or header)
