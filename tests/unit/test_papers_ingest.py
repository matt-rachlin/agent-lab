"""Unit tests for tools/add_paper.py — no network."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest
import yaml

# Load tools/add_paper.py as a module (it's not under src/).
_SPEC = importlib.util.spec_from_file_location(
    "add_paper",
    Path(__file__).resolve().parents[2] / "tools" / "add_paper.py",
)
assert _SPEC is not None
assert _SPEC.loader is not None
add_paper_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["add_paper"] = add_paper_mod
_SPEC.loader.exec_module(add_paper_mod)


# Minimal arXiv Atom XML response — only the fields we read.
ARXIV_XML = (
    b"<?xml version='1.0' encoding='UTF-8'?>"
    b'<feed xmlns="http://www.w3.org/2005/Atom">'
    b"<entry>"
    b"<title>From Local to Global: A GraphRAG Approach</title>"
    b"<summary>We introduce GraphRAG, a method for global summarization "
    b"over a corpus using a graph-based index.</summary>"
    b"<published>2024-04-24T17:30:00Z</published>"
    b"<author><name>Darren Edge</name></author>"
    b"<author><name>Ha Trinh</name></author>"
    b"</entry>"
    b"</feed>"
)


def _arxiv_fetcher(url: str) -> bytes:
    return ARXIV_XML


def test_parse_arxiv_id_bare() -> None:
    """Bare arxiv id like '2404.16130' is recognized."""
    ref = add_paper_mod.parse_identifier("2404.16130")
    assert ref.kind == "arxiv"
    assert ref.canonical_id == "2404.16130"
    assert ref.slug == "paper-2404-16130"


def test_parse_arxiv_id_strips_version_suffix() -> None:
    """'2404.16130v2' canonicalizes to '2404.16130'."""
    ref = add_paper_mod.parse_identifier("2404.16130v2")
    assert ref.kind == "arxiv"
    assert ref.canonical_id == "2404.16130"


def test_parse_arxiv_url() -> None:
    """arxiv URLs are recognized and parsed."""
    ref = add_paper_mod.parse_identifier("https://arxiv.org/abs/2404.16130")
    assert ref.kind == "arxiv"
    assert ref.canonical_id == "2404.16130"
    assert ref.slug == "paper-2404-16130"


def test_parse_doi() -> None:
    """DOI form '10.1145/...' is recognized."""
    ref = add_paper_mod.parse_identifier("10.1145/3589334.3645708")
    assert ref.kind == "doi"
    assert ref.canonical_id == "10.1145/3589334.3645708"
    assert ref.slug.startswith("paper-10-1145")
    assert "/" not in ref.slug
    assert "." not in ref.slug  # kebab-case for m docs


def test_parse_invalid_raises() -> None:
    """Garbage input raises ValueError."""
    with pytest.raises(ValueError, match="unrecognized identifier"):
        add_paper_mod.parse_identifier("not-a-real-id-at-all xxx ")


def test_add_paper_creates_dir_meta_notes(tmp_path: Path) -> None:
    """add_paper creates the dir, meta.yaml, notes.md, and a stub PDF."""
    paper_dir, created = add_paper_mod.add_paper(
        "2404.16130",
        papers_dir=tmp_path,
        meta_fetcher=_arxiv_fetcher,
        stub_pdf=True,
        today=date(2026, 5, 27),
    )
    assert created
    assert paper_dir == tmp_path / "paper-2404-16130"
    assert (paper_dir / "meta.yaml").exists()
    assert (paper_dir / "notes.md").exists()
    assert (paper_dir / "paper.pdf").exists()

    # meta.yaml: plain YAML (no doc-meta frontmatter), paper-specific fields
    meta = yaml.safe_load((paper_dir / "meta.yaml").read_text())
    assert meta["doc_id"] == "paper-2404-16130"
    assert meta["arxiv_id"] == "2404.16130"
    assert "Darren Edge" in meta["authors"]
    assert "Ha Trinh" in meta["authors"]
    assert meta["year"] == 2024
    assert "GraphRAG" in meta["title"]

    # notes.md: doc-meta frontmatter (indexable via m docs scan)
    notes_text = (paper_dir / "notes.md").read_text()
    assert notes_text.startswith("---\n")
    notes_fm = yaml.safe_load(notes_text.split("---")[1])
    assert notes_fm["doc_id"] == "paper-2404-16130"
    assert notes_fm["kind"] == "paper"
    assert notes_fm["zone"] == "research"


def test_add_paper_idempotent(tmp_path: Path) -> None:
    """Running twice on the same id is a no-op (no clobber)."""
    add_paper_mod.add_paper(
        "2404.16130",
        papers_dir=tmp_path,
        meta_fetcher=_arxiv_fetcher,
        stub_pdf=True,
        today=date(2026, 5, 27),
    )
    notes = tmp_path / "paper-2404-16130" / "notes.md"
    notes.write_text("# my hand notes\n")  # simulate user edits
    _paper_dir, created = add_paper_mod.add_paper(
        "2404.16130",
        papers_dir=tmp_path,
        meta_fetcher=_arxiv_fetcher,
        stub_pdf=True,
    )
    assert created is False
    assert notes.read_text() == "# my hand notes\n"  # preserved


def test_render_meta_yaml_quotes_tricky_title() -> None:
    """Titles with colons get YAML-quoted so they parse cleanly."""
    pm = add_paper_mod.PaperMeta(
        title="From Local to Global: A GraphRAG Approach",
        authors=["Darren Edge"],
        year=2024,
        venue="arxiv",
        abstract="x",
        arxiv_id="2404.16130",
    )
    ref = add_paper_mod.parse_identifier("2404.16130")
    text = add_paper_mod.render_meta_yaml(ref, pm)
    parsed = yaml.safe_load(text)
    assert parsed["title"] == "From Local to Global: A GraphRAG Approach"
    assert parsed["arxiv_id"] == "2404.16130"
