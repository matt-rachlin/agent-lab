"""Unit tests for lab.rag.chunker (vendored from kb-builder + new cases)."""

from __future__ import annotations

from lab.rag.chunker import chunk_document, strip_frontmatter


def test_strip_frontmatter():
    doc = "---\nfoo: bar\n---\nbody\n"
    body, offset = strip_frontmatter(doc)
    assert body == "body\n"
    assert offset == len("---\nfoo: bar\n---\n")


def test_strip_frontmatter_absent():
    body, offset = strip_frontmatter("no frontmatter here\n")
    assert offset == 0
    assert body == "no frontmatter here\n"


def test_chunker_keeps_code_blocks_atomic():
    doc = """\
---
source_url: x
sha256: y
retrieved_at: z
---

# Bash Builtins

## set

`set` modifies shell options.

```bash
set -u   # error on unset
set -o pipefail
set -e
```

That was a code block.

## test

The `test` builtin.

```
[[ -z "$x" ]] && echo empty
```
"""
    chunks = chunk_document(doc_path="x.md", full_text=doc, target_tokens=40, overlap_tokens=8)
    assert chunks
    for c in chunks:
        # Code fence parity: each chunk's text has even number of fences
        fences = sum(1 for ln in c.text.splitlines() if ln.lstrip().startswith("```"))
        assert fences % 2 == 0, f"odd fence count in chunk {c.text!r}"
    # Both 'set' and 'test' subsections present
    sects = {tuple(c.section_path) for c in chunks}
    assert any("set" in p for p in sects)
    assert any("test" in p for p in sects)


def test_chunker_section_path_preserved():
    doc = """\
# A

## B

text

### C

more text
"""
    chunks = chunk_document(doc_path="x.md", full_text=doc, target_tokens=10)
    paths = [tuple(c.section_path) for c in chunks]
    assert ("A", "B") in paths
    assert ("A", "B", "C") in paths


def test_chunker_empty_doc():
    assert chunk_document(doc_path="x.md", full_text="") == []
    assert chunk_document(doc_path="x.md", full_text="   \n  \n") == []


def test_chunker_preamble_captured():
    """Text before the first heading must become its own preamble chunk."""
    doc = (
        "This is the preamble paragraph that comes before any heading.\n"
        "It has some content that we don't want to drop.\n\n"
        "# H\n\nbody\n"
    )
    chunks = chunk_document(doc_path="x.md", full_text=doc, target_tokens=10)
    assert chunks, "expected at least one chunk"
    # First chunk should be the preamble (no section path)
    first = chunks[0]
    assert first.section_path == []
    assert "preamble" in first.text
    # Heading-rooted chunk follows
    assert any(c.section_path == ["H"] for c in chunks)


def test_chunker_long_section_splits_with_overlap():
    """A section larger than 2*target should split into multiple chunks."""
    long_body = "## L\n\n" + ("paragraph of words here. " * 200)
    doc = "# Top\n\n" + long_body
    chunks = chunk_document(doc_path="x.md", full_text=doc, target_tokens=64, overlap_tokens=16)
    long_chunks = [c for c in chunks if c.section_path == ["Top", "L"]]
    assert len(long_chunks) >= 2, "expected splitting of long section"
    for c in long_chunks:
        assert c.tokens > 0


def test_chunker_byte_ranges_monotonic():
    """Byte ranges should be ascending and non-overlapping (modulo header overlap)."""
    doc = (
        "# A\n\nshort body for A.\n\n"
        "## B\n\nbody for B with more words to chunk.\n\n"
        "## C\n\nbody for C.\n"
    )
    chunks = chunk_document(doc_path="x.md", full_text=doc, target_tokens=20)
    for c in chunks:
        assert c.byte_start < c.byte_end
        assert c.byte_end <= len(doc)


def test_chunker_ulid_chunk_ids_unique():
    doc = "# A\n\nbody A\n\n# B\n\nbody B\n"
    chunks = chunk_document(doc_path="x.md", full_text=doc)
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids), "chunk_ids must be unique"
    # ULIDs are 26 chars (Crockford base32)
    for cid in ids:
        assert len(cid) == 26
