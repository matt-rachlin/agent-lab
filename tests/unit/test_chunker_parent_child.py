"""Unit tests for Phase 9 parent-child chunking.

The chunker emits ``(parent, child, child, ...)`` records when invoked with
``ChunkMode.PARENT_CHILD``. These tests cover ordering, boundary cases,
token-count edges, and the section-header fallback. We don't touch real
embedders or LanceDB.
"""

from __future__ import annotations

from lab.rag.chunker import (
    DEFAULT_CHILD_TARGET_TOKENS,
    DEFAULT_PARENT_TARGET_TOKENS,
    ChunkMode,
    chunk_document,
)


def _by_parent(chunks):  # type: ignore[no-untyped-def]
    """Return ``{parent_id: [child_chunks]}`` grouped from a chunk list."""
    parents: dict[str, list] = {}
    for c in chunks:
        if c.is_parent:
            parents.setdefault(c.chunk_id, [])
    for c in chunks:
        if c.parent_id is not None:
            parents.setdefault(c.parent_id, []).append(c)
    return parents


def test_parent_child_emits_parent_before_children() -> None:
    doc = "# A\n\n" + ("Sentence one. " * 60)
    chunks = chunk_document(
        doc_path="x.md",
        full_text=doc,
        mode=ChunkMode.PARENT_CHILD,
        parent_target_tokens=200,
        child_target_tokens=50,
    )
    # First chunk must be a parent. Every child's parent_id appears earlier
    # in the list than the child itself.
    assert chunks
    assert chunks[0].is_parent is True
    seen_parents: set[str] = set()
    for c in chunks:
        if c.is_parent:
            seen_parents.add(c.chunk_id)
        else:
            assert c.parent_id in seen_parents


def test_parent_child_children_carry_index() -> None:
    """child_index is dense (0, 1, 2, ...) within each parent."""
    doc = "# H\n\n" + (". ".join(f"word{i} fact {i}" for i in range(200))) + "."
    chunks = chunk_document(
        doc_path="x.md",
        full_text=doc,
        mode=ChunkMode.PARENT_CHILD,
        parent_target_tokens=200,
        child_target_tokens=40,
    )
    groups = _by_parent(chunks)
    for _pid, kids in groups.items():
        kid_indices = [k.child_index for k in kids]
        assert kid_indices == list(range(len(kid_indices)))


def test_parent_child_parent_path_inherits_section() -> None:
    """Parents (and their children) carry the section_path of the source
    section."""
    doc = (
        "# Top\n\n"
        "## Body\n\n"
        + (". ".join(f"alpha beta gamma delta {i}" for i in range(80)))
        + ".\n"
    )
    chunks = chunk_document(
        doc_path="x.md",
        full_text=doc,
        mode=ChunkMode.PARENT_CHILD,
        parent_target_tokens=150,
        child_target_tokens=40,
    )
    body_chunks = [c for c in chunks if c.section_path == ["Top", "Body"]]
    assert body_chunks
    # Both parents and children inherit the section path.
    parents = [c for c in body_chunks if c.is_parent]
    kids = [c for c in body_chunks if not c.is_parent]
    assert parents
    assert kids
    for c in body_chunks:
        assert c.section_path == ["Top", "Body"]


def test_parent_child_small_section_one_parent() -> None:
    """A section that fits in a single parent yields exactly one parent."""
    doc = "# Tiny\n\nShort sentence one. Short sentence two."
    chunks = chunk_document(
        doc_path="x.md",
        full_text=doc,
        mode=ChunkMode.PARENT_CHILD,
        parent_target_tokens=DEFAULT_PARENT_TARGET_TOKENS,
        child_target_tokens=DEFAULT_CHILD_TARGET_TOKENS,
    )
    parents = [c for c in chunks if c.is_parent]
    assert len(parents) == 1
    # That parent yields at least one child.
    children = [c for c in chunks if c.parent_id == parents[0].chunk_id]
    assert children, "expected at least one child for the lone parent"


def test_parent_child_large_section_splits_into_many_parents() -> None:
    """A section >> parent_target should split into multiple parents."""
    big = ". ".join(f"the quick brown fox jumps {i}" for i in range(800)) + "."
    doc = f"# Big\n\n{big}\n"
    chunks = chunk_document(
        doc_path="x.md",
        full_text=doc,
        mode=ChunkMode.PARENT_CHILD,
        parent_target_tokens=300,
        child_target_tokens=80,
    )
    parents = [c for c in chunks if c.is_parent]
    assert len(parents) >= 2


def test_parent_child_byte_ranges_inside_doc() -> None:
    """All byte ranges fall within [0, len(doc)] and are non-empty."""
    doc = "# Section\n\n" + ("Hello world. " * 30)
    chunks = chunk_document(
        doc_path="x.md",
        full_text=doc,
        mode=ChunkMode.PARENT_CHILD,
        parent_target_tokens=100,
        child_target_tokens=20,
    )
    for c in chunks:
        assert 0 <= c.byte_start < c.byte_end
        assert c.byte_end <= len(doc)


def test_parent_child_empty_doc() -> None:
    assert (
        chunk_document(
            doc_path="x.md",
            full_text="",
            mode=ChunkMode.PARENT_CHILD,
        )
        == []
    )


def test_flat_mode_unchanged_bitfor_bit() -> None:
    """FLAT mode is the v1 baseline — no parent_id / child_index / is_parent
    populated, even though the fields now exist on the dataclass."""
    doc = (
        "# Top\n\n"
        "## A\n\nbody A.\n\n"
        "## B\n\nbody B with words.\n\n"
        "## C\n\nbody C.\n"
    )
    chunks = chunk_document(
        doc_path="x.md",
        full_text=doc,
        target_tokens=20,
        mode=ChunkMode.FLAT,
    )
    assert chunks
    for c in chunks:
        assert c.is_parent is False
        assert c.parent_id is None
        assert c.child_index is None


def test_parent_child_default_targets_constants() -> None:
    """The default constants should land inside the spec'd bands."""
    assert 512 <= DEFAULT_PARENT_TARGET_TOKENS <= 1024
    assert 128 <= DEFAULT_CHILD_TARGET_TOKENS <= 256
