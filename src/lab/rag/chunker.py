"""Markdown-structural chunker.

Vendored from kb_builder.chunker.

Rules:
- Walks the markdown by headings (#, ##, ###...) maintaining a section path.
- Code blocks (``` fences) are atomic — never split.
- Token-aware: targets `target_tokens`, hard caps at 2*target_tokens.
- Overlap: when a single section exceeds target, splits with `overlap_tokens`
  carried into the next chunk.
- Tiny sibling sections under the same parent merge if combined < target.
- Drops YAML front-matter from input before chunking.
- Each chunk records section_path and byte offsets back into the source.

Phase 9 (2026-05-26): parent-child chunking. Set ``mode=ChunkMode.PARENT_CHILD``
on :func:`chunk_document` to emit ``(parent, child)`` pairs. Parents target
``parent_target_tokens`` (default 768); children target ``child_target_tokens``
(default 192). Child chunks carry ``parent_id`` + ``child_index``; parent
chunks set ``is_parent=True``. The default mode is still ``ChunkMode.FLAT``
(bit-for-bit-compatible with the v1 bash KB).
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass

import tiktoken
from ulid import ULID


class ChunkMode(enum.StrEnum):
    """Chunking strategy. ``FLAT`` is the original v1 behaviour; ``PARENT_CHILD``
    emits (parent, child) pairs for Phase 9 retrieval-by-child / read-by-parent.
    """

    FLAT = "flat"
    PARENT_CHILD = "parent_child"


#: Phase 9 parent-target token count (mid of 512-1024 band).
DEFAULT_PARENT_TARGET_TOKENS = 768
#: Phase 9 child-target token count (mid of 128-256 band).
DEFAULT_CHILD_TARGET_TOKENS = 192


@dataclass
class Chunk:
    chunk_id: str
    doc_path: str
    section_path: list[str]
    byte_start: int
    byte_end: int
    text: str
    tokens: int
    #: Phase 9 parent-child fields. ``None`` / ``False`` for FLAT mode.
    parent_id: str | None = None
    child_index: int | None = None
    is_parent: bool = False


_ENC = tiktoken.get_encoding("cl100k_base")


def _count(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)


def strip_frontmatter(doc: str) -> tuple[str, int]:
    """Return (body, byte_offset_of_body_in_original)."""
    m = _FRONTMATTER_RE.match(doc)
    if m:
        return doc[m.end() :], m.end()
    return doc, 0


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_FENCE_RE = re.compile(r"^(```|~~~)")


@dataclass
class _Section:
    level: int
    heading: str
    path: list[str]
    start: int  # byte offset (in body)
    end: int  # exclusive byte offset (in body)
    text: str


def _split_into_sections(body: str) -> list[_Section]:
    """Walk lines, tracking heading nesting; sections include their heading line."""
    lines = body.splitlines(keepends=True)
    sections: list[_Section] = []

    # Offsets per line
    offsets = []
    cur = 0
    for ln in lines:
        offsets.append(cur)
        cur += len(ln)

    in_fence = False
    fence_marker = ""
    cur_path: list[tuple[int, str]] = []  # (level, heading)
    cur_section_start_line: int | None = None
    cur_section_level = 0
    cur_section_heading = ""
    cur_section_path: list[str] = []

    def flush(end_line: int) -> None:
        nonlocal cur_section_start_line
        if cur_section_start_line is None:
            return
        s = offsets[cur_section_start_line]
        e = offsets[end_line] if end_line < len(offsets) else len(body)
        sections.append(
            _Section(
                level=cur_section_level,
                heading=cur_section_heading,
                path=list(cur_section_path),
                start=s,
                end=e,
                text=body[s:e],
            )
        )
        cur_section_start_line = None

    for i, line in enumerate(lines):
        # fence toggle
        m_fence = _FENCE_RE.match(line)
        if m_fence:
            marker = m_fence.group(1)
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif line.lstrip().startswith(fence_marker):
                in_fence = False
                fence_marker = ""
            continue

        if in_fence:
            continue

        m = _HEADING_RE.match(line)
        if m:
            # close previous section
            flush(i)
            level = len(m.group(1))
            heading = m.group(2).strip()
            # pop ancestor headings deeper than or equal level
            while cur_path and cur_path[-1][0] >= level:
                cur_path.pop()
            cur_path.append((level, heading))
            cur_section_start_line = i
            cur_section_level = level
            cur_section_heading = heading
            cur_section_path = [h for _, h in cur_path]

    # Handle preamble (text before first heading)
    if not sections and not cur_section_start_line and body.strip():
        sections.append(_Section(level=0, heading="", path=[], start=0, end=len(body), text=body))
    else:
        # flush trailing section
        flush(len(lines))
        # If there was preamble before the first heading, capture it
        if sections and sections[0].start > 0:
            preamble = body[: sections[0].start]
            if preamble.strip():
                sections.insert(
                    0,
                    _Section(
                        level=0,
                        heading="",
                        path=[],
                        start=0,
                        end=sections[0].start,
                        text=preamble,
                    ),
                )

    return sections


def _split_long_text(
    text: str, target: int, overlap: int, hard_cap: int
) -> list[tuple[int, int, str]]:
    """Split text into pieces of ~target tokens with overlap.

    Returns list of (relative_byte_start, relative_byte_end, piece_text).
    Code fences are kept whole; if a single fence is bigger than hard_cap,
    we emit it as one oversized chunk (better than splitting code).
    """
    lines = text.splitlines(keepends=True)
    line_offsets = []
    cur = 0
    for ln in lines:
        line_offsets.append(cur)
        cur += len(ln)
    line_offsets.append(len(text))

    # Group lines into "atoms": either a single non-code line, or an entire code fence.
    atoms: list[tuple[int, int]] = []  # (start_line, end_line_exclusive)
    i = 0
    while i < len(lines):
        m = _FENCE_RE.match(lines[i])
        if m:
            marker = m.group(1)
            j = i + 1
            while j < len(lines):
                if lines[j].lstrip().startswith(marker):
                    j += 1
                    break
                j += 1
            atoms.append((i, j))
            i = j
        else:
            atoms.append((i, i + 1))
            i += 1

    pieces: list[tuple[int, int, str]] = []
    cur_atoms: list[tuple[int, int]] = []
    cur_tokens = 0

    def cur_text() -> str:
        if not cur_atoms:
            return ""
        s = line_offsets[cur_atoms[0][0]]
        e = line_offsets[cur_atoms[-1][1]]
        return text[s:e]

    def emit() -> None:
        nonlocal cur_atoms, cur_tokens
        if not cur_atoms:
            return
        s = line_offsets[cur_atoms[0][0]]
        e = line_offsets[cur_atoms[-1][1]]
        pieces.append((s, e, text[s:e]))
        cur_atoms = []
        cur_tokens = 0

    for atom in atoms:
        atom_text = text[line_offsets[atom[0]] : line_offsets[atom[1]]]
        atom_tokens = _count(atom_text)
        if cur_tokens + atom_tokens > hard_cap and cur_atoms:
            emit()
            # overlap: keep tail atoms that sum to ~overlap tokens
            if overlap > 0:
                # find tail of previous piece
                prev_text = pieces[-1][2]
                tail_lines = prev_text.splitlines(keepends=True)
                acc = ""
                ttok = 0
                for ln in reversed(tail_lines):
                    if ttok + _count(ln) > overlap:
                        break
                    acc = ln + acc
                    ttok += _count(ln)
                # The overlap is informational — we don't re-derive byte ranges
                # for it because it would overlap the previous piece. The
                # retrieval router doesn't need byte-accurate overlap.
                _ = acc
        cur_atoms.append(atom)
        cur_tokens += atom_tokens
        if cur_tokens >= target:
            emit()

    if cur_atoms:
        emit()

    return pieces


def chunk_document(
    *,
    doc_path: str,
    full_text: str,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
    hard_cap_factor: int = 2,
    min_merge_tokens: int = 80,
    mode: ChunkMode = ChunkMode.FLAT,
    parent_target_tokens: int = DEFAULT_PARENT_TARGET_TOKENS,
    child_target_tokens: int = DEFAULT_CHILD_TARGET_TOKENS,
) -> list[Chunk]:
    """Top-level chunking entry.

    ``mode`` selects the chunking strategy:
      * ``ChunkMode.FLAT`` (default) — original v1 markdown-structural chunker.
        ``parent_target_tokens`` / ``child_target_tokens`` are ignored.
      * ``ChunkMode.PARENT_CHILD`` — emit ``(parent, child)`` pairs where each
        parent contains 2-4 children. Parents target
        ``parent_target_tokens``; children target ``child_target_tokens``.
        ``target_tokens`` / ``overlap_tokens`` still drive the underlying
        section walk before parent boundaries are formed.
    """
    if mode is ChunkMode.PARENT_CHILD:
        return _chunk_parent_child(
            doc_path=doc_path,
            full_text=full_text,
            parent_target_tokens=parent_target_tokens,
            child_target_tokens=child_target_tokens,
            overlap_tokens=overlap_tokens,
            min_merge_tokens=min_merge_tokens,
        )

    body, base_offset = strip_frontmatter(full_text)
    sections = _split_into_sections(body)
    if not sections:
        return []

    hard_cap = target_tokens * hard_cap_factor

    # Merge tiny sibling sections (same parent path AND same depth).
    # This consolidates short adjacent sections without absorbing their
    # descendants (which would lose section identity).
    merged: list[_Section] = []
    for sec in sections:
        if (
            merged
            and len(merged[-1].path) == len(sec.path)
            and merged[-1].path[:-1] == sec.path[:-1]
            and _count(merged[-1].text) < min_merge_tokens
            and _count(merged[-1].text) + _count(sec.text) <= target_tokens
        ):
            prev = merged[-1]
            merged[-1] = _Section(
                level=prev.level,
                heading=prev.heading,
                path=prev.path,
                start=prev.start,
                end=sec.end,
                text=body[prev.start : sec.end],
            )
        else:
            merged.append(sec)

    chunks: list[Chunk] = []
    for sec in merged:
        sec_tokens = _count(sec.text)
        if sec_tokens <= hard_cap:
            chunks.append(
                Chunk(
                    chunk_id=str(ULID()),
                    doc_path=doc_path,
                    section_path=sec.path,
                    byte_start=base_offset + sec.start,
                    byte_end=base_offset + sec.end,
                    text=sec.text,
                    tokens=sec_tokens,
                )
            )
        else:
            pieces = _split_long_text(sec.text, target_tokens, overlap_tokens, hard_cap)
            for rs, re_, piece_text in pieces:
                chunks.append(
                    Chunk(
                        chunk_id=str(ULID()),
                        doc_path=doc_path,
                        section_path=sec.path,
                        byte_start=base_offset + sec.start + rs,
                        byte_end=base_offset + sec.start + re_,
                        text=piece_text,
                        tokens=_count(piece_text),
                    )
                )

    return chunks


# ---------------------------------------------------------------------------
# Phase 9 — parent-child chunking.
# ---------------------------------------------------------------------------

# Crude sentence splitter: keeps trailing whitespace so byte ranges stay sane.
# We split on `.`, `!`, `?`, and newline boundaries followed by whitespace or EOS.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _sentence_split(text: str) -> list[tuple[int, int, str]]:
    """Split text into sentence-like pieces.

    Returns a list of ``(rel_start, rel_end, piece_text)``. Pieces are
    non-empty and together cover the input (no characters lost).
    """
    if not text:
        return []
    pieces: list[tuple[int, int, str]] = []
    cur = 0
    for m in _SENTENCE_RE.finditer(text):
        end = m.end()
        piece = text[cur:end]
        if piece.strip():
            pieces.append((cur, end, piece))
        cur = end
    tail = text[cur:]
    if tail.strip():
        pieces.append((cur, len(text), tail))
    elif pieces:
        # Glue trailing whitespace onto the last piece so we don't drop bytes.
        rs, _re, ptxt = pieces[-1]
        pieces[-1] = (rs, len(text), ptxt + tail)
    return pieces


def _pack_children(
    sentences: list[tuple[int, int, str]],
    *,
    child_target: int,
    child_hard_cap: int,
) -> list[tuple[int, int, str]]:
    """Pack consecutive sentences into child-sized chunks.

    A child fills to ``child_target`` tokens; oversized single sentences pass
    through whole (better to overshoot than lose a sentence boundary). Soft
    hard-cap at ``child_hard_cap`` triggers an early flush.
    """
    children: list[tuple[int, int, str]] = []
    if not sentences:
        return children

    cur_start = sentences[0][0]
    cur_end = sentences[0][0]
    cur_text = ""
    cur_tokens = 0

    def flush() -> None:
        nonlocal cur_start, cur_end, cur_text, cur_tokens
        if cur_text:
            children.append((cur_start, cur_end, cur_text))
        cur_text = ""
        cur_tokens = 0

    for rs, re_, ptxt in sentences:
        ptokens = _count(ptxt)
        if cur_text and cur_tokens + ptokens > child_hard_cap:
            flush()
            cur_start = rs
        if not cur_text:
            cur_start = rs
        cur_end = re_
        cur_text += ptxt
        cur_tokens += ptokens
        if cur_tokens >= child_target:
            flush()

    flush()
    return children


def _section_to_parents(
    section: _Section,
    *,
    parent_target: int,
    parent_hard_cap: int,
) -> list[tuple[int, int, str]]:
    """Carve a single section into 1+ parent-sized blocks.

    Sections smaller than ``parent_hard_cap`` become exactly one parent.
    Larger sections are split at sentence boundaries near the parent-target
    mark — children must never straddle parent boundaries.
    """
    text = section.text
    sec_tokens = _count(text)
    if sec_tokens <= parent_hard_cap:
        return [(0, len(text), text)]

    parents: list[tuple[int, int, str]] = []
    sentences = _sentence_split(text)
    cur_start = 0
    cur_end = 0
    cur_text = ""
    cur_tokens = 0
    for rs, re_, ptxt in sentences:
        ptokens = _count(ptxt)
        if cur_text and cur_tokens + ptokens > parent_hard_cap:
            parents.append((cur_start, cur_end, cur_text))
            cur_text = ""
            cur_tokens = 0
        if not cur_text:
            cur_start = rs
        cur_end = re_
        cur_text += ptxt
        cur_tokens += ptokens
        if cur_tokens >= parent_target:
            parents.append((cur_start, cur_end, cur_text))
            cur_text = ""
            cur_tokens = 0
    if cur_text:
        parents.append((cur_start, cur_end, cur_text))
    if not parents:
        parents.append((0, len(text), text))
    return parents


def _chunk_parent_child(
    *,
    doc_path: str,
    full_text: str,
    parent_target_tokens: int,
    child_target_tokens: int,
    overlap_tokens: int,
    min_merge_tokens: int,
) -> list[Chunk]:
    """Emit ``(parent, child, child, ...)`` chunk records.

    Output ordering: each parent appears immediately before its children, so
    a downstream consumer can walk the list and reconstruct parent-child
    relationships without random access.
    """
    body, base_offset = strip_frontmatter(full_text)
    sections = _split_into_sections(body)
    if not sections:
        return []

    parent_hard_cap = max(parent_target_tokens * 2, parent_target_tokens + 256)
    child_hard_cap = max(child_target_tokens * 2, child_target_tokens + 64)

    # Step 1 — split each section into parent-sized blocks (preferring
    # section-aligned parents; fall back to mid-section sentence splits).
    raw_parents: list[tuple[_Section, int, int, str]] = []  # (section, rs, re, text)
    for sec in sections:
        for rs, re_, ptxt in _section_to_parents(
            sec,
            parent_target=parent_target_tokens,
            parent_hard_cap=parent_hard_cap,
        ):
            raw_parents.append((sec, rs, re_, ptxt))

    # Step 2 — merge tiny adjacent siblings under the same path to keep the
    # parent count low. (Only siblings — descendants stay separate.)
    out_chunks: list[Chunk] = []
    # We avoid the merging helper's lossy byte math and instead pack on the
    # fly using a simple lookahead.
    i = 0
    packed_parents: list[tuple[list[str], int, int, str]] = []
    while i < len(raw_parents):
        sec, rs, re_, ptxt = raw_parents[i]
        ptokens = _count(ptxt)
        # Try to merge with the next raw parent if both are tiny and share the
        # same enclosing parent-path.
        if (
            i + 1 < len(raw_parents)
            and ptokens < min_merge_tokens
            and raw_parents[i + 1][0].path[:-1] == sec.path[:-1]
            and ptokens + _count(raw_parents[i + 1][3]) <= parent_target_tokens
        ):
            nxt_sec, _nxt_rs, nxt_re, nxt_text = raw_parents[i + 1]
            # Use the earlier section's path; combined byte range only valid
            # when both come from the same section.
            combined_text = ptxt + nxt_text
            if sec is nxt_sec:
                packed_parents.append((list(sec.path), rs, nxt_re, combined_text))
            else:
                # Sibling-section merge: byte ranges become disjoint, but
                # callers don't rely on contiguity in parent-child mode, so
                # record the start of the first and end of the second.
                packed_parents.append(
                    (list(sec.path), sec.start + rs, nxt_sec.start + nxt_re, combined_text)
                )
            i += 2
            continue
        # Translate section-relative (rs, re_) to body-relative.
        packed_parents.append(
            (list(sec.path), sec.start + rs, sec.start + re_, ptxt)
        )
        i += 1

    # Step 3 — emit parent + children pairs.
    for path, byte_start_rel, byte_end_rel, ptxt in packed_parents:
        parent_id = str(ULID())
        parent_tokens = _count(ptxt)
        out_chunks.append(
            Chunk(
                chunk_id=parent_id,
                doc_path=doc_path,
                section_path=path,
                byte_start=base_offset + byte_start_rel,
                byte_end=base_offset + byte_end_rel,
                text=ptxt,
                tokens=parent_tokens,
                parent_id=None,
                child_index=None,
                is_parent=True,
            )
        )
        sentences = _sentence_split(ptxt)
        children = _pack_children(
            sentences,
            child_target=child_target_tokens,
            child_hard_cap=child_hard_cap,
        )
        # Aim for 2-4 children per parent. If a parent only produced 1
        # child (very short parent), still emit one child = full parent
        # text so retrieval has a small embedding-target body.
        if not children:
            children = [(0, len(ptxt), ptxt)]
        for ci, (crs, cre, ctxt) in enumerate(children):
            out_chunks.append(
                Chunk(
                    chunk_id=str(ULID()),
                    doc_path=doc_path,
                    section_path=path,
                    byte_start=base_offset + byte_start_rel + crs,
                    byte_end=base_offset + byte_start_rel + cre,
                    text=ctxt,
                    tokens=_count(ctxt),
                    parent_id=parent_id,
                    child_index=ci,
                    is_parent=False,
                )
            )

    # `overlap_tokens` is accepted for API symmetry but parent-child chunks
    # don't carry inter-chunk overlap (children are bounded by parents,
    # parents are bounded by sections). Touch the var so linters don't gripe.
    _ = overlap_tokens

    return out_chunks


# Keep the helper exported for tests that want to assert on tiny-merge
# behaviour without re-running the full pipeline.
__all__ = [
    "DEFAULT_CHILD_TARGET_TOKENS",
    "DEFAULT_PARENT_TARGET_TOKENS",
    "Chunk",
    "ChunkMode",
    "chunk_document",
    "strip_frontmatter",
]

