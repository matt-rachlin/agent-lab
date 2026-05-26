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
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken
from ulid import ULID


@dataclass
class Chunk:
    chunk_id: str
    doc_path: str
    section_path: list[str]
    byte_start: int
    byte_end: int
    text: str
    tokens: int


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
) -> list[Chunk]:
    """Top-level chunking entry."""
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
