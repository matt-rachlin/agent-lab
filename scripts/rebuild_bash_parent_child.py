"""Rebuild the bash KB with Phase 9 parent-child chunking.

Reads source markdown from an existing v1 KB's ``sources/normalized/`` dir,
re-chunks each via :class:`lab.rag.chunker.ChunkMode.PARENT_CHILD`, embeds
the children + parents fresh, and writes a new LanceDB index + v2 manifest
into a sibling KB dir.

Source-of-truth-preserving: we do NOT re-fetch anything. The ``sha256`` /
``retrieved_at`` of each source carries over from the v1 manifest.

Usage:

    uv run python scripts/rebuild_bash_parent_child.py --dry-run
    uv run python scripts/rebuild_bash_parent_child.py

Optional flags let the script run against KBs other than bash and write to
an alternate destination, but the defaults are wired for the Phase 9 bash
rebuild.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from pathlib import Path

# Make the lab package importable when run via `uv run`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lab.rag import DEFAULT_EMBED_MODEL  # noqa: E402
from lab.rag.chunker import (  # noqa: E402
    DEFAULT_CHILD_TARGET_TOKENS,
    DEFAULT_PARENT_TARGET_TOKENS,
    ChunkMode,
    chunk_document,
)
from lab.rag.embedder import (  # noqa: E402
    build_bm25,
    embed_texts,
    sparse_for_text,
    tokenize_for_bm25,
)
from lab.rag.index import index_bytes, replace_table  # noqa: E402
from lab.rag.manifest import (  # noqa: E402
    Manifest,
    load_manifest,
    now_iso,
    write_manifest,
)


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _load_sources(src_kb: Path, manifest: Manifest) -> list[tuple[dict, str, Path]]:
    """Return [(source_entry_dict, full_text, abs_normalized_path), ...].

    Only includes sources with a non-null ``normalized`` field — same set the
    v1 builder embedded.
    """
    out: list[tuple[dict, str, Path]] = []
    for src in manifest.sources:
        if not src.normalized:
            continue
        norm_path = src_kb / src.normalized
        if not norm_path.exists():
            _eprint(f"WARN: missing normalized source: {norm_path}")
            continue
        text = norm_path.read_text(encoding="utf-8")
        out.append((src.model_dump(mode="json"), text, norm_path))
    return out


def _chunk_all(
    sources: list[tuple[dict, str, Path]],
    *,
    parent_target: int,
    child_target: int,
) -> list[tuple[dict, object]]:
    """Re-chunk every source. Returns ``[(source_entry_dict, Chunk), ...]``.

    Preserves chunker output order: each parent appears immediately before
    its children, so downstream code can pair them up.
    """
    out: list[tuple[dict, object]] = []
    for src_dict, text, norm_path in sources:
        # doc_path is relative to the KB dir per the v1 convention.
        doc_path = str(Path("sources/normalized") / norm_path.name)
        chunks = chunk_document(
            doc_path=doc_path,
            full_text=text,
            mode=ChunkMode.PARENT_CHILD,
            parent_target_tokens=parent_target,
            child_target_tokens=child_target,
        )
        for c in chunks:
            out.append((src_dict, c))
    return out


def _validate_parent_child_invariant(chunks_with_src: list[tuple[dict, object]]) -> tuple[int, int]:
    """Walk the parent-child chunk stream and confirm every child's text is a
    substring of its parent's text. Returns ``(n_parents, n_children)``.

    Raises ``RuntimeError`` on the first violation.
    """
    n_parents = 0
    n_children = 0
    cur_parent = None  # Chunk
    for _src, ch in chunks_with_src:
        if ch.is_parent:
            cur_parent = ch
            n_parents += 1
            continue
        n_children += 1
        if cur_parent is None or ch.parent_id != cur_parent.chunk_id:
            raise RuntimeError(
                f"child {ch.chunk_id} parent_id={ch.parent_id} but cur_parent="
                f"{cur_parent.chunk_id if cur_parent else None}"
            )
        if ch.text not in cur_parent.text:
            raise RuntimeError(
                f"child {ch.chunk_id} text not substring of parent "
                f"{cur_parent.chunk_id}: first 80 chars of child = "
                f"{ch.text[:80]!r}"
            )
    return n_parents, n_children


def _build_rows(
    chunks_with_src: list[tuple[dict, object]],
    *,
    chunk_format_version: int,
) -> tuple[list[dict], list[str]]:
    """Build (lance rows without 'vector', embed_texts) — one row per chunk.

    Both parents and children get embedded so the rerank/expand code in
    ``lab.rag.index`` can look up either via ``chunk_id``. The embed text
    for each chunk is just the chunk's body (no enrichment yet on the v2
    rebuild — keeping the build simple and reproducible).
    """
    rows: list[dict] = []
    embed_inputs: list[str] = []
    for src_dict, ch in chunks_with_src:
        rows.append(
            {
                "chunk_id": ch.chunk_id,
                "source_id": src_dict.get("id", ""),
                "source_url": src_dict.get("url", ""),
                "source_sha256": src_dict.get("sha256") or "",
                "retrieved_at": src_dict.get("retrieved_at") or "",
                "doc_path": ch.doc_path,
                "section_path": list(ch.section_path),
                "byte_start": int(ch.byte_start),
                "byte_end": int(ch.byte_end),
                "text": ch.text,
                "title": "",
                "summary": "",
                "keywords": [],
                "prerequisites": [],
                # vector filled in later
                "sparse_json": "{}",
                "tokens": int(ch.tokens),
                "chunk_format_version": chunk_format_version,
                "authority": src_dict.get("authority") or "",
                # ---- Phase 9 v2 fields. -------------------------------
                "parent_chunk_id": ch.parent_id,  # None for parents, set for children
                "child_index": ch.child_index,  # None for parents
                "is_parent": bool(ch.is_parent),
            }
        )
        embed_inputs.append(ch.text)
    return rows, embed_inputs


def _v2_manifest(src_manifest: Manifest, *, parent_target: int, child_target: int) -> Manifest:
    """Clone the v1 manifest into a v2-shaped one.

    - bumps ``chunk_format_version`` to 2
    - sets ``models.chunker.mode = parent_child`` + token targets
    - assigns a fresh ``kb_version`` token
    - resets ``status`` to ``intake`` (we'll advance it during build)
    - leaves sources / authority / models.embedding untouched
    """
    data = src_manifest.model_dump(mode="json")
    data["chunk_format_version"] = 2
    data["models"]["chunker"]["mode"] = "parent_child"
    data["models"]["chunker"]["parent_target_tokens"] = parent_target
    data["models"]["chunker"]["child_target_tokens"] = child_target
    # Keep the legacy target_tokens for downstream consumers — unused in PC mode.
    data["kb_version"] = "phase9-pc-" + secrets.token_hex(6)
    data["status"] = "intake"
    data["last_refreshed_at"] = now_iso()
    # Reset stats — we recompute below.
    data["stats"] = {
        "source_count": len(src_manifest.sources),
        "raw_bytes": data["stats"].get("raw_bytes", 0),
        "normalized_bytes": data["stats"].get("normalized_bytes", 0),
        "chunk_count": 0,
        "embedded_token_count": 0,
        "index_bytes": 0,
    }
    # Reset eval — to be re-run in a separate step if needed.
    data["eval"] = {
        "synthetic_query_count": 0,
        "retrieval_at_1": 0.0,
        "retrieval_at_5": 0.0,
        "failure_modes": [],
    }
    return Manifest.model_validate(data)


def _advance_status(dest_kb: Path, manifest: Manifest, status: str) -> None:
    """Write the manifest with a new status atomically."""
    manifest.status = status  # type: ignore[assignment]
    write_manifest(dest_kb / "manifest.yaml", manifest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="~/db/kb/bash", help="Source (v1) KB dir")
    ap.add_argument(
        "--dst",
        default="~/db/kb/bash-v2",
        help="Destination (v2) KB dir — must NOT overwrite an existing index",
    )
    ap.add_argument(
        "--parent-target",
        type=int,
        default=DEFAULT_PARENT_TARGET_TOKENS,
        help=f"Parent target tokens (default {DEFAULT_PARENT_TARGET_TOKENS})",
    )
    ap.add_argument(
        "--child-target",
        type=int,
        default=DEFAULT_CHILD_TARGET_TOKENS,
        help=f"Child target tokens (default {DEFAULT_CHILD_TARGET_TOKENS})",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Chunk only — print counts and validate invariant; write nothing",
    )
    ap.add_argument(
        "--embed-batch",
        type=int,
        default=8,
        help="Batch size for Ollama embedding calls (default 8)",
    )
    args = ap.parse_args()

    src_kb = Path(args.src).expanduser().resolve()
    dst_kb = Path(args.dst).expanduser().resolve()

    if not (src_kb / "manifest.yaml").exists():
        _eprint(f"ERROR: no manifest at {src_kb / 'manifest.yaml'}")
        return 2

    if not args.dry_run and dst_kb.exists() and any(dst_kb.iterdir()):
        _eprint(f"ERROR: destination {dst_kb} exists and is non-empty (refuse to clobber)")
        return 2

    t0 = time.time()
    src_manifest = load_manifest(src_kb / "manifest.yaml")
    _eprint(f"[1/6] loaded v1 manifest: {len(src_manifest.sources)} sources, status={src_manifest.status}")

    sources = _load_sources(src_kb, src_manifest)
    _eprint(f"[2/6] loaded {len(sources)} normalized source files")

    _eprint(
        f"[3/6] chunking PARENT_CHILD (parent={args.parent_target}, child={args.child_target})..."
    )
    t_chunk0 = time.time()
    chunks_with_src = _chunk_all(
        sources,
        parent_target=args.parent_target,
        child_target=args.child_target,
    )
    t_chunk = time.time() - t_chunk0
    n_parents, n_children = _validate_parent_child_invariant(chunks_with_src)
    n_total = len(chunks_with_src)
    _eprint(
        f"        chunked {n_total} records ({n_parents} parents, {n_children} children) "
        f"in {t_chunk:.1f}s; invariant OK"
    )

    if args.dry_run:
        _eprint(
            f"[dry-run] summary: parents={n_parents}, children={n_children}, "
            f"total={n_total}, t_chunk={t_chunk:.1f}s"
        )
        # Print first parent + its children for visual inspection.
        first_parent = None
        for _src, ch in chunks_with_src:
            if ch.is_parent:
                first_parent = ch
                break
        if first_parent is not None:
            _eprint(f"first parent ({first_parent.tokens} toks, section={first_parent.section_path}):")
            _eprint(f"  {first_parent.text[:200]!r} ...")
        return 0

    # --- Build path: prepare dest, write a v2 manifest in 'chunking_done' state ---
    dst_kb.mkdir(parents=True, exist_ok=True)
    (dst_kb / "index").mkdir(exist_ok=True)
    v2 = _v2_manifest(src_manifest, parent_target=args.parent_target, child_target=args.child_target)
    v2.stats.chunk_count = n_total
    _advance_status(dst_kb, v2, "chunking_done")
    _eprint(f"[3/6] wrote v2 manifest skeleton at {dst_kb / 'manifest.yaml'}")

    rows, embed_inputs = _build_rows(chunks_with_src, chunk_format_version=2)

    # --- Embed ---
    _advance_status(dst_kb, v2, "embedding_pending")
    _eprint(f"[4/6] embedding {len(embed_inputs)} chunks via {DEFAULT_EMBED_MODEL} (batch={args.embed_batch})...")
    t_emb0 = time.time()

    progress_last = [0]

    def _ep(done: int, total: int) -> None:
        # Print every ~256 chunks.
        if done - progress_last[0] >= 256 or done == total:
            progress_last[0] = done
            elapsed = time.time() - t_emb0
            rate = done / elapsed if elapsed > 0 else 0.0
            _eprint(f"        embed {done}/{total} ({rate:.1f}/s, {elapsed:.0f}s)")

    er = embed_texts(
        embed_inputs,
        model=DEFAULT_EMBED_MODEL,
        batch_size=args.embed_batch,
        progress=_ep,
        use_cache=False,
    )
    t_emb = time.time() - t_emb0
    _eprint(f"        done in {t_emb:.0f}s; model={er.model}, dims={er.dimensions}")

    if er.model != DEFAULT_EMBED_MODEL:
        _eprint(f"        NOTE: embedder fell back to {er.model}")
        v2.models.embedding.name = er.model
        v2.models.embedding.dimensions = er.dimensions
        v2.models.embedding.quantization = "Q8_0" if "q8" in er.model else ""

    if len(er.vectors) != len(rows):
        _eprint(
            f"ERROR: embed count mismatch — {len(er.vectors)} vectors vs {len(rows)} rows"
        )
        v2.status = "failed"
        write_manifest(dst_kb / "manifest.yaml", v2)
        return 3

    # --- BM25 sparse vectors over the same texts ---
    bm, _ = build_bm25(embed_inputs)
    for row, vec, text in zip(rows, er.vectors, embed_inputs, strict=True):
        sp = sparse_for_text(bm, tokenize_for_bm25(text))
        row["vector"] = vec
        row["sparse_json"] = json.dumps(sp)

    _advance_status(dst_kb, v2, "embedding_done")

    # --- Write index ---
    _advance_status(dst_kb, v2, "indexing_pending")
    _eprint(f"[5/6] writing LanceDB table at {dst_kb / 'index'}...")
    t_idx0 = time.time()
    replace_table(dst_kb, rows, er.dimensions)
    t_idx = time.time() - t_idx0
    _eprint(f"        wrote {len(rows)} rows in {t_idx:.1f}s")

    embedded_token_count = sum(int(r["tokens"]) for r in rows)
    v2.stats.embedded_token_count = embedded_token_count
    v2.stats.index_bytes = index_bytes(dst_kb)
    v2.stats.chunk_count = len(rows)
    _advance_status(dst_kb, v2, "indexing_done")

    # Final: mark indexed (the kb_builder pipeline goes -> validation -> sealed,
    # but we don't run enrichment/eval here; an explicit `lab kb eval` can run
    # afterward to populate retrieval@k. Sealing happens only after that.)
    v2.status = "indexing_done"  # type: ignore[assignment]
    write_manifest(dst_kb / "manifest.yaml", v2)

    # Mirror the source files into the new KB dir so the v2 KB is self-contained.
    sources_dst = dst_kb / "sources" / "normalized"
    if not sources_dst.exists():
        sources_dst.mkdir(parents=True, exist_ok=True)
        for _src, _text, norm_path in sources:
            (sources_dst / norm_path.name).write_text(
                norm_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
        _eprint(f"        mirrored {len(sources)} source files into {sources_dst}")

    t_total = time.time() - t0
    _eprint(
        f"[6/6] DONE — parents={n_parents}, children={n_children}, "
        f"index_bytes={v2.stats.index_bytes}, total_wall={t_total:.0f}s"
    )
    print(
        json.dumps(
            {
                "parents": n_parents,
                "children": n_children,
                "rows": len(rows),
                "index_bytes": v2.stats.index_bytes,
                "embedded_token_count": embedded_token_count,
                "kb_version": v2.kb_version,
                "wall_seconds": round(t_total, 1),
                "chunk_seconds": round(t_chunk, 1),
                "embed_seconds": round(t_emb, 1),
                "index_seconds": round(t_idx, 1),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
