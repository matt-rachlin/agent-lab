"""Add HyPE (Phase 11) columns to an existing KB — generate hypothetical
questions per chunk, embed them, write back to LanceDB.

DO NOT RUN this while a KB rebuild is in flight. As of 2026-05-26 the
Phase 9 parent-child rebuild owns the embedding model (Ollama
qwen3-embedding:8b-q8_0 is kept resident) and the LanceDB lockfile on
``~/db/kb/bash-v2/``. Run this script only after the Phase 9 swap is
complete and the host's Ollama is idle.

What it does, per row in the chunks table:

  1. Skip if ``hype_questions`` is already non-null (idempotent).
  2. Call :func:`lab.rag.hype.generate_hype_questions` to produce N
     question strings via local Ollama (qwen3:8b by default).
  3. Embed each question with the KB's configured embedding model
     (``manifest.models.embedding.name``).
  4. Write ``hype_questions`` + ``hype_vectors`` back to the row.

Usage:

    python scripts/add_hype_to_kb.py ~/db/kb/bash
    python scripts/add_hype_to_kb.py ~/db/kb/bash --n-questions 3 --dry-run
    python scripts/add_hype_to_kb.py ~/db/kb/bash --hype-model qwen3:8b

The script also bumps ``manifest.chunk_format_version`` to 3 and sets
``models.hype.enabled = True`` so the next ``hybrid_query`` call
auto-routes through the HyPE path. Use ``--no-manifest-bump`` to skip
that step (testing only).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kb_dir", type=Path, help="Path to KB root (contains manifest.yaml)")
    p.add_argument(
        "--n-questions",
        type=int,
        default=3,
        help="Hypothetical questions per chunk (default 3)",
    )
    p.add_argument(
        "--hype-model",
        type=str,
        default="qwen3:8b",
        help="Ollama chat model for question generation (default qwen3:8b)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM temperature for question generation (default 0.3)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change; don't write anything",
    )
    p.add_argument(
        "--no-manifest-bump",
        action="store_true",
        help="Skip bumping chunk_format_version + models.hype in manifest.yaml",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N chunks (smoke test)",
    )
    return p.parse_args(argv)


def _bump_manifest(kb_dir: Path, *, n_questions: int, model: str, dry_run: bool) -> None:
    from lab.rag import CHUNK_FORMAT_VERSION
    from lab.rag.manifest import load_manifest, write_manifest

    manifest_path = kb_dir / "manifest.yaml"
    manifest = load_manifest(manifest_path)
    manifest.chunk_format_version = int(CHUNK_FORMAT_VERSION)
    manifest.models.hype.enabled = True
    manifest.models.hype.n_questions = int(n_questions)
    manifest.models.hype.model = model
    if dry_run:
        print(
            f"[dry-run] would bump manifest: chunk_format_version="
            f"{manifest.chunk_format_version}, hype.enabled=True, "
            f"hype.n_questions={n_questions}, hype.model={model}"
        )
        return
    write_manifest(manifest_path, manifest)
    print(
        f"manifest: chunk_format_version={manifest.chunk_format_version}, "
        f"hype.enabled=True, hype.n_questions={n_questions}, hype.model={model}"
    )


def _row_already_has_hype(row: dict[str, Any]) -> bool:
    q = row.get("hype_questions")
    return bool(q)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    kb_dir: Path = args.kb_dir.expanduser().resolve()

    manifest_path = kb_dir / "manifest.yaml"
    if not manifest_path.is_file():
        print(f"ERROR: no manifest at {manifest_path}", file=sys.stderr)
        return 2

    import lancedb

    from lab.rag.embedder import embed_texts
    from lab.rag.hype import generate_hype_questions
    from lab.rag.manifest import load_manifest

    manifest = load_manifest(manifest_path)
    embed_model = manifest.models.embedding.name

    db = lancedb.connect(str(kb_dir / "index"))
    if "chunks" not in db.list_tables().tables:
        print(f"ERROR: no chunks table under {kb_dir}/index", file=sys.stderr)
        return 2
    tbl = db.open_table("chunks")
    rows = tbl.to_arrow().to_pylist()
    if args.limit is not None:
        rows = rows[: args.limit]

    total = len(rows)
    todo = [r for r in rows if not _row_already_has_hype(r)]
    print(f"chunks total={total} todo={len(todo)} (skip already-hype)")

    if args.dry_run:
        print(f"[dry-run] would generate HyPE for {len(todo)} chunks")
        print(f"[dry-run] hype_model={args.hype_model} embed_model={embed_model}")
        if not args.no_manifest_bump:
            _bump_manifest(
                kb_dir,
                n_questions=args.n_questions,
                model=args.hype_model,
                dry_run=True,
            )
        return 0

    updates: list[dict[str, Any]] = []
    for i, row in enumerate(todo, start=1):
        chunk_text = row.get("text") or ""
        section_path = list(row.get("section_path") or [])
        questions = generate_hype_questions(
            chunk_text,
            section_path=section_path,
            n_questions=args.n_questions,
            model=args.hype_model,
            temperature=args.temperature,
        )
        if not questions:
            print(f"[{i}/{len(todo)}] {row.get('chunk_id')}: no questions generated; skipping")
            continue
        emb = embed_texts(questions, model=embed_model, batch_size=4, use_cache=False)
        updates.append(
            {
                "chunk_id": row["chunk_id"],
                "hype_questions": questions,
                "hype_vectors": emb.vectors,
            }
        )
        if i % 25 == 0:
            print(f"[{i}/{len(todo)}] generated+embedded")

    # LanceDB's table.update by predicate is feature-flagged across versions;
    # the most portable path is a per-row merge. Stay defensive: if the table
    # supports add_columns / merge_insert, prefer that.
    if updates:
        try:
            tbl.merge_insert(
                "chunk_id"
            ).when_matched_update_all().when_not_matched_insert_all().execute(updates)
        except Exception:
            # Fallback: rewrite the table with the new columns populated.
            # We pull all rows, merge updates by chunk_id, and replace the
            # table contents via :func:`replace_table`.
            from lab.rag.index import replace_table

            by_id: dict[str, dict[str, Any]] = {u["chunk_id"]: u for u in updates}
            merged: list[dict[str, Any]] = []
            for r in rows:
                cid = r.get("chunk_id")
                u = by_id.get(cid)
                if u is not None:
                    r = dict(r)
                    r["hype_questions"] = u["hype_questions"]
                    r["hype_vectors"] = u["hype_vectors"]
                merged.append(r)
            replace_table(kb_dir, merged, dims=int(manifest.models.embedding.dimensions))

    if not args.no_manifest_bump:
        _bump_manifest(
            kb_dir,
            n_questions=args.n_questions,
            model=args.hype_model,
            dry_run=False,
        )
    print(f"done: wrote {len(updates)} chunks with HyPE columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
