"""Migrate a v1 KB's LanceDB schema to v2 (Phase 9 parent-child).

Adds three columns to ``<kb>/index/chunks.lance/`` and bumps
``chunk_format_version`` in the manifest:

  - ``parent_chunk_id`` (string, NULL)
  - ``child_index`` (int32, NULL)
  - ``is_parent`` (bool, default False)

Idempotent: re-running on an already-v2 KB is a no-op. The actual parent-
child population happens on the next KB rebuild — this script only widens
the schema so v2 code can read v1 data.

Usage:

    python scripts/migrate_kb_schema_v2.py ~/db/kb/bash
    python scripts/migrate_kb_schema_v2.py ~/db/kb/bash --dry-run

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import lab.rag lazily so --dry-run on a path with permissions issues still
# emits a sane preview.


def _detect_schema_version(kb_dir: Path) -> tuple[int, set[str]]:
    """Return ``(schema_version, present_column_names)``.

    ``schema_version`` is 2 when all three v2 columns exist on disk, else 1.
    """
    try:
        import lancedb
    except Exception as exc:  # pragma: no cover - import guard
        print(f"ERROR: cannot import lancedb: {exc}", file=sys.stderr)
        sys.exit(2)

    db = lancedb.connect(str(kb_dir / "index"))
    if "chunks" not in db.list_tables().tables:
        return (0, set())
    tbl = db.open_table("chunks")
    cols = set(tbl.schema.names)
    v2_cols = {"parent_chunk_id", "child_index", "is_parent"}
    return (2 if v2_cols.issubset(cols) else 1, cols)


def _bump_manifest(kb_dir: Path, *, dry_run: bool) -> tuple[int, int]:
    """Bump chunk_format_version in the manifest. Returns (old, new)."""
    from lab.rag import CHUNK_FORMAT_VERSION
    from lab.rag.manifest import load_manifest, write_manifest

    manifest_path = kb_dir / "manifest.yaml"
    manifest = load_manifest(manifest_path)
    old = int(manifest.chunk_format_version)
    new = int(CHUNK_FORMAT_VERSION)
    if old == new:
        return (old, new)
    if not dry_run:
        manifest.chunk_format_version = new
        write_manifest(manifest_path, manifest)
    return (old, new)


def _add_columns(kb_dir: Path, *, dry_run: bool) -> tuple[int, set[str]]:
    """Add v2 columns to the chunks table. Returns ``(rows_touched, added_cols)``.

    Idempotent: if a column already exists, we skip it.
    """
    import lancedb
    import pyarrow as pa

    db = lancedb.connect(str(kb_dir / "index"))
    if "chunks" not in db.list_tables().tables:
        return (0, set())
    tbl = db.open_table("chunks")
    have = set(tbl.schema.names)
    want_cols: dict[str, pa.DataType] = {
        "parent_chunk_id": pa.string(),
        "child_index": pa.int32(),
        "is_parent": pa.bool_(),
    }
    to_add = {n: t for n, t in want_cols.items() if n not in have}
    if not to_add:
        return (tbl.count_rows(), set())

    row_count = tbl.count_rows()
    if dry_run:
        return (row_count, set(to_add))

    # LanceDB ``add_columns`` takes a dict of name -> SQL expr; we use NULL
    # for the nullable text/int and false for the bool default.
    add_specs = {}
    for name in to_add:
        if name == "is_parent":
            add_specs[name] = "CAST(false AS BOOLEAN)"
        elif name == "child_index":
            add_specs[name] = "CAST(NULL AS INT)"
        else:
            add_specs[name] = "CAST(NULL AS STRING)"
    try:
        tbl.add_columns(add_specs)
    except AttributeError:
        # Older LanceDB releases used ``alter_columns`` / merge semantics;
        # fall back to a re-create that preserves data + injects the columns
        # via pyarrow. This path is slow but correct.
        rows = tbl.to_arrow().to_pylist()
        for r in rows:
            for n in to_add:
                r[n] = False if n == "is_parent" else None
        from lab.rag.index import _schema

        # Vector dims live on the existing schema's vector field.
        vec_field = tbl.schema.field("vector")
        dims = vec_field.type.list_size
        new_schema = _schema(dims)
        db.drop_table("chunks")
        new_tbl = db.create_table("chunks", schema=new_schema)
        if rows:
            new_tbl.add(rows)
    return (row_count, set(to_add))


def migrate(kb_dir: Path, *, dry_run: bool) -> dict[str, object]:
    """Run the migration end-to-end. Returns a summary dict."""
    if not (kb_dir / "manifest.yaml").is_file():
        raise FileNotFoundError(f"no manifest at {kb_dir / 'manifest.yaml'}")
    if not (kb_dir / "index").is_dir():
        raise FileNotFoundError(f"no index directory at {kb_dir / 'index'}")

    before_version, before_cols = _detect_schema_version(kb_dir)
    rows_touched, added_cols = _add_columns(kb_dir, dry_run=dry_run)
    manifest_old, manifest_new = _bump_manifest(kb_dir, dry_run=dry_run)
    after_version, _after_cols = (
        _detect_schema_version(kb_dir)
        if not dry_run
        else (
            2 if added_cols else before_version,
            before_cols | added_cols,
        )
    )

    return {
        "kb_dir": str(kb_dir),
        "dry_run": dry_run,
        "before_schema_version": before_version,
        "after_schema_version": after_version,
        "columns_before": sorted(before_cols),
        "columns_added": sorted(added_cols),
        "manifest_chunk_format_version_before": manifest_old,
        "manifest_chunk_format_version_after": manifest_new,
        "rows_touched": rows_touched,
    }


def _print_summary(summary: dict[str, object]) -> None:
    print(f"KB:            {summary['kb_dir']}")
    print(f"Dry-run:       {summary['dry_run']}")
    print(
        f"Schema:        v{summary['before_schema_version']} -> v{summary['after_schema_version']}"
    )
    print(
        f"Manifest:      chunk_format_version "
        f"{summary['manifest_chunk_format_version_before']} -> "
        f"{summary['manifest_chunk_format_version_after']}"
    )
    cols_added = summary["columns_added"]
    if cols_added:
        print(f"Columns added: {', '.join(cols_added)}")  # type: ignore[arg-type]
    else:
        print("Columns added: (none — already v2)")
    print(f"Rows touched:  {summary['rows_touched']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kb_dir", type=Path, help="path to a KB directory (e.g. ~/db/kb/bash)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would change without writing anything",
    )
    args = parser.parse_args(argv)

    kb_dir = args.kb_dir.expanduser().resolve()
    try:
        summary = migrate(kb_dir, dry_run=args.dry_run)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
