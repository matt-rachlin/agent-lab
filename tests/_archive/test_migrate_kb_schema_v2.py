"""Phase 9 — schema migration script (lab.scripts.migrate_kb_schema_v2).

We build a tiny v1-style KB in tmp_path, run the migration, and confirm:
  - the three new columns appear with safe defaults
  - the manifest's chunk_format_version flips to 2
  - re-running is a no-op (idempotent)
  - --dry-run does not mutate anything
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load the script by file path — it lives under scripts/ which isn't on PYTHONPATH.
_SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "migrate_kb_schema_v2.py"


def _load_script():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("migrate_kb_schema_v2", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["migrate_kb_schema_v2"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_v1_kb(tmp_path: Path) -> Path:
    """Stage a minimal v1 KB (v1 schema; chunk_format_version=1 manifest)."""
    import lancedb
    import pyarrow as pa

    kb_dir = tmp_path / "tinykb"
    (kb_dir / "index").mkdir(parents=True)
    # v1 schema lacks the three Phase 9 columns.
    v1_schema = pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 4)),
        ]
    )
    db = lancedb.connect(str(kb_dir / "index"))
    tbl = db.create_table("chunks", schema=v1_schema)
    tbl.add(
        [
            {"chunk_id": "a", "text": "alpha", "vector": [1.0, 0.0, 0.0, 0.0]},
            {"chunk_id": "b", "text": "beta", "vector": [0.0, 1.0, 0.0, 0.0]},
        ]
    )
    # Minimal v1 manifest.
    (kb_dir / "manifest.yaml").write_text(
        "kb_format_version: 1\nchunk_format_version: 1\nname: tinykb\nslug: tinykb\n",
        encoding="utf-8",
    )
    return kb_dir


def test_dry_run_reports_changes_without_writing(tmp_path: Path) -> None:
    mod = _load_script()
    kb_dir = _make_v1_kb(tmp_path)
    summary = mod.migrate(kb_dir, dry_run=True)
    assert summary["before_schema_version"] == 1
    assert set(summary["columns_added"]) == {"parent_chunk_id", "child_index", "is_parent"}
    assert summary["rows_touched"] == 2
    # Manifest untouched after dry-run.
    import yaml

    manifest_text = (kb_dir / "manifest.yaml").read_text()
    parsed = yaml.safe_load(manifest_text)
    assert parsed["chunk_format_version"] == 1


def test_migrate_adds_columns_and_bumps_manifest(tmp_path: Path) -> None:
    from lab.rag import CHUNK_FORMAT_VERSION

    mod = _load_script()
    kb_dir = _make_v1_kb(tmp_path)
    summary = mod.migrate(kb_dir, dry_run=False)
    # The migration brings the schema to v2 (Phase 9 columns); the manifest
    # bump tracks the current ``CHUNK_FORMAT_VERSION`` constant (which is 3
    # post-Phase 11; the migration script still only adds v2 columns).
    assert summary["after_schema_version"] == 2
    import yaml

    parsed = yaml.safe_load((kb_dir / "manifest.yaml").read_text())
    assert parsed["chunk_format_version"] == CHUNK_FORMAT_VERSION
    # The Phase 9 columns are present on the table.
    import lancedb

    db = lancedb.connect(str(kb_dir / "index"))
    tbl = db.open_table("chunks")
    assert {"parent_chunk_id", "child_index", "is_parent"}.issubset(set(tbl.schema.names))


def test_migrate_idempotent(tmp_path: Path) -> None:
    from lab.rag import CHUNK_FORMAT_VERSION

    mod = _load_script()
    kb_dir = _make_v1_kb(tmp_path)
    mod.migrate(kb_dir, dry_run=False)
    # Second run is a no-op: nothing added, manifest already at current.
    summary = mod.migrate(kb_dir, dry_run=False)
    assert summary["columns_added"] == []
    assert summary["manifest_chunk_format_version_before"] == CHUNK_FORMAT_VERSION
    assert summary["manifest_chunk_format_version_after"] == CHUNK_FORMAT_VERSION


def test_migrate_missing_manifest_errors(tmp_path: Path) -> None:
    mod = _load_script()
    with pytest.raises(FileNotFoundError):
        mod.migrate(tmp_path / "nope", dry_run=True)
