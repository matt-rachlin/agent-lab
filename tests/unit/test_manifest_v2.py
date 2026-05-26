"""Phase 9 — manifest v2 round-trip + v1 back-compat.

A freshly-built manifest gets ``chunk_format_version=2`` and the new
``ChunkerSpec`` fields (``mode``, ``parent_target_tokens``,
``child_target_tokens``). v1 manifests that pre-date Phase 9 must still
deserialise cleanly, with the new fields filled in via defaults.
"""

from __future__ import annotations

from pathlib import Path

from lab.rag import CHUNK_FORMAT_VERSION
from lab.rag.manifest import Manifest, dump_manifest, load_manifest, write_manifest


def test_manifest_v2_defaults() -> None:
    m = Manifest(name="t", slug="t")
    assert m.chunk_format_version == CHUNK_FORMAT_VERSION == 2
    assert m.models.chunker.mode == "flat"
    assert m.models.chunker.parent_target_tokens == 768
    assert m.models.chunker.child_target_tokens == 192


def test_manifest_v2_round_trip(tmp_path: Path) -> None:
    m = Manifest(name="bash", slug="bash")
    m.models.chunker.mode = "parent_child"
    m.models.chunker.parent_target_tokens = 800
    m.models.chunker.child_target_tokens = 200
    path = tmp_path / "manifest.yaml"
    write_manifest(path, m)
    loaded = load_manifest(path)
    assert loaded.chunk_format_version == 2
    assert loaded.models.chunker.mode == "parent_child"
    assert loaded.models.chunker.parent_target_tokens == 800
    assert loaded.models.chunker.child_target_tokens == 200


def test_v1_manifest_still_loads(tmp_path: Path) -> None:
    """A v1 manifest YAML (no mode / no parent fields, chunk_format_version=1)
    must deserialise without errors; missing fields fall back to defaults."""
    v1_yaml = """
kb_format_version: 1
chunk_format_version: 1
name: legacy
slug: legacy
status: sealed
models:
  embedding:
    provider: ollama
    name: qwen3-embedding:8b-q8_0
    quantization: Q8_0
    dimensions: 4096
  enrichment:
    provider: claude
    name: qwen3:8b
  chunker:
    name: structural-markdown
    version: 1
    target_tokens: 512
    overlap_tokens: 64
"""
    path = tmp_path / "manifest.yaml"
    path.write_text(v1_yaml, encoding="utf-8")
    loaded = load_manifest(path)
    assert loaded.chunk_format_version == 1  # explicit field wins over default
    assert loaded.models.chunker.mode == "flat"  # default for missing field
    assert loaded.models.chunker.parent_target_tokens == 768
    assert loaded.models.chunker.child_target_tokens == 192


def test_dump_manifest_includes_chunker_mode() -> None:
    """The serialised YAML must carry the new fields so downstream tooling
    sees them without needing to re-derive defaults."""
    m = Manifest(name="t", slug="t")
    text = dump_manifest(m)
    assert "mode: flat" in text
    assert "parent_target_tokens: 768" in text
    assert "child_target_tokens: 192" in text
