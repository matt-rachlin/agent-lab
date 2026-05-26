"""Manifest schema and YAML IO. Source of truth for what's in a KB.

Vendored from kb_builder.manifest.
"""

from __future__ import annotations

from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from ruamel.yaml import YAML

from lab.rag import (
    CHUNK_FORMAT_VERSION,
    DEFAULT_EMBED_DIMS,
    DEFAULT_EMBED_MODEL,
    DEFAULT_ENRICH_MODEL,
    KB_FORMAT_VERSION,
)
from lab.rag._util import atomic_write_text, utcnow_iso

SourceType = Literal[
    "manpage",
    "git-repo",
    "html-single-page",
    "html-sitemap",
    "html-spa",
    "pdf",
    "rfc",
    "github-source-tree",
    "stack-exchange",
    "webfetch",
]

Authority = Literal["official", "manpage", "rfc", "book", "community"]


class Budget(BaseModel):
    max_pages: int = 500
    max_tokens_embedded: int = 5_000_000
    max_wall_minutes: int = 120


class AgentSpec(BaseModel):
    model: str = "claude-opus-4-7"
    agent_file: str = "~/workspace/.claude/agents/dev-research-kb.md"
    trace: str = "agent-trace/discovery.jsonl"


class EmbeddingModel(BaseModel):
    provider: str = "ollama"
    name: str = DEFAULT_EMBED_MODEL
    quantization: str = "Q8_0"
    dimensions: int = DEFAULT_EMBED_DIMS
    revision: str | None = None


class EnrichmentModel(BaseModel):
    provider: str = "claude"
    name: str = DEFAULT_ENRICH_MODEL


ChunkerMode = Literal["flat", "parent_child"]


class ChunkerSpec(BaseModel):
    name: str = "structural-markdown"
    version: int = 1
    target_tokens: int = 512
    overlap_tokens: int = 64
    #: Phase 9 (v2): chunking strategy. ``"flat"`` is the legacy v1 behaviour;
    #: ``"parent_child"`` emits (parent, child) pairs. Old manifests without
    #: this field default to ``"flat"`` so loading stays back-compatible.
    mode: ChunkerMode = "flat"
    #: Parent-target token count (PARENT_CHILD only). Ignored for FLAT.
    parent_target_tokens: int = 768
    #: Child-target token count (PARENT_CHILD only). Ignored for FLAT.
    child_target_tokens: int = 192


class Models(BaseModel):
    embedding: EmbeddingModel = Field(default_factory=EmbeddingModel)
    enrichment: EnrichmentModel = Field(default_factory=EnrichmentModel)
    chunker: ChunkerSpec = Field(default_factory=ChunkerSpec)


class BuildSpec(BaseModel):
    prompt: str = ""
    budget: Budget = Field(default_factory=Budget)
    agent: AgentSpec = Field(default_factory=AgentSpec)


class SourceEntry(BaseModel):
    id: str
    url: str
    type: SourceType
    fetcher: str
    retrieved_at: str | None = None
    sha256: str | None = None
    bytes: int | None = None
    normalized: str | None = None  # relative path under KB dir
    license: str | None = None
    authority: Authority = "community"
    inclusion_rationale: str = ""
    skipped_with_reason: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Stats(BaseModel):
    source_count: int = 0
    raw_bytes: int = 0
    normalized_bytes: int = 0
    chunk_count: int = 0
    embedded_token_count: int = 0
    index_bytes: int = 0


class EvalRecord(BaseModel):
    synthetic_query_count: int = 0
    retrieval_at_1: float = 0.0
    retrieval_at_5: float = 0.0
    failure_modes: list[str] = Field(default_factory=list)


BuildStatus = Literal[
    "intake",
    "discovery_pending",
    "discovery_done",
    "acquisition_pending",
    "acquisition_done",
    "chunking_pending",
    "chunking_done",
    "enrichment_pending",
    "enrichment_done",
    "embedding_pending",
    "embedding_done",
    "indexing_pending",
    "indexing_done",
    "validation_pending",
    "validation_done",
    "sealed",
    "failed",
]


class Manifest(BaseModel):
    kb_format_version: int = KB_FORMAT_VERSION
    chunk_format_version: int = CHUNK_FORMAT_VERSION
    name: str
    slug: str
    description: str = ""
    created_at: str = Field(default_factory=utcnow_iso)
    last_refreshed_at: str | None = None
    status: BuildStatus = "intake"
    #: Optional cache-invalidation token (Phase 8). When a KB rebuilds and
    #: bumps this value, the RAG cache namespace flips and old keys become
    #: unreachable. Old manifests without this field still load — see
    #: :func:`lab.rag.cache.kb_version_token` for the fallback (content
    #: hash of the manifest body).
    kb_version: str | None = None

    build: BuildSpec = Field(default_factory=BuildSpec)
    models: Models = Field(default_factory=Models)
    sources: list[SourceEntry] = Field(default_factory=list)
    stats: Stats = Field(default_factory=Stats)
    eval: EvalRecord = Field(default_factory=EvalRecord)


# ---------- YAML IO ----------

_yaml = YAML(typ="rt")
_yaml.default_flow_style = False
_yaml.width = 100
_yaml.preserve_quotes = True


def dump_manifest(manifest: Manifest) -> str:
    buf = StringIO()
    _yaml.dump(manifest.model_dump(mode="json"), buf)
    return buf.getvalue()


def write_manifest(path: Path, manifest: Manifest) -> None:
    atomic_write_text(path, dump_manifest(manifest))


def load_manifest(path: Path) -> Manifest:
    with path.open("r", encoding="utf-8") as fh:
        data = _yaml.load(fh)
    return Manifest.model_validate(data)


def now_iso() -> str:
    return utcnow_iso()


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))
