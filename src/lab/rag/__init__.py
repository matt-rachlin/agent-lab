"""lab.rag — vendored retrieval-augmented-generation modules.

Vendored from kb-builder (Phase 6h-a). The standalone kb-builder repo is no
longer the source of truth; lab.rag is. Constants live here; runtime paths
(LAB_KB_ROOT) read from lab.settings.

Phase 9 (2026-05-26): parent-child chunking. ``lab.rag.chunker.ChunkMode``
selects between flat (v1, default) and parent_child (v2) chunking. Parent-
child mode emits paired ``(parent, child)`` records: children carry the
retrieval signal (small, precise embeddings); parents carry the read context
(large, semantically complete passages). At query time, ``kb_query`` /
``hybrid_query`` rank by child relevance but return parent text — controlled
by ``expand_to_parent`` / ``dedupe_by_parent``. Both default ON for v2 KBs
and are no-ops on legacy v1 KBs.
"""

from __future__ import annotations

from lab.settings import get_settings

KB_FORMAT_VERSION = 1
#: Bumped to 2 in Phase 9 to mark KBs that may carry parent-child chunks.
#: Existing v1 KBs are read-compatible — the new schema columns default to
#: null/false so a v1 row deserialises cleanly under the v2 reader.
CHUNK_FORMAT_VERSION = 2
DEFAULT_EMBED_MODEL = "qwen3-embedding:8b-q8_0"
DEFAULT_EMBED_DIMS = 4096
FALLBACK_EMBED_MODEL = "qwen3-embedding:4b"
FALLBACK_EMBED_DIMS = 2560
DEFAULT_ENRICH_MODEL = "qwen3:8b"

#: Primary cross-encoder reranker (Phase 7). Apache 2.0; ~1.2 GB VRAM.
DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
#: Mature fallback if the primary fails to load (still Apache 2.0).
FALLBACK_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
#: Env var used to override / disable the reranker (``=none`` for pass-through).
RERANKER_ENV_VAR = "LAB_RAG_RERANKER"
#: RRF constant — Cormack et al. 2009 recommend k=60.
RRF_K = 60


def kb_root() -> str:
    """KB root directory (`~/db/kb` by default). Reads from lab settings."""
    return str(get_settings().kb_root)


# Back-compat alias for vendored modules that imported `DEFAULT_KB_ROOT`.
DEFAULT_KB_ROOT = "~/db/kb"
