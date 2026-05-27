"""lab.rag — vendored retrieval-augmented-generation modules.

Vendored from kb-builder (Phase 6h-a). The standalone kb-builder repo is no
longer the source of truth; lab.rag is. Constants live here; runtime paths
(LAB_KB_ROOT) read from lab.core.settings.

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

from lab.core.settings import get_settings

KB_FORMAT_VERSION = 1
#: Bumped to 2 in Phase 9 to mark KBs that may carry parent-child chunks.
#: Existing v1 KBs are read-compatible — the new schema columns default to
#: null/false so a v1 row deserialises cleanly under the v2 reader.
#: Bumped to 3 in Phase 11 (HyPE) to mark KBs that may carry hypothetical-
#: question vectors alongside content vectors. v1/v2 KBs read cleanly under
#: the v3 reader (hype_questions / hype_vectors default to null).
CHUNK_FORMAT_VERSION = 3
DEFAULT_EMBED_MODEL = "qwen3-embedding:8b-q8_0"
DEFAULT_EMBED_DIMS = 4096
FALLBACK_EMBED_MODEL = "qwen3-embedding:4b"
FALLBACK_EMBED_DIMS = 2560
DEFAULT_ENRICH_MODEL = "qwen3:8b"

#: Primary cross-encoder reranker (Phase 7). Apache 2.0; ~1.2 GB VRAM.
#: Used when the caller opts in (env var set to this, or ``rerank=True``
#: passed explicitly). Phase 7's reranker-by-default behaviour was
#: reverted after EXP-004c (see F-007 amendment) — the +5pp recall@5
#: lift didn't earn the +700ms-per-call latency cost for the average
#: caller. The constant is kept; what changed is the env-var fallback
#: (``DEFAULT_RERANKER_MODE_WHEN_UNSET``) and the ``rerank=`` defaults
#: at the call sites (``kb_query``, ``hybrid_query``).
DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
#: Mature fallback if the primary fails to load (still Apache 2.0).
FALLBACK_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
#: Env var used to override / disable the reranker (``=none`` for pass-through).
RERANKER_ENV_VAR = "LAB_RAG_RERANKER"
#: When ``LAB_RAG_RERANKER`` is unset or empty, default to this mode.
#: Sentinel ``"none"`` post-EXP-004c (previously fell through to
#: :data:`DEFAULT_RERANKER_MODEL`, which caused every :class:`LabReranker`
#: instantiation without an env var to load the cross-encoder). See F-007
#: amendment.
DEFAULT_RERANKER_MODE_WHEN_UNSET = "none"
#: RRF constant — Cormack et al. 2009 recommend k=60.
RRF_K = 60


def kb_root() -> str:
    """KB root directory (`~/db/kb` by default). Reads from lab settings."""
    return str(get_settings().kb_root)


# Back-compat alias for vendored modules that imported `DEFAULT_KB_ROOT`.
DEFAULT_KB_ROOT = "~/db/kb"
