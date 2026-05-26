"""lab.rag — vendored retrieval-augmented-generation modules.

Vendored from kb-builder (Phase 6h-a). The standalone kb-builder repo is no
longer the source of truth; lab.rag is. Constants live here; runtime paths
(LAB_KB_ROOT) read from lab.settings.
"""

from __future__ import annotations

from lab.settings import get_settings

KB_FORMAT_VERSION = 1
CHUNK_FORMAT_VERSION = 1
DEFAULT_EMBED_MODEL = "qwen3-embedding:8b-q8_0"
DEFAULT_EMBED_DIMS = 4096
FALLBACK_EMBED_MODEL = "qwen3-embedding:4b"
FALLBACK_EMBED_DIMS = 2560
DEFAULT_ENRICH_MODEL = "qwen3:8b"


def kb_root() -> str:
    """KB root directory (`~/db/kb` by default). Reads from lab settings."""
    return str(get_settings().kb_root)


# Back-compat alias for vendored modules that imported `DEFAULT_KB_ROOT`.
DEFAULT_KB_ROOT = "~/db/kb"
