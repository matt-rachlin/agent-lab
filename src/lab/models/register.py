"""Register pulled local models + Ollama Cloud models into lab.models table."""

from __future__ import annotations

import hashlib
import json
import sys

import httpx
import psycopg

from lab.settings import get_settings

# Curated initial catalog. Edit + re-run to refresh.
CLOUD_MODELS = [
    {
        "publisher": "openai",
        "name": "gpt-oss",
        "variant": "20b",
        "quant": "cloud",
        "backend": "ollama-cloud",
        "litellm_id": "gpt-oss-20b-cloud",
        "ollama_tag": "gpt-oss:20b-cloud",
        "vram_gb": None,
        "context_max": 131072,
        "output_max": 16384,
        "license": "apache-2.0",
        "capabilities": ["tool_call", "reasoning", "json"],
        "notes": "Cheap cloud fan-out; planning + judges.",
    },
    {
        "publisher": "openai",
        "name": "gpt-oss",
        "variant": "120b",
        "quant": "cloud",
        "backend": "ollama-cloud",
        "litellm_id": "gpt-oss-120b-cloud",
        "ollama_tag": "gpt-oss:120b-cloud",
        "vram_gb": None,
        "context_max": 131072,
        "output_max": 16384,
        "license": "apache-2.0",
        "capabilities": ["tool_call", "reasoning", "json"],
        "notes": "General-purpose cloud orchestrator.",
    },
    {
        "publisher": "qwen",
        "name": "qwen3-coder",
        "variant": "480b",
        "quant": "cloud",
        "backend": "ollama-cloud",
        "litellm_id": "qwen3-coder-480b-cloud",
        "ollama_tag": "qwen3-coder:480b-cloud",
        "vram_gb": None,
        "context_max": 262144,
        "output_max": 16384,
        "license": "apache-2.0",
        "capabilities": ["tool_call", "code", "json"],
        "notes": "Best non-frontier coding agent; reserve for code work.",
    },
    {
        "publisher": "deepseek",
        "name": "deepseek-v3.1",
        "variant": "671b",
        "quant": "cloud",
        "backend": "ollama-cloud",
        "litellm_id": "deepseek-v31-671b-cloud",
        "ollama_tag": "deepseek-v3.1:671b-cloud",
        "vram_gb": None,
        "context_max": 163840,
        "output_max": 16384,
        "license": "deepseek-license",
        "capabilities": ["tool_call", "reasoning", "json"],
        "notes": "Oracle judge / second opinion. Cloud budget heavy — use sparingly.",
    },
    {
        "publisher": "moonshot",
        "name": "kimi-k2-thinking",
        "variant": "1t",
        "quant": "cloud",
        "backend": "ollama-cloud",
        "litellm_id": "kimi-k2-thinking-cloud",
        "ollama_tag": "kimi-k2-thinking",
        "vram_gb": None,
        "context_max": 262144,
        "output_max": 16384,
        "license": "kimi-license",
        "capabilities": ["tool_call", "reasoning"],
        "notes": "Deep reasoning oracle.",
    },
]


def _ollama_local_models() -> list[dict]:
    """Query the local Ollama daemon for currently-pulled models."""
    url = get_settings().ollama_local_url.rstrip("/") + "/api/tags"
    try:
        r = httpx.get(url, timeout=10)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"[register] could not reach local ollama: {exc}", file=sys.stderr)
        return []
    return r.json().get("models", [])


def _parse_local(entry: dict) -> dict | None:
    """Parse one entry from /api/tags into our models row format."""
    tag = entry.get("name", "")  # e.g. "qwen3:14b-q4_K_M"
    if not tag:
        return None
    name_part, _, variant_quant = tag.partition(":")
    digest = entry.get("digest", "")
    details = entry.get("details", {}) or {}
    size_bytes = entry.get("size", 0)
    vram_gb = round(size_bytes / 1e9, 1)

    publisher, _, model_name = name_part.partition("/")
    if not model_name:
        publisher, model_name = "ollama", publisher

    # Heuristically split "14b-q4_K_M" → variant=14b quant=Q4_K_M
    variant, _, quant = variant_quant.partition("-")
    quant_norm = quant.upper().replace("INSTRUCT-", "") if quant else "fp16"

    return {
        "publisher": publisher,
        "name": model_name,
        "variant": variant or None,
        "quant": quant_norm or None,
        "backend": "ollama-local",
        "litellm_id": _local_litellm_id(model_name, variant, quant),
        "source_sha256": digest.replace("sha256:", ""),
        "ollama_tag": tag,
        "vram_gb": vram_gb,
        "context_max": details.get("parameter_size", None) and None,
        "output_max": None,
        "license": None,
        "capabilities": [],
        "notes": f"family={details.get('family')} fmt={details.get('format')}",
    }


_FRIENDLY = {
    "qwen3": {"14b-q4_k_m": "qwen3-14b-q4", "8b-q5_k_m": "qwen3-8b-q5"},
    "llama3.1": {"8b-instruct-q4_k_m": "llama3.1-8b-q4"},
    "phi-4": {"14b-q4_k_m": "phi-4-q4"},
    "gemma3": {"12b-it-q4_k_m": "gemma3-12b-q4"},
}


def _local_litellm_id(name: str, variant: str, quant: str) -> str:
    """Map an Ollama tag to its canonical LiteLLM model_name."""
    name_key = name.lower()
    variant_quant = f"{variant}-{quant}".lower() if quant else variant.lower()
    return _FRIENDLY.get(name_key, {}).get(variant_quant, f"{name_key}-{variant}-{quant}".lower())


UPSERT_SQL = """
INSERT INTO models
    (publisher, name, variant, quant, backend, litellm_id,
     source_sha256, ollama_tag, vram_gb, context_max, output_max,
     license, capabilities, notes)
VALUES
    (%(publisher)s, %(name)s, %(variant)s, %(quant)s, %(backend)s, %(litellm_id)s,
     %(source_sha256)s, %(ollama_tag)s, %(vram_gb)s, %(context_max)s, %(output_max)s,
     %(license)s, %(capabilities)s, %(notes)s)
ON CONFLICT (litellm_id) DO UPDATE SET
    publisher = EXCLUDED.publisher,
    name = EXCLUDED.name,
    variant = EXCLUDED.variant,
    quant = EXCLUDED.quant,
    backend = EXCLUDED.backend,
    source_sha256 = EXCLUDED.source_sha256,
    ollama_tag = EXCLUDED.ollama_tag,
    vram_gb = EXCLUDED.vram_gb,
    context_max = EXCLUDED.context_max,
    output_max = EXCLUDED.output_max,
    license = EXCLUDED.license,
    capabilities = EXCLUDED.capabilities,
    notes = EXCLUDED.notes,
    pulled_at = NOW();
"""


def main() -> int:
    rows: list[dict] = []
    for raw in _ollama_local_models():
        parsed = _parse_local(raw)
        if parsed:
            for k in ("source_sha256",):
                parsed.setdefault(k, None)
            rows.append(parsed)
    for cloud in CLOUD_MODELS:
        row = {"source_sha256": None, **cloud}
        rows.append(row)

    with psycopg.connect(get_settings().pg_dsn) as conn:
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(UPSERT_SQL, row)
        conn.commit()

    print(f"registered {len(rows)} models ({len([r for r in rows if r['backend'] == 'ollama-local'])} local, "
          f"{len([r for r in rows if r['backend'] == 'ollama-cloud'])} cloud)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
