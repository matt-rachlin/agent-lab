"""Register pulled local models + Ollama Cloud models into lab.models table."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
import psycopg

from lab.core.settings import get_settings

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


# Curated non-ollama local catalog (SGLang container-served, llama-swap-routed).
# These models are not discoverable via the Ollama /api/tags path, so they are
# declared here and merged into the upsert alongside CLOUD_MODELS. Provenance
# (source_sha256 + quant recipe) is read from the on-disk MANIFEST.json written
# by the AWQ pipeline — see _sglang_models() below.
SGLANG_MODELS: list[dict[str, Any]] = [
    {
        "publisher": "qwen",
        "name": "qwen3",
        "variant": "4b",
        "quant": "awq-w4a16",
        "backend": "sglang-local",
        "litellm_id": "qwen3-4b-awq",
        # source_sha256 + notes filled in from the manifest by _sglang_models().
        "manifest_path": "/data/lab/models/awq/qwen3-4b-awq/MANIFEST.json",
        "ollama_tag": None,
        "vram_gb": 7,
        "context_max": 40960,
        "output_max": 4096,
        "license": "apache-2.0",
        "capabilities": ["tool_call"],
        "notes": "In-house AWQ-W4A16 of Qwen3-4B; SGLang throughput tier (Phase 1).",
    },
]


def _sglang_models() -> list[dict[str, Any]]:
    """Materialize SGLANG_MODELS rows, reading provenance from each MANIFEST.json.

    The on-disk manifest's ``output_sha256`` becomes the row ``source_sha256``;
    the quant recipe/calibration fields are appended to ``notes``. A missing or
    unreadable manifest is non-fatal: the row is still emitted with
    ``source_sha256=None`` and a warning so registration stays idempotent.
    """
    rows: list[dict[str, Any]] = []
    for spec in SGLANG_MODELS:
        row = {k: v for k, v in spec.items() if k != "manifest_path"}
        row.setdefault("source_sha256", None)
        manifest_path = spec.get("manifest_path")
        if manifest_path:
            try:
                manifest = json.loads(Path(manifest_path).read_text())
                row["source_sha256"] = manifest.get("output_sha256")
                method = manifest.get("quant_method")
                scheme = manifest.get("scheme")
                calib = manifest.get("calibration_dataset")
                provenance = "; ".join(
                    p
                    for p in (
                        f"recipe={method}" if method else "",
                        f"scheme={scheme}" if scheme else "",
                        f"calib={calib}" if calib else "",
                    )
                    if p
                )
                if provenance:
                    row["notes"] = f"{row.get('notes', '')} [{provenance}]".strip()
            except (OSError, ValueError) as exc:
                print(
                    f"[register] could not read sglang manifest {manifest_path}: {exc}",
                    file=sys.stderr,
                )
        rows.append(row)
    return rows


def _ollama_local_models() -> list[dict[str, Any]]:
    """Query the local Ollama daemon for currently-pulled models."""
    url = get_settings().ollama_local_url.rstrip("/") + "/api/tags"
    try:
        r = httpx.get(url, timeout=10)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"[register] could not reach local ollama: {exc}", file=sys.stderr)
        return []
    models: list[dict[str, Any]] = r.json().get("models", [])
    return models


_QUANT_PREFIXES = ("q2_", "q3_", "q4_", "q5_", "q6_", "q8_", "fp16", "fp8", "iq")
# Ollama "modality" or "edition" markers that should not be treated as quant.
_NON_QUANT_TOKENS = {"instruct", "it", "chat", "base", "thinking", "cloud", "vl"}


def _split_variant_quant(variant_quant: str) -> tuple[str, str, bool]:
    """Split a tag suffix into (variant, quant, is_cloud).

    Heuristics:
      - tokens are dash-separated
      - first token is always the size/variant (e.g. "8b", "14b", "120b", "latest")
      - a trailing "cloud" token marks an Ollama Cloud model and is stripped
      - quant is everything matching `_QUANT_PREFIXES` joined back with "-"
      - non-quant tokens (instruct/it/chat/...) are dropped from the quant
    """
    if not variant_quant:
        return "", "", False
    tokens = variant_quant.split("-")
    is_cloud = False
    if tokens and tokens[-1].lower() == "cloud":
        is_cloud = True
        tokens = tokens[:-1]
    if not tokens:
        return "", "", is_cloud
    variant = tokens[0]
    quant_tokens: list[str] = []
    for tok in tokens[1:]:
        if not tok:
            continue
        if tok.lower() in _NON_QUANT_TOKENS:
            continue
        if tok.lower().startswith(_QUANT_PREFIXES):
            quant_tokens.append(tok)
        # else: drop unknown trailing token rather than smuggling it into quant
    quant = "-".join(quant_tokens)
    return variant, quant, is_cloud


def _parse_local(entry: dict[str, Any]) -> dict[str, Any] | None:
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

    variant, quant, is_cloud = _split_variant_quant(variant_quant)
    quant_norm = quant.upper() if quant else ("cloud" if is_cloud else "fp16")
    backend = "ollama-cloud" if is_cloud else "ollama-local"

    return {
        "publisher": publisher,
        "name": model_name,
        "variant": variant or None,
        "quant": quant_norm or None,
        "backend": backend,
        "litellm_id": _local_litellm_id(
            model_name, variant, quant, is_cloud=is_cloud, raw_suffix=variant_quant
        ),
        "source_sha256": digest.replace("sha256:", ""),
        "ollama_tag": tag,
        "vram_gb": vram_gb,
        "context_max": None,  # local models: unknown from /api/tags; set per-model later
        "output_max": None,
        "license": None,
        "capabilities": [],
        "notes": f"family={details.get('family')} fmt={details.get('format')}",
    }


_FRIENDLY = {
    "qwen3": {
        "14b-q4_k_m": "qwen3-14b-q4",
        "8b": "qwen3-8b",  # bare `qwen3:8b` (no explicit quant suffix in tag)
        "8b-q5_k_m": "qwen3-8b-q5",
    },
    "llama3.1": {"8b-instruct-q4_k_m": "llama3.1-8b-q4"},
    "phi4": {"latest": "phi4", "14b-q4_k_m": "phi-4-q4"},
    "gemma3": {"12b-it-q4_k_m": "gemma3-12b-q4"},
    "gemma4": {"12b": "gemma4-12b"},
}


def _local_litellm_id(
    name: str,
    variant: str,
    quant: str,
    *,
    is_cloud: bool = False,
    raw_suffix: str = "",
) -> str:
    """Map an Ollama tag to its canonical LiteLLM model_name.

    Lookup precedence: hand-curated `_FRIENDLY` map (keyed on either the cleaned
    or the raw `variant-quant` suffix) → deterministic fallback. The fallback
    is `<name>-<variant>[-<quant>][-cloud]`, lowercased and with no stray
    trailing dashes.
    """
    name_key = name.lower()
    variant_l = variant.lower()
    quant_l = quant.lower()
    cleaned = f"{variant_l}-{quant_l}" if quant_l else variant_l
    friendly = _FRIENDLY.get(name_key, {})
    if raw_suffix and raw_suffix.lower() in friendly:
        return friendly[raw_suffix.lower()]
    if cleaned in friendly:
        return friendly[cleaned]
    parts = [name_key]
    if variant_l:
        parts.append(variant_l)
    if quant_l:
        parts.append(quant_l)
    if is_cloud:
        parts.append("cloud")
    return "-".join(p for p in parts if p)


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


def sync_models(*, include_cloud: bool = True) -> dict[str, int]:
    """UPSERT the `models` table from the current `ollama list` output.

    Does NOT pull any model bytes — purely a metadata refresh. Returns a
    summary dict with `local`, `cloud`, and `total` counts.
    """
    rows: list[dict[str, Any]] = []
    for raw in _ollama_local_models():
        parsed = _parse_local(raw)
        if parsed:
            for k in ("source_sha256",):
                parsed.setdefault(k, None)
            rows.append(parsed)
    if include_cloud:
        for cloud in CLOUD_MODELS:
            row = {"source_sha256": None, **cloud}
            rows.append(row)
    # Curated non-ollama local models (SGLang). Always included — they are not
    # gated on the Ollama daemon being reachable. Idempotent via ON CONFLICT.
    rows.extend(_sglang_models())

    with psycopg.connect(get_settings().pg_dsn) as conn:
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(UPSERT_SQL, row)
        conn.commit()

    # Phase 15.2: additive MLflow mirror. Best-effort, never blocks.
    _mirror_models_to_mlflow(rows)

    return {
        "total": len(rows),
        "local": len([r for r in rows if r["backend"] == "ollama-local"]),
        "cloud": len([r for r in rows if r["backend"] == "ollama-cloud"]),
        "sglang": len([r for r in rows if r["backend"] == "sglang-local"]),
    }


def _mirror_models_to_mlflow(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    try:
        from lab.observability.mlflow_mirror import MlflowMirror

        mirror = MlflowMirror()
        if not mirror.enabled:
            return
        for row in rows:
            litellm_id = row.get("litellm_id")
            if not litellm_id:
                continue
            mlflow_uri = mirror.log_model_card(
                litellm_id,
                publisher=row.get("publisher") or "",
                variant=row.get("variant"),
                capabilities=row.get("capabilities") or [],
                known_issues=None,
            )
            if mlflow_uri:
                with (
                    psycopg.connect(get_settings().pg_dsn) as conn,
                    conn.cursor() as cur,
                ):
                    cur.execute(
                        "UPDATE models SET mlflow_model_uri = %s WHERE litellm_id = %s",
                        (mlflow_uri, litellm_id),
                    )
    except Exception:  # noqa: S110 — belt-and-suspenders; mirror already logs
        pass


def main() -> int:
    summary = sync_models()
    print(
        f"registered {summary['total']} models "
        f"({summary['local']} local, {summary['cloud']} cloud, {summary['sglang']} sglang)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
