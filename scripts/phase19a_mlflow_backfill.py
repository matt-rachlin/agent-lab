#!/usr/bin/env python3
"""Phase 19a — backfill 6 new lab.models rows into MLflow's lab-models experiment.

Idempotent: MlflowMirror.log_model_card finds-or-creates a run by name, so
re-running this only updates tags and re-writes mlflow_model_uri to the
canonical run-id. Safe to invoke from a sweep or by hand:

    cd /data/lab/code && uv run python scripts/phase19a_mlflow_backfill.py

Returns:
    exit 0 on success
    exit 1 if MlflowMirror is disabled (MLflow tracking server unreachable)
    exit 2 if any of the 6 rows is missing from lab.models (run Phase 19a DB
        seed first)
"""

from __future__ import annotations

import sys

import psycopg

from lab.core.settings import get_settings
from lab.observability.mlflow_mirror import MlflowMirror

PHASE_19A_MODELS = [
    "qwen3-30b-a3b-moe",
    "gpt-oss-20b-local",
    "phi-4-reasoning-14b",
    "xlam-2-7b-fc-r",
    "hermes-4.3-36b",
    "llama-3.3-70b-q4",
]

# Per-model known_issues — mirrored as MLflow tag. Edit + re-run to refresh.
KNOWN_ISSUES = {
    "qwen3-30b-a3b-moe": [
        "Expert-offload requires llama.cpp --n-cpu-moe / -ot exps=CPU",
        "12 GB-card throughput unconfirmed (research is 24 GB)",
    ],
    "gpt-oss-20b-local": [
        "MXFP4 compat with llama.cpp can be flaky; vLLM fallback documented",
        "Cold load slower than active-param speed suggests",
    ],
    "phi-4-reasoning-14b": [
        "MS benchmark claims (beats R1-Distill-Llama-70B on AIME 2025) single-source",
        "Reasoning-mode high token consumption; budget max_tokens >= 2K",
    ],
    "xlam-2-7b-fc-r": [
        "DEFERRED: no pre-quantized GGUF on HF for 2.x as of 2026-05-27",
        "CC-BY-NC-4.0: research-use only; derived adapters inherit",
    ],
    "hermes-4.3-36b": [
        "Single-source benchmarks (Nous-published) — sanity-check before headline use",
        "512K context claim; realistic 32K-64K on DDR5 budget",
    ],
    "llama-3.3-70b-q4": [
        "Hybrid offload only; 6-10 tok/s; --allow-slow-models gate enforced",
        "Cold-load 60-90s from NVMe",
    ],
}


def main() -> int:
    mirror = MlflowMirror()
    if not mirror.enabled:
        print("[backfill] MlflowMirror is disabled (tracking server unreachable)")
        return 1

    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT litellm_id, publisher, variant, capabilities
               FROM models WHERE litellm_id = ANY(%s)""",
            (PHASE_19A_MODELS,),
        )
        rows = cur.fetchall()

    found = {r[0] for r in rows}
    missing = [m for m in PHASE_19A_MODELS if m not in found]
    if missing:
        print(f"[backfill] missing lab.models rows: {missing}")
        return 2

    mirrored = 0
    for litellm_id, publisher, variant, capabilities in rows:
        uri = mirror.log_model_card(
            litellm_id,
            publisher=publisher or "",
            variant=variant,
            capabilities=list(capabilities or []),
            known_issues=KNOWN_ISSUES.get(litellm_id),
        )
        if not uri:
            print(f"[backfill] log_model_card({litellm_id}) returned None — skipped")
            continue
        with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE models SET mlflow_model_uri = %s WHERE litellm_id = %s",
                (uri, litellm_id),
            )
        mirrored += 1
        print(f"[backfill] {litellm_id} -> {uri}")

    print(f"[backfill] mirrored {mirrored}/{len(PHASE_19A_MODELS)} model(s)")
    return 0 if mirrored == len(PHASE_19A_MODELS) else 1


if __name__ == "__main__":
    sys.exit(main())
