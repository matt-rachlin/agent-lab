#!/usr/bin/env bash
# Wrapper for the 70B ceiling-llm sweep: evicts the rerank server's GPU
# tensors before launching the sweep, restores rerank availability after.
#
# Why this exists (Fix for #73, 2026-05-27):
#   * Phase 19e tuned `llama-3.3-70b-q4` to --n-gpu-layers=14 because the
#     rerank server (lab.rag.rerank_server, systemd `rerank.service`,
#     port 8401) holds ~2.6 GB persistent VRAM. That cuts the 12 GB card
#     to ~8.5 GB free; ngl=21 OOMs and throughput collapses to ~1.8 tok/s.
#   * llama-swap's group system cannot evict the rerank server because
#     rerank.service is NOT a llama-swap-managed process — llama-swap
#     just proxies to it. The persistent: true on the small-tools group
#     is a llama-swap statement about itself.
#   * The rerank server, however, exposes POST /unload which releases
#     the cross-encoder's VRAM but keeps the FastAPI process alive. The
#     next /rerank call auto-reloads. That's the right primitive here.
#
# What this script does:
#   1. Snapshots which Ollama models are resident (so we can complain if
#      they're hogging VRAM; we don't auto-evict them — leave that to
#      the operator and llama-swap's normal eviction during the sweep).
#   2. POSTs /unload to the rerank server.
#   3. Optionally hits llama-swap's /api/models/unload/<id> for any small
#      models loaded under the small-tools / medium-llm groups.
#   4. Runs the passed-through command (e.g.
#        ./ceiling-sweep-wrapper.sh .venv/bin/lab sweep run ...
#      ) in the foreground.
#   5. On exit (success OR failure), prints a one-line notice telling
#      the operator the rerank server will auto-reload on next call —
#      we do NOT proactively reload because the sweep may have other
#      models still resident.
#
# Usage:
#   scripts/ceiling-sweep-wrapper.sh .venv/bin/lab sweep run conf/sweep/<sweep>.yaml --allow-slow-models
#   scripts/ceiling-sweep-wrapper.sh .venv/bin/lab agent run --suite pbs-agent-v0.1 --task fs-read-and-copy --model llama-3.3-70b-q4-local --allow-slow-models
#
# After the wrapper has freed VRAM, you can also bump the
# `--n-gpu-layers` value in conf/serving/llama-swap.yaml from 14 → 21 for the
# duration of this sweep (Phase 19e measured ngl=21 OOMs at 8.5 GB free;
# with the rerank server's 2.6 GB freed, the budget rises to ~11 GB and
# ngl=21 fits — see Phase 19e tuning notes in the runbook).
#
# Exit codes:
#   - Whatever the wrapped command exits with.
#   - 2 if rerank unload fails AND --strict was passed.
#
# This script is intentionally simple. It is NOT a daemon, NOT a service,
# and does NOT auto-restore — keep operator control over reload timing.

set -euo pipefail

RERANK_URL="${LAB_RERANK_URL:-http://127.0.0.1:8401}"
STRICT=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict) STRICT=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; break ;;
    *) break ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "usage: $0 [--strict] [--dry-run] -- <command...>" >&2
  exit 64
fi

log() { printf '[ceiling-sweep] %s\n' "$*" >&2; }

# --- 1. snapshot current GPU residents (advisory only) -------------------------
if command -v nvidia-smi >/dev/null; then
  log "GPU state before unload:"
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader >&2 || true
fi

# --- 2. unload rerank server ---------------------------------------------------
if [[ $DRY_RUN -eq 1 ]]; then
  log "[dry-run] would POST ${RERANK_URL}/unload"
else
  if curl -fsS -X POST "${RERANK_URL}/unload" -o /tmp/.rerank-unload.json --max-time 10; then
    log "rerank unload OK: $(cat /tmp/.rerank-unload.json)"
  else
    log "WARNING: rerank unload failed (server may be down)"
    if [[ $STRICT -eq 1 ]]; then
      log "--strict set, aborting"
      exit 2
    fi
  fi
fi

# --- 3. unload everything in llama-swap's small-tools (defensive) --------------
# In the current config, the only small-tools member is qwen3-reranker-0.6b
# which is just a proxy to the rerank server we already unloaded. The call
# below also handles any future small-tools members. It is best-effort.
for model_id in qwen3-reranker-0.6b; do
  if [[ $DRY_RUN -eq 1 ]]; then
    log "[dry-run] would POST http://localhost:8080/api/models/unload/${model_id}"
  else
    curl -fsS -X POST "http://localhost:8080/api/models/unload/${model_id}" \
      --max-time 5 >/dev/null 2>&1 \
      && log "llama-swap unload ${model_id} OK" \
      || true
  fi
done

if command -v nvidia-smi >/dev/null; then
  log "GPU state after unload (settling 2s):"
  sleep 2
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader >&2 || true
fi

# --- 4. run wrapped command ----------------------------------------------------
log "executing: $*"
if [[ $DRY_RUN -eq 1 ]]; then
  log "[dry-run] would run the above; exiting 0"
  exit 0
fi

# Trap to print the restore-hint regardless of outcome.
trap 'log "wrapper exiting; rerank server will auto-reload on next /rerank call"' EXIT

"$@"
