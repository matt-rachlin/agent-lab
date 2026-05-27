---
doc_id: f-001-phase1-plumbing-validated
title: 'F-001: Phase 1 sweep harness produces persisted, queryable runs end-to-end'
zone: lab
kind: finding
status: active
owner: m
created: '2026-05-25'
last_updated: '2026-05-25'
last_verified: '2026-05-25'
tags:
- lab
- finding
- findings
---
# F-001: Phase 1 sweep harness produces persisted, queryable runs end-to-end

Date: 2026-05-25
Confidence: high
Source: EXP SWEEP-SMOKE-001

## Claim

The lab's Phase 1 plumbing — sweep config → matrix expansion → LiteLLM call → trace upload → Postgres row → DuckDB analyze report — works end-to-end for a 60-cell sweep across three local 12 GB models, with zero harness crashes and verified resumability.

## Evidence

`uv run lab sweep run conf/sweep/smoke.yaml` completed in **~4 minutes** on a single RTX 3080 Ti:

- **60/60 cells reached `status='done'`** (0 errors)
- **60 trace blobs uploaded to MinIO** (one per cell, in `s3://lab/runs/2026-05/25/<run_id>/trace.jsonl`)
- **60 manifests persisted** to the `manifests` table (one per cell)
- **Re-running the same sweep with `--resume`** detected all 60 cells as already-done and executed 0

### Per-model summary (from `lab analyze report SWEEP-SMOKE-001`)

| model | backend | n | done | err | latency p50 (ms) | latency p95 (ms) | tokens_out mean |
|---|---|---:|---:|---:|---:|---:|---:|
| gemma3-12b-q4 | ollama-local | 20 | 20 | 0 | 294 | 3,275 | 7.6 |
| llama3.1-8b-q4 | ollama-local | 20 | 20 | 0 | 110 | 417 | 6.2 |
| qwen3-14b-q4 | ollama-local | 20 | 20 | 0 | 10,393 | 11,396 | 186.8 |

### Incidental observations (not the focus of this finding, but worth noting)

- **`qwen3-14b-q4` is doing reasoning by default** — output is ~30× larger than the other two (186 vs 6-7 tokens mean) and latency is correspondingly higher. The smoke prompts asked for short answers; qwen3 elaborated. For Phase 2 sweeps we'll need a sampling-config option to suppress chain-of-thought, or a separate prompt family that hides the reasoning channel.
- **Cold-start spikes are real and large**: `gemma3-12b-q4` has p50=294ms but p95=3,275ms. The first call to a freshly loaded model dominates. Outer-loop-by-model (which the sweep does) amortizes this; concurrent multi-model sweeps would be much worse.
- **No quality scores in this report** — Phase 1 doesn't run evaluators. The smoke tasks have rubrics in `tasks.payload.rubric` but they aren't yet applied to traces. That's Phase 2.

## Caveats / limits

- Plumbing-only validation: this finding says **nothing** about the quality of any model's outputs on the smoke tasks. The smoke tasks were chosen to be trivial; differences in tokens_out and latency are about model behavior under default settings, not capability.
- All runs sequential (`max_concurrency=1`). Phase 1 doesn't yet exercise cloud routing.
- Cost tracking is null in Phase 1; LiteLLM's spend ledger lands in Phase 4.

## Implications

- The sweep harness is ready to use for real experiments. Phase 2 evaluators can layer on top of the existing trace store without changing the sweep code.
- Resumability is verified — interrupted sweeps can resume cleanly, and re-running a fully-done sweep is a no-op.
- The qwen3 reasoning-by-default observation is a real input to EXP-001's prompt-axis design.

## Open questions

- How much of the latency variance is intrinsic vs cold-start vs scheduler? Defer to Phase 4 observability + a real reliability sweep.
- Should the sweep collect a per-cell GPU utilization sample? Cheap to add via `nvidia-smi` snapshot in the manifest extra; worth doing in Phase 2.

## Status

- [x] Logged
- [ ] Replicated
- [ ] Published
