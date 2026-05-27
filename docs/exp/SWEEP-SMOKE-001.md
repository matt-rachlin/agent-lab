---
doc_id: sweep-smoke-001
title: 'SWEEP-SMOKE-001: Phase-1 end-to-end pipeline validation'
zone: lab
kind: exp
status: active
owner: m
created: '2026-05-25'
last_updated: '2026-05-25'
last_verified: '2026-05-25'
tags:
- lab
- exp
---
# SWEEP-SMOKE-001: Phase-1 end-to-end pipeline validation

Date created: 2026-05-25
Status: planned
Pre-registered: <git commit sha>

## Hypothesis

The lab's Phase 1 sweep plumbing executes 60 cells (3 models × 2 configs × 5 tasks × 2 seeds) end-to-end:
- every cell produces an `experiment_runs` row
- every cell uploads a `trace.jsonl` blob to MinIO
- every cell captures and references a manifest
- resumability works (re-running skips done cells)
- the `lab analyze report` command produces a coherent per-model summary

## Why this matters

Without a validated sweep harness, no real research question can be executed.

## Method

- Models: `qwen3-14b-q4`, `llama3.1-8b-q4`, `gemma3-12b-q4` (all local, 12 GB-fitting)
- Configs: `greedy` (T=0, top_p=1.0), `sampled` (T=0.7, top_p=0.9), `max_tokens=200`
- Tasks: full `smoke` suite (5 trivial tasks — arithmetic, capitals, JSON, reasoning)
- Seeds: [1, 2]
- Eval: none in Phase 1 (just plumbing); evaluators land in Phase 2

## Success / failure criteria (defined before running)

- All 60 cells execute (60 `experiment_runs` rows, status='done' or 'error')
- ≥ 90% of cells reach `status='done'` (some token-budget edge cases acceptable)
- Each cell has a `manifest_sha` and `trace_path`
- Re-running the sweep without `--no-resume` skips all 60 cells (0 executed)
- `lab analyze report SWEEP-SMOKE-001` produces a markdown report with both summary tables

## Confounders to control

- N/A — this is plumbing verification, not a real comparison

## Kill criteria

- If any infrastructure error (DB, MinIO, LiteLLM, Ollama) prevents any cell from completing, stop and fix.

## Pre-mortem

- LiteLLM 16K-token clamp could clip a verbose response — mitigated by `max_tokens=200`
- GPU lease contention with other host work — single-user box, low risk
- Ollama daemon swap overhead — 3 × ~15s = ~45s of model swap, acceptable

## Estimated cost

GPU-hours: ~0.5 (60 calls × ~25s each, sequential)
Cloud calls: 0
Wall time: ~25 min
