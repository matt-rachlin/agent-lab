---
doc_id: reliability-001
title: 'RELIABILITY-001: variance characterization for 3 local 12 GB models at N=8
  seeds'
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
# RELIABILITY-001: variance characterization for 3 local 12 GB models at N=8 seeds

Date created: 2026-05-25
Status: planned

## Hypothesis

H1: On easy/medium PBS tasks, llama3.1-8b-q4 will have `pass^8 / pass@1 >= 0.75` (reliability ratio).

H2: qwen3-14b-q4 with default reasoning enabled will exhibit visible token-budget pressure even at `max_tokens=1024`, manifesting as a non-zero rate of empty responses or low pass@1.

H3: All three models will achieve `pass@1 = 1.0` on at least one easy task (math-001 `(47*8) - (12*19) = 148`), establishing a sanity floor.

## Why this matters

The minimum required calibration for any future Pareto claim: how much variance does each model have at temperature 0 on tasks they "should" know? Without this, `pass@1 = 0.7 vs 0.8` is unfalsifiable noise.

## Method

- Models: `qwen3-14b-q4`, `llama3.1-8b-q4`, `gemma3-12b-q4`
- Config: greedy, `temperature=0.0`, `top_p=1.0`, `max_tokens=1024`
- Tasks: 5 from PBS-v0.1 (math-001, math-002, fmt-001, know-001, know-004)
- Seeds: 8 (1..8)
- Evaluators (deterministic only — judge slice runs separately):
  - `exact_match`, `regex_match`, `not_empty`, `latency_under`, `tokens_under`, `json_valid`
- Pre-registered statistics:
  - per-(model, task): pass@1, pass^4, pass^8, bootstrap 95% CI
  - per-model reliability ratio = pass^8 / pass@1, averaged across tasks

## Success / failure criteria

- All 120 cells reach `status='done'` (errors permitted only for infrastructure failures)
- Report tables populate `pass^8` (non-zero where pass@1 ≥ 7/8)
- A finding `F-002` is filed regardless of direction, documenting numbers + interpretation.

## Confounders to control

- Model swap order — `outer = model` (per Phase 1 playbook), minimizes cold-start variance
- GPU lease — only one model loaded at a time
- Inference backend — local Ollama daemon for all

## Kill criteria

- If any infra component (DB, MinIO, Ollama, LiteLLM) prevents any cell from completing, stop, fix, restart.

## Pre-mortem

- qwen3 thinking can blow past 1024 tokens for math word problems — possible non-zero empty rate. Mitigation: log + analyze; if too severe, add a `disable-thinking` prompt variant in a follow-up sweep.
- 3rd-party Ollama daemon could OOM on model swap — mitigated by sequential outer-loop-by-model.

## Estimated cost

GPU-time: ~25 min (3 models × ~8 min/model)
Cloud calls: 0
Wall time: ~25 min
