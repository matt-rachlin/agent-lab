# Changelog

All notable changes to this lab are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Backfilled at phase granularity on 2026-06-11 from git history (the file
sat untouched at 0.0.1 from scaffold until then).

## [Unreleased]

### Added
- EXP-009/HARD-BENCH-003 pre-registered: N=8 seed confirmation of the
  hard-suite ranking per ADR-004 (`conf/sweep/hard-bench-v3.yaml`).
- Retroactive experiment records EXP-007 (CODER-BENCH-001) and EXP-008
  (HARD-BENCH-001/002); findings F-012 (agentic tool-calling failure
  modes) and F-013 (prompt robustness is a model property).
- CARD.md for pbs-agent-hard-v0.1.

## [0.3.0] — 2026-06-10 — local coding-agent benchmark campaign

### Added
- `pbs-agent-hard-v0.1`: 32-task hard agentic suite (code/data/shell/multi
  × 8; offline HTTP fixtures; machine-verified answers).
- `tool_use_system_v2` prompt: one strict act-don't-narrate sentence;
  +15.6pp for Devstral-24B at suite scale (F-013).
- LiteLLM lanes: qwen3-coder-30b, qwen2.5-coder-32b-q4_k_m, devstral-24b.
- Sweeps: CODER-BENCH-001 (12 tasks × 3 seeds × 3 models),
  HARD-BENCH-001/002 (32 tasks × 3 models, v1/v2 prompt A/B).
- Harness: text-tool-call fallback parser in the Inspect solver
  (diagnostic for models that emit tool calls as content text).
- Public-facing README and `docs/writeups/local-coding-agent-benchmark.md`;
  repo flipped public (github.com/matt-rachlin/agent-lab).

### Fixed
- `all_of` success predicate wired into the scorer dispatch (was
  implemented but unreachable; 7 hard-suite tasks use it).
- `lab analyze report`: agent-path pass rates now read from MLflow when
  `eval_results` is empty (agent experiments no longer render blank).
- model_pool preflight 400: llama-swap registers models without the
  `-local` suffix; pool lookup now strips it.
- llama-3.3-70b OOM segfault: n-gpu-layers 16 → 10.
- 3 hard-suite `multi` tasks pointed at invented fixture subdomains that
  NXDOMAIN'd in the sandbox; moved to reserved domains (example.com/org/net).
- Orphan adhoc MLflow runs pruned.

## [0.2.0] — 2026-05-27 .. 2026-06-09 — external anchoring + lab infrastructure

### Added
- EXP-005/F-011: BFCL v3 AST vendored (1000 tasks); local qwen3-14b-q4
  ties best cloud arm (0.910 vs 0.925, CI straddles zero).
- EXP-006/006b, F-009/F-010: qwen3-30b-MoE evaluated, not promoted.
- Phase 13–19: pre-commit gates (ruff/mypy/pyright/gitleaks), model +
  task-suite cards, docs-lint + doc-meta backfill, DVC KB versioning
  (MinIO remote, ADR-006), Streamlit dashboard (findings/experiments/
  leaderboard), MLflow Tracing dual-emit + OTel/Tempo, GPU Prometheus
  exporter, spend tracking, MLflow score backfill for single-turn/BFCL
  cells.

## [0.1.0] — 2026-05-25 .. 2026-05-27 — harness build-out + first experiments

### Added
- Phase 1–5: sweep harness + `lab` CLI + analyzer; evaluator framework +
  judge; PBS v0.1 suite; reliability discipline (ADR-004, N≥8 + pass^k;
  F-002); observability + docs discipline.
- Phase 6: agent path — Podman+gVisor sandboxes, 6 FastMCP tool servers,
  Inspect solver loop, agent scorers, PBS-Agent v0.1 (12 tasks), RAG
  slice (kb_query tool, retrieval scorers).
- Phase 7–12: cross-encoder reranker + RRF, Valkey two-tier RAG cache,
  parent-child chunking (schema v2), smart rerank skip, HyPE indexing,
  multi-query expansion.
- EXP-001/001b (12 GB Agent v0.1; F-003/F-004), EXP-002 (F-005: local
  tool-call accuracy is real; end-state chaining is the constraint),
  EXP-003a/b (F-006: hybrid retrieval wins), EXP-004a/c (rerank +10pp
  REFUTED, +5pp validated; rerank reverted to opt-in).

## [0.0.1] — 2026-05-25

Initial scaffold.
