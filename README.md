# agent-lab

A solo research lab for comparing agentic AI workflows across models, prompting strategies, and hardware constraints. Built on a single RTX 3080 Ti (12 GB VRAM), it runs the full stack — model serving, sandboxed agent execution, multi-seed sweep orchestration, and trace-level observability — to produce findings that are statistically defensible and failure-mode-transparent. The headline result so far: a 12B model (gemma4-12b) beats models 2–6× its size at agentic coding — 100% vs 58% (Llama-3.3-70B) and 0% (Qwen2.5-Coder-32B) on the base suite, and 94% vs 78% (Qwen3-Coder-30B) and 53% (Devstral-24B) on a 32-task hard suite — because agentic tool-calling fidelity, not size or coding skill, is the gating factor. Trace-level analysis (trajectory JSONL in S3, MLflow spans) identified three distinct failure modes: tool calls emitted as JSON text in `content`, premature chain termination, and plan narration in place of action. Full story: [docs/writeups/local-coding-agent-benchmark.md](./docs/writeups/local-coding-agent-benchmark.md).

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         lab (uv workspace)                          │
│                                                                     │
│  lab-core          settings, migrations, gpu_lease, minio_io        │
│    └── lab-rag     chunker, embedder, LanceDB index, rerank         │
│          └── lab-agent   ToolPool, gVisor sandbox, MCP tool servers │
│                └── lab-eval    evaluator framework, judge, scorers   │
│                      └── lab-inspect   Inspect AI adapter + scorers  │
│                            ├── lab-sweep   Hydra sweeps, DuckDB      │
│                            │               analysis, MLflow logging  │
│                            └── lab-observability  GPU Prometheus     │
│                                           exporter, spend, OTel      │
│                                     │                               │
│                                  lab-cli  (Typer; all commands)     │
└─────────────────────────────────────────────────────────────────────┘

Inference layer
  llama-swap (model-slot manager) → Ollama daemon
  LiteLLM proxy (port 4000) → unified OpenAI-compatible endpoint
  pueue gpu group → serialized job queue (no VRAM contention)

Persistence
  Postgres  — experiment registry, spend tracking, Alembic migrations
  MinIO     — trajectory JSONL, sweep artifacts (S3-compatible)
  MLflow    — 10 000+ scored runs, full param/metric/trace capture
  LanceDB   — RAG vector index

Observability
  OTel / Tempo  — distributed traces
  Prometheus    — GPU utilization, VRAM, spend quota
```

## Capabilities

- **Multi-seed sweep engine** — Hydra-driven parameter sweeps with pre-registered hypotheses, automatic `pass^k` (k=1,4,8) + bootstrap 95% CI reporting (ADR-004: N≥8 seeds is the non-negotiable lab default).
- **Sandboxed agent execution** — Podman + gVisor OCI runtime; each agent run gets a fresh workspace with verified end-state predicates (file-content equality, DB queries, composite `all_of`).
- **Tool suite** — `fs_read/write/grep`, `shell_exec`, `python_eval`, `http_fetch`, `kb_query`; pooled via `ToolPool`, served over MCP.
- **Inspect AI integration** — `lab-inspect` wraps Inspect AI solvers and scorers; the same task definitions run under both the native sweep runner and Inspect AI for cross-validation.
- **RAG pipeline** — LanceDB index with hybrid BM25 + vector retrieval, client/server reranker, query expansion, and result caching.
- **Trace-level failure analysis** — MLflow Tracing dual-emit (Traces tab) + raw trajectory JSONL in MinIO; postmortems are linked from findings by commit SHA.
- **Model serving on 12 GB** — llama-swap manages model slot turnover; 70B models run via llama.cpp hybrid CPU/GPU offload (Q4_K_M, n-gpu-layers tuned to avoid OOM segfaults); gemma4-12b, Qwen3-Coder-30B, Qwen2.5-Coder-32B all routable.

## Benchmark results

### CODER-BENCH-001 — base agentic suite (pbs-agent-v0.1, 12 tasks)

react scaffold · 3 seeds · temp 0

| Model | Pass rate | Notes |
|---|---|---|
| gemma4-12b (local) | **100%** | Structured tool calls |
| Qwen3-Coder-30B (local) | **100%** | Structured tool calls |
| Llama-3.3-70B (local, hybrid offload) | 58% | Emits tool calls as text; stops mid-chain |
| Qwen2.5-Coder-32B | **0%** | Emits tool calls as JSON text in `content` |

### HARD-BENCH-001/002 — hard agentic suite (pbs-agent-hard-v0.1, 32 tasks, 4 categories)

react scaffold · temp 0 · single seed · prompt-robustness A/B (v2 adds one strict act-don't-narrate sentence)

| Model | v1 prompt | v2 prompt | Δ |
|---|---|---|---|
| gemma4-12b | **93.8%** | **93.8%** | 0 |
| Qwen3-Coder-30B | 78.1% | 81.3% | +3.2 |
| Devstral-24B | 37.5% | 53.1% | **+15.6** |

Categories: `code`, `data`, `shell`, `multi` (multi-file bug hunts, ETL pipelines, multi-hop HTTP fixture chains) — every task has a machine-verified answer and end-state predicate. The v1→v2 delta is itself a finding: prompt robustness is a model property, and single-prompt benchmarks silently measure it.

### BFCL-v3 AST — 1000 tasks (EXP-005, F-011)

| Model | Score | 95% CI |
|---|---|---|
| qwen3-14b-q4 (local, reasoning off) | **0.910** | — |
| glm-5.1-cloud (best cloud arm) | 0.925 | paired delta +0.015, CI [-0.002, +0.032] |

Local ties cloud; CI straddles zero. Two of three cloud arms collapse on parallel function calling.

### PBS-Agent v0.1 — 12 tasks, 5 models (EXP-002, F-005)

480 cells · 8 seeds · temp 0

| Cohort | Mean `tool_correctness` |
|---|---|
| Cloud (gpt-oss-20b, gpt-oss-120b, glm-5.1) | 0.965 |
| Local (qwen3-14b-q4, llama3.1-8b-q4) | 0.833 |
| qwen3-14b-q4 alone | **1.000** |

Finding: local tool-call accuracy is not the bottleneck. The binding constraint is end-state chaining — correct individual tool calls that don't compose into a working solution.

## Engineering notes

- **Strict typing** — `mypy --strict` on all packages; pyright baseline tracked (5 pre-existing errors at Phase 15.1). Violations block the gate (`just check`).
- **Migrations** — Alembic for all schema changes; no ad-hoc DDL.
- **ADRs** — 7 architecture decision records (`docs/adr/`) covering storage stack, inference routing, task taxonomy, reliability discipline, failure handling, data versioning, and RAG registry.
- **Task cards** — every benchmark suite has a machine-readable `CARD.md` with categories, difficulty distribution, tool union, predicate types, and links to experiments/findings.
- **Pre-registered experiments** — each `EXP-*.md` commits hypotheses before data collection; `lab exp register` stamps the SHA. Findings report all pre-registered verdicts regardless of direction.
- **Postmortems** — infra failures and benchmark regressions get postmortems (`docs/postmortems/`) with `resolved_by` SHA. No silent fixes.
- **Test suite** — 718 unit tests; single `tests/` tree spanning all 8 packages.
- **158 commits** from initial scaffold to current phase; CHANGELOG.md tracks phase boundaries.

## Quickstart

```bash
git clone https://github.com/matt-rachlin/agent-lab
cd agent-lab
just bootstrap        # uv sync --all-extras
just db-init          # Postgres lab DB + Alembic migrations
just services-up      # MinIO + MLflow + LiteLLM proxy (Podman)
just check            # ruff + mypy --strict + 718 unit tests
lab sweep run conf/sweep/coder-bench-v1.yaml   # run a sweep
lab analyze report                          # DuckDB → markdown report
```

Requires: Postgres, Podman, Ollama daemon, Python 3.13 via uv. GPU optional for text evals; required for local model inference.

## License

MIT
