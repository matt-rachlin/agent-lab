---
doc_id: dashboard-readme
title: 'apps/eval-dashboard: local Streamlit dashboard for the lab'
zone: lab
kind: readme
status: active
owner: m
created: '2026-05-27'
last_updated: '2026-05-27'
last_verified: '2026-05-27'
depends_on:
- kind: code
  target: lab:apps/eval-dashboard/Home.py
- kind: code
  target: lab:apps/eval-dashboard/pages/1_Findings.py
- kind: code
  target: lab:apps/eval-dashboard/pages/2_Experiments.py
- kind: code
  target: lab:apps/eval-dashboard/pages/3_Leaderboard.py
- kind: code
  target: lab:apps/eval-dashboard/pages/4_Sweep_Monitor.py
- kind: code
  target: lab:apps/eval-dashboard/pages/5_Docs.py
- kind: doc
  target: dashboard-claude
tags:
- streamlit
- dashboard
- phase-15.3
---

# apps/eval-dashboard

Local-only Streamlit dashboard for the lab. Five pages: Home, Findings,
Experiments, Leaderboard, Sweep Monitor, Docs.

## Run

```sh
# from repo root
uv sync -E dash       # one-time
just dash             # starts http://localhost:8501
```

## Architecture

The dashboard intentionally does **NOT** import from `lab.*`. It reads
Postgres + MinIO + Valkey directly via `psycopg`, `boto3`, and a tiny
`redis` import in the sweep monitor. This decouples it from the
src/lab/ -> packages/lab-* refactor in Phase 15.1: the dashboard keeps
working even if every package boundary moves.

```
apps/eval-dashboard/
├── Home.py                  # service health, stats, sweeps, findings
├── pages/
│   ├── 1_Findings.py        # F-NNN list + body + dep graph
│   ├── 2_Experiments.py     # per-experiment summary + cell drilldown
│   ├── 3_Leaderboard.py     # per-model aggregates over agent_logs.turns
│   ├── 4_Sweep_Monitor.py   # 10s live; GPU lease; rerank queue
│   └── 5_Docs.py            # SQLite doc graph from ~/db/m/docs.db
├── lib/
│   ├── db.py        # Postgres + DuckDB cache
│   ├── minio.py     # S3 via boto3 (endpoint_url=MinIO)
│   ├── services.py  # health probes
│   └── docs.py      # SQLite read-only against ~/db/m/docs.db
├── tests/
│   └── test_lib.py  # 17 unit tests; no live services required
├── .streamlit/config.toml
├── README.md
└── CLAUDE.md
```

## Env vars

Inherited from `~/.env` at repo root (or the shell):

| var | default | notes |
| --- | --- | --- |
| `LAB_PG_DSN` | `postgresql://m@/lab` | Unix socket; override if remote |
| `LAB_S3_ENDPOINT` | `http://localhost:9000` | MinIO |
| `LAB_S3_BUCKET` | `lab` | |
| `LAB_S3_ACCESS_KEY` | `labadmin` | |
| `LAB_S3_SECRET_KEY` | (empty) | required for real reads |
| `LAB_REDIS_URL` | `redis://localhost:6379/0` | Valkey |
| `LAB_LITELLM_URL` | `http://localhost:4000` | |

Missing services degrade to red dots / empty tables; the dashboard
always renders.

## Tests

```sh
cd apps/eval-dashboard
uv run pytest tests/ -q
```

Tests run without any live services - everything is monkey-patched
against dead ports or an in-memory SQLite to verify the silent-fallback
behavior.
