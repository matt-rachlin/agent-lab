# ADR-001: Storage stack — Postgres + MinIO + DuckDB

Status: accepted
Date: 2026-05-25
Deciders: Matt Rachlin

## Context

The lab needs storage for: structured records (experiments, runs, results), trace blobs (per-run JSONL ≤ 1 MB each, accumulating to GBs), curated datasets, model artifacts, MLflow's tracking data.

We need: privacy (no SaaS lock-in), single-machine simplicity, queryability with SQL, blob storage with an S3-compatible API for downstream tool compatibility (DVC, MLflow, anything cloud-native we may use later).

## Decision

- **PostgreSQL 18** (existing host install) hosts three databases:
  - `lab` — research records (experiments, runs, evals, findings, models, etc.)
  - `mlflow` — MLflow's tracking backend (kept separate to avoid table name collisions like `experiments` / `datasets` / `runs`)
  - `litellm` — LiteLLM proxy's spend ledger + API key store
- **MinIO** (Podman container, port 9000) hosts two buckets:
  - `lab` — trace blobs, manifests, per-run outputs, datasets
  - `mlflow` — MLflow's artifact store (model checkpoints, plots, etc.)
- **DuckDB** is the analytical query engine for offline analysis — points at Parquet dumps in `lab/runs/` and/or `postgres_scanner` for live joins to `lab` DB.

All three are local-first. No cloud storage. MinIO is S3-compatible, so future-cloud or future-DVC integrations are one config line.

## Consequences

- **Easier**: SQL queries across all research records; MLflow + DVC + LiteLLM all speak Postgres/S3 natively; backup is `pg_dump` + rsync of MinIO data dir; everything works offline.
- **Harder**: three databases to back up (vs one); separate auth concerns in each; the `lab` DB needs schema discipline to avoid clashing with MLflow's table names if we ever consolidate.
- **Risks**: `/data` is RAID0 (no redundancy) — both Postgres data and MinIO live there. Nightly backup of `pg_dump lab > /backup/lab.sql` and MinIO bucket sync to a non-RAID disk is mandatory. (Tracked as a Phase 4 task.)

## Considered alternatives

- **SQLite for everything** — simpler, but no concurrent writers, no JSONB indexing, no pgvector. Rejected.
- **DuckDB as primary OLTP** — DuckDB isn't designed for high-concurrency writes; fine as the analytical layer, wrong as the transactional store.
- **Single Postgres database for lab + MLflow + LiteLLM** — collides on table names (`experiments`, `runs`, `datasets`); cleaner to separate.
- **lakeFS / DuckLake** — overkill for solo work; DuckLake is also too new (one more release). DVC + MinIO covers the dataset-versioning need.
- **Local filesystem only (no MinIO)** — works but every downstream tool (MLflow, DVC, future-cloud) wants S3; one Podman container is cheap.
