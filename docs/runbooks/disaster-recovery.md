---
doc_id: disaster-recovery
title: Runbook — disaster recovery (backup + cold restore)
zone: lab
kind: runbook
status: active
owner: m
created: '2026-05-27'
last_updated: '2026-05-27'
last_verified: '2026-05-27'
tags:
- lab
- runbook
- runbooks
- backup
- recovery
---

# Runbook — disaster recovery (backup + cold restore)

Phase 4.7 shipped a nightly backup; Phase 17.10 verified the restore
half end-to-end. This runbook codifies the procedure and the gap
analysis.

## What is backed up

`scripts/backup.sh` runs nightly (cron / `just backup`) and writes a
timestamped daily snapshot to `${LAB_BACKUP_ROOT}/daily/YYYY-MM-DD_HHMMSS/`
(default root: `/mnt/backup/lab/`). One snapshot contains:

| Artifact            | Source                            | Size (typical) |
| ------------------- | --------------------------------- | -------------- |
| `lab.dump.gz`       | `pg_dump -Fc lab`                 | ~700 KB        |
| `mlflow.dump.gz`    | `pg_dump -Fc mlflow`              | ~10 KB         |
| `litellm.dump.gz`   | `pg_dump -Fc litellm`             | ~300 KB        |
| `minio-lab/`        | `mc mirror lab/lab`               | ~30 MB         |
| `minio-mlflow/`     | `mc mirror lab/mlflow`            | (see gaps)     |
| `lab-code.bundle`   | `git bundle create --all` on repo | ~500 KB        |
| `MANIFEST.txt`      | `pg_dump`/host/sha/sizes          | <1 KB          |

Rotation: 14 dailies retained; older deleted.

## What is NOT backed up (gaps)

These are intentional gaps as of 2026-05-27. Anything in this list is
re-derivable, but recovery requires extra time/manual effort.

- **MinIO `lab-dvc/` bucket.** DVC remote storage. Re-derivable from
  versioned content but not in the snapshot rotation. Add to
  `backup.sh` if DVC objects ever go off-host or grow valuable.
- **Knowledge bases on disk** (`/data/lab/code/kbs/` LanceDB indexes
  and source corpora). Source markdown lives in the git bundle;
  embeddings/indexes are rebuilt by the KB ingest scripts. Costs
  CPU time, no data loss.
- **Ollama model weights.** Re-pullable from upstream. Cost: hours to
  re-download Q4/Q5 quantizations.
- **HF cache.** Re-pullable.
- **Grafana dashboards (`grafana-data/`), Tempo traces, Prometheus
  data.** Operational telemetry only. Re-accrues from new traffic.
- **LiteLLM master key / MinIO secret.** Stored on host filesystem,
  NOT in the snapshot. If the host disk dies, regenerate from the
  service compose stack (Phase 4.6 onboarding).
- **`/data/lab/code` working tree state** (uncommitted edits, .venv,
  `.dvc/cache`). The git bundle restores the committed history but
  not in-flight work.

## Snapshot layout

```
/mnt/backup/lab/daily/2026-05-26_033425/
├── MANIFEST.txt                # backup timestamp, host, pg version,
│                               # lab repo HEAD sha, per-file sizes
├── lab.dump.gz                 # pg_dump -Fc lab
├── mlflow.dump.gz              # pg_dump -Fc mlflow
├── litellm.dump.gz             # pg_dump -Fc litellm
├── minio-lab/                  # mc mirror of lab/lab (manifests/, runs/)
│   ├── manifests/
│   └── runs/
└── lab-code.bundle             # git bundle create --all
```

## Cold restore — verified procedure

The restore script (`scripts/restore.sh`, Phase 17.10) restores into
**namespaced** targets so it never clobbers live state. The drill is
safe to run any time as a smoke test.

```bash
# Restore newest snapshot (default)
scripts/restore.sh --latest

# Restore a specific snapshot
scripts/restore.sh /mnt/backup/lab/daily/2026-05-26_033425

# Drop the namespaced artifacts after a drill
scripts/restore.sh --cleanup
```

Namespaced targets (controlled by `LAB_RESTORE_NS` /
`LAB_RESTORE_BUCKET_NS` env vars):

- Postgres: `lab_restore_test`, `mlflow_restore_test`, `litellm_restore_test`
- MinIO buckets: `lab-restore-test`, `mlflow-restore-test`
- Git working tree: `/tmp/lab-restore-test/repo`

Verification anchors checked at the end of each drill:

- `F-001` row present in restored `findings` (foundational, in every snapshot)
- `findings` count >= 1, `experiment_runs` count >= 1
- `EXP-001` run count >= 1 (older anchor, present from Phase 1)
- MinIO bucket non-empty after mirror
- Restored git HEAD sha == manifest sha

### Real-disaster procedure (clobbering, not namespaced)

When the live DB is unrecoverable and you actually need to overwrite it:

```bash
# 1. Stop services that write to lab/* DBs and MinIO
podman-compose -f /data/lab/services/compose.yml down

# 2. Backup the live state ONE MORE TIME (so we have an immediate fallback)
just backup     # writes a fresh snapshot

# 3. Drop & restore each DB (no _restore_test suffix, real names)
SNAP=/mnt/backup/lab/daily/2026-05-26_033425
for db in lab mlflow litellm; do
    dropdb "$db"
    createdb "$db"
    gunzip -c "$SNAP/$db.dump.gz" | pg_restore --no-owner --no-privileges --dbname="$db"
done

# 4. Restore MinIO buckets (replace, not mirror)
SECRET=$(cat /data/lab/services/minio-secret)
for b in lab mlflow; do
    podman run --rm --network host --entrypoint /bin/sh \
        -v "$SNAP/minio-$b:/src:Z" \
        -e MC_HOST_lab="http://labadmin:$SECRET@localhost:9000" \
        docker.io/minio/mc:latest \
        -c "mc rb --force lab/$b; mc mb lab/$b; mc mirror --quiet /src lab/$b"
done

# 5. Restore git repo (only if /data/lab/code is gone)
git clone "$SNAP/lab-code.bundle" /data/lab/code

# 6. Restart services
podman-compose -f /data/lab/services/compose.yml up -d

# 7. Smoke gate
just check
```

## RTO (recovery time objective) vs reality

| Phase                   | Observed (2026-05-27 drill, 30 MB snapshot) |
| ----------------------- | ------------------------------------------- |
| Postgres restore (3 DB) | ~2 s                                        |
| MinIO mirror restore    | ~1 s                                        |
| Git bundle clone        | <1 s                                        |
| Verification            | <1 s                                        |
| **Total wall time**     | **4 s**                                     |

The above is for a small snapshot (~30 MB total). RTO scales roughly
linearly with MinIO bucket size; budget ~1 minute per GB of MinIO data
on the current pod. Postgres dumps decompress at ~50 MB/s so are
negligible up to ~10 GB.

Real-disaster RTO budget (clobber path, including service stop/start
and a fresh pre-disaster backup): **~5 minutes** for current data
volume.

## Cadence

- Backups: nightly cron (3:34 AM local).
- Drill: monthly, manually. Open a task with title
  `restore-drill-YYYY-MM` and link this runbook. The drill is
  cleanup-safe (`--cleanup` removes all `*_restore_test` artifacts).
- After any schema migration: run the drill within 24 h to confirm
  the new schema restores cleanly.

## Verified

- 2026-05-27: drill on snapshot 2026-05-26_033425 — PASS, wall time
  4 s, all five anchors green (F-001 row, findings>=1, runs>=1, EXP-001
  runs>=1, MinIO non-empty, git sha matches manifest).
