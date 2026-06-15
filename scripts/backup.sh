#!/usr/bin/env bash
# Nightly backup: lab DBs + MinIO bucket sync → /mnt/backup/lab/
# Idempotent: keeps the last 14 daily snapshots, rotates older to weekly.
# Designed to run from cron OR `just backup`.

set -euo pipefail

source ~/scripts/jobs-status.sh
js_job_start "backup-$(date +%Y-%m-%d)"
js_install_default_traps

BACKUP_ROOT="${LAB_BACKUP_ROOT:-/mnt/backup/lab}"
TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
DAY_DIR="${BACKUP_ROOT}/daily/${TIMESTAMP}"

mkdir -p "${DAY_DIR}"

echo "[$(date)] backup → ${DAY_DIR}"

# Postgres dumps (3 lab-related DBs)
js_log "pg_dump: lab mlflow litellm"
for db in lab mlflow litellm; do
    echo "  pg_dump ${db}..."
    js_log "pg_dump ${db}..."
    pg_dump -Fc "${db}" > "${DAY_DIR}/${db}.dump"
    gzip "${DAY_DIR}/${db}.dump"
done

# MinIO buckets via mc (uses host secret)
SECRET_FILE="/data/lab/services/minio-secret"
if [[ -f "${SECRET_FILE}" ]]; then
    SECRET="$(cat "${SECRET_FILE}")"
    echo "  mc mirror lab/..."
    js_log "mc mirror lab/lab + lab/mlflow"
    podman run --rm --network host --entrypoint /bin/sh \
        -v "${DAY_DIR}:/backup:Z" \
        -e MC_HOST_lab="http://labadmin:${SECRET}@localhost:9000" \
        docker.io/minio/mc:latest \
        -c 'mc mirror --quiet --overwrite lab/lab /backup/minio-lab && mc mirror --quiet --overwrite lab/mlflow /backup/minio-mlflow'
else
    echo "  (skipping MinIO sync — no secret file)"
    js_log "MinIO sync skipped (no secret file)"
fi

# AWQ model artifacts (wave-2 finding: qwen3-4b-awq on RAID0 with zero backup)
# Mirror at most weekly (Sunday) to avoid spending 3.3 GB of backup budget daily.
if [[ -d /data/lab/models/awq/qwen3-4b-awq ]] && [[ "$(date +%u)" == "7" || ! -f "${BACKUP_ROOT}/awq-last-sync" || $(( $(date +%s) - $(date -r "${BACKUP_ROOT}/awq-last-sync" +%s 2>/dev/null || echo 0) )) -gt 604800 ]]; then
    if [[ -f "${SECRET_FILE}" ]]; then
        echo "  mc mirror lab/models/awq/..."
        js_log "mc mirror awq-models"
        SECRET="$(cat "${SECRET_FILE}")"
        podman run --rm --network host --entrypoint /bin/sh \
            -v /data/lab/models/awq:/awq:ro,Z \
            -e MC_HOST_lab="http://labadmin:${SECRET}@localhost:9000" \
            docker.io/minio/mc:latest \
            -c 'mc mirror --quiet --overwrite /awq lab/awq-models'
        touch "${BACKUP_ROOT}/awq-last-sync"
    else
        echo "  (skipping AWQ mirror — no secret file)"
    fi
else
    echo "  (skipping AWQ mirror — not Sunday or already synced this week)"
fi

# Lab code repo bundle (single git bundle — fully restorable)
if [[ -d /data/lab/code/.git ]]; then
    echo "  git bundle lab/code..."
    js_log "git bundle lab/code"
    git -C /data/lab/code bundle create "${DAY_DIR}/lab-code.bundle" --all 2>&1 | tail -1
fi

# Manifest
{
    echo "backup_timestamp=${TIMESTAMP}"
    echo "hostname=$(hostname)"
    echo "pg_version=$(psql -V | head -1)"
    echo "lab_git_sha=$(git -C /data/lab/code rev-parse HEAD 2>/dev/null || echo unknown)"
    du -sh "${DAY_DIR}"/* 2>/dev/null | awk '{print "size_"NR"="$0}'
} > "${DAY_DIR}/MANIFEST.txt"

# Rotation: keep last 14 daily, promote weekly Sunday
find "${BACKUP_ROOT}/daily" -mindepth 1 -maxdepth 1 -type d -mtime +14 -print0 |
    xargs -0r -n1 echo "  pruning:" && \
find "${BACKUP_ROOT}/daily" -mindepth 1 -maxdepth 1 -type d -mtime +14 -exec rm -rf {} +

echo "[$(date)] backup done — $(du -sh "${DAY_DIR}" | awk '{print $1}')"
