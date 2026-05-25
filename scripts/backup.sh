#!/usr/bin/env bash
# Nightly backup: lab DBs + MinIO bucket sync → /mnt/backup/lab/
# Idempotent: keeps the last 14 daily snapshots, rotates older to weekly.
# Designed to run from cron OR `just backup`.

set -euo pipefail

BACKUP_ROOT="${LAB_BACKUP_ROOT:-/mnt/backup/lab}"
TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
DAY_DIR="${BACKUP_ROOT}/daily/${TIMESTAMP}"

mkdir -p "${DAY_DIR}"

echo "[$(date)] backup → ${DAY_DIR}"

# Postgres dumps (3 lab-related DBs)
for db in lab mlflow litellm; do
    echo "  pg_dump ${db}..."
    pg_dump -Fc "${db}" > "${DAY_DIR}/${db}.dump"
    gzip "${DAY_DIR}/${db}.dump"
done

# MinIO buckets via mc (uses host secret)
SECRET_FILE="/data/lab/services/minio-secret"
if [[ -f "${SECRET_FILE}" ]]; then
    SECRET="$(cat "${SECRET_FILE}")"
    echo "  mc mirror lab/..."
    podman run --rm --network host --entrypoint /bin/sh \
        -v "${DAY_DIR}:/backup:Z" \
        -e MC_HOST_lab="http://labadmin:${SECRET}@localhost:9000" \
        docker.io/minio/mc:latest \
        -c 'mc mirror --quiet --overwrite lab/lab /backup/minio-lab && mc mirror --quiet --overwrite lab/mlflow /backup/minio-mlflow'
else
    echo "  (skipping MinIO sync — no secret file)"
fi

# Lab code repo bundle (single git bundle — fully restorable)
if [[ -d /data/lab/code/.git ]]; then
    echo "  git bundle lab/code..."
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
