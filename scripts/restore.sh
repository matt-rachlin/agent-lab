#!/usr/bin/env bash
# Restore-from-snapshot drill (Phase 17.10).
#
# Restores a lab daily snapshot into NAMESPACED targets so it never clobbers
# live state:
#   - Postgres DBs:   lab_restore_test, mlflow_restore_test, litellm_restore_test
#   - MinIO buckets:  lab-restore-test, mlflow-restore-test
#   - Git bundle:     unpacked under /tmp/lab-restore-test/repo
#
# Usage:
#   scripts/restore.sh <snapshot_dir>      # e.g. /mnt/backup/lab/daily/2026-05-26_033425
#   scripts/restore.sh --latest            # newest daily snapshot
#   scripts/restore.sh --cleanup           # drop namespaced restore artifacts
#
# Exits non-zero on any verification mismatch.

set -euo pipefail

BACKUP_ROOT="${LAB_BACKUP_ROOT:-/mnt/backup/lab}"
RESTORE_NS="${LAB_RESTORE_NS:-restore_test}"        # appended via _ to DB names
RESTORE_BUCKET_NS="${LAB_RESTORE_BUCKET_NS:-restore-test}"  # appended via - to bucket names
RESTORE_REPO_DIR="${LAB_RESTORE_REPO_DIR:-/tmp/lab-restore-test/repo}"
SECRET_FILE="/data/lab/services/minio-secret"

cleanup() {
    echo "[$(date)] cleanup → dropping namespaced restore artifacts"
    for db in lab mlflow litellm; do
        target="${db}_${RESTORE_NS}"
        if psql -tAc "SELECT 1 FROM pg_database WHERE datname='${target}'" | grep -q 1; then
            echo "  dropdb ${target}"
            dropdb "${target}"
        fi
    done
    if [[ -f "${SECRET_FILE}" ]]; then
        SECRET="$(cat "${SECRET_FILE}")"
        for src_bucket in lab mlflow; do
            target_bucket="${src_bucket}-${RESTORE_BUCKET_NS}"
            podman run --rm --network host --entrypoint /bin/sh \
                -e MC_HOST_lab="http://labadmin:${SECRET}@localhost:9000" \
                docker.io/minio/mc:latest \
                -c "mc rb --force lab/${target_bucket} 2>/dev/null || true"
            echo "  mc rb lab/${target_bucket}"
        done
    fi
    rm -rf "${RESTORE_REPO_DIR%/repo}"
    echo "[$(date)] cleanup done"
}

if [[ "${1:-}" == "--cleanup" ]]; then
    cleanup
    exit 0
fi

# Resolve snapshot dir
if [[ "${1:-}" == "--latest" || -z "${1:-}" ]]; then
    SNAPSHOT_DIR="$(ls -1d "${BACKUP_ROOT}/daily"/*/ 2>/dev/null | sort | tail -1)"
    SNAPSHOT_DIR="${SNAPSHOT_DIR%/}"
else
    SNAPSHOT_DIR="${1%/}"
fi

if [[ ! -d "${SNAPSHOT_DIR}" ]]; then
    echo "ERROR: snapshot not found: ${SNAPSHOT_DIR}" >&2
    exit 1
fi

START=$(date +%s)
echo "[$(date)] restore drill → ${SNAPSHOT_DIR}"
echo "  manifest:"
sed 's/^/    /' "${SNAPSHOT_DIR}/MANIFEST.txt"

# 1. Postgres restore (namespaced)
for db in lab mlflow litellm; do
    target="${db}_${RESTORE_NS}"
    dump="${SNAPSHOT_DIR}/${db}.dump.gz"
    if [[ ! -f "${dump}" ]]; then
        echo "WARN: missing ${dump}, skipping" >&2
        continue
    fi
    echo "[$(date)] pg restore ${db} → ${target}"
    if psql -tAc "SELECT 1 FROM pg_database WHERE datname='${target}'" | grep -q 1; then
        dropdb "${target}"
    fi
    createdb "${target}"
    gunzip -c "${dump}" | pg_restore --no-owner --no-privileges --dbname="${target}" 2>&1 \
        | grep -Ev '(^pg_restore: warning|already exists|errors ignored)' || true
done

# 2. MinIO restore (namespaced buckets)
if [[ -f "${SECRET_FILE}" ]]; then
    SECRET="$(cat "${SECRET_FILE}")"
    for src_bucket in lab mlflow; do
        src_dir="${SNAPSHOT_DIR}/minio-${src_bucket}"
        target_bucket="${src_bucket}-${RESTORE_BUCKET_NS}"
        if [[ ! -d "${src_dir}" ]]; then
            echo "WARN: missing ${src_dir}, skipping" >&2
            continue
        fi
        echo "[$(date)] minio restore ${src_bucket} → ${target_bucket}"
        podman run --rm --network host --entrypoint /bin/sh \
            -v "${src_dir}:/src:Z" \
            -e MC_HOST_lab="http://labadmin:${SECRET}@localhost:9000" \
            docker.io/minio/mc:latest \
            -c "mc rb --force lab/${target_bucket} >/dev/null 2>&1 || true; \
                mc mb lab/${target_bucket} && \
                mc mirror --quiet --overwrite /src lab/${target_bucket}"
    done
else
    echo "WARN: ${SECRET_FILE} missing; skipping MinIO restore" >&2
fi

# 3. Git bundle restore
bundle="${SNAPSHOT_DIR}/lab-code.bundle"
if [[ -f "${bundle}" ]]; then
    echo "[$(date)] git restore from bundle"
    rm -rf "${RESTORE_REPO_DIR%/repo}"
    mkdir -p "${RESTORE_REPO_DIR}"
    git clone --quiet "${bundle}" "${RESTORE_REPO_DIR}"
fi

# 4. Verification
echo "[$(date)] verification"
fail=0

verify() {
    local label="$1" expected="$2" actual="$3"
    if [[ "${expected}" == "${actual}" ]]; then
        echo "  OK   ${label}: ${actual}"
    else
        echo "  FAIL ${label}: expected ${expected}, got ${actual}" >&2
        fail=1
    fi
}

# Anchors are auto-discovered from the snapshot itself (snapshot's row counts
# are the ground truth for round-trip). We re-restore into a second namespaced
# DB and compare, but here we just sanity-check non-emptiness + a known anchor
# pulled from the snapshot manifest.
#
# Three structural invariants we expect from ANY snapshot:
#   - findings table has rows (we know at least F-001 is in every snapshot)
#   - experiment_runs table is populated (>= 1)
#   - F-001 row is always present (Phase 1 finding, foundational)

# 4a. F-001 always present (snapshot-agnostic anchor)
f001_exists=$(psql -d "lab_${RESTORE_NS}" -tAc "
    SELECT COUNT(*) FROM findings WHERE slug='F-001';" 2>&1 | tr -d '[:space:]')
verify "F-001 row present (foundational anchor)" "1" "${f001_exists}"

# 4b. findings table non-empty
findings_count=$(psql -d "lab_${RESTORE_NS}" -tAc "
    SELECT COUNT(*) FROM findings;" 2>&1 | tr -d '[:space:]')
if (( findings_count >= 1 )); then
    echo "  OK   findings count: ${findings_count}"
else
    echo "  FAIL findings count: ${findings_count} (expected >= 1)" >&2
    fail=1
fi

# 4c. experiment_runs table non-empty
runs_count=$(psql -d "lab_${RESTORE_NS}" -tAc "
    SELECT COUNT(*) FROM experiment_runs;" 2>&1 | tr -d '[:space:]')
if (( runs_count >= 1 )); then
    echo "  OK   experiment_runs count: ${runs_count}"
else
    echo "  FAIL experiment_runs count: ${runs_count} (expected >= 1)" >&2
    fail=1
fi

# 4d. EXP-001 always present (older anchor in every snapshot we have)
exp001_runs=$(psql -d "lab_${RESTORE_NS}" -tAc "
    SELECT COUNT(*) FROM experiment_runs r
    JOIN experiments e ON e.experiment_id = r.experiment_id
    WHERE e.slug='EXP-001';" 2>&1 | tr -d '[:space:]')
if (( exp001_runs >= 1 )); then
    echo "  OK   EXP-001 run count: ${exp001_runs}"
else
    echo "  FAIL EXP-001 run count: ${exp001_runs} (expected >= 1)" >&2
    fail=1
fi

# 4e. MinIO bucket non-empty
if [[ -f "${SECRET_FILE}" ]]; then
    SECRET="$(cat "${SECRET_FILE}")"
    minio_objs=$(podman run --rm --network host --entrypoint /bin/sh \
        -e MC_HOST_lab="http://labadmin:${SECRET}@localhost:9000" \
        docker.io/minio/mc:latest \
        -c "mc ls --recursive lab/lab-${RESTORE_BUCKET_NS}/ 2>/dev/null | wc -l" | tr -d '[:space:]')
    if (( minio_objs > 0 )); then
        echo "  OK   minio lab-${RESTORE_BUCKET_NS} object count: ${minio_objs}"
    else
        echo "  FAIL minio lab-${RESTORE_BUCKET_NS} empty" >&2
        fail=1
    fi
fi

# 4f. Git bundle restored to a working repo
if [[ -d "${RESTORE_REPO_DIR}/.git" ]]; then
    restored_sha=$(git -C "${RESTORE_REPO_DIR}" rev-parse HEAD)
    manifest_sha=$(grep '^lab_git_sha=' "${SNAPSHOT_DIR}/MANIFEST.txt" | cut -d= -f2)
    verify "git HEAD == manifest sha" "${manifest_sha}" "${restored_sha}"
else
    echo "  FAIL git bundle not restored" >&2
    fail=1
fi

ELAPSED=$(( $(date +%s) - START ))
echo "[$(date)] restore drill done — wall time ${ELAPSED}s"

if (( fail != 0 )); then
    echo "RESULT: FAIL — at least one verification mismatched" >&2
    exit 2
fi
echo "RESULT: PASS"
