#!/usr/bin/env bash
# Lab SessionStart hook — fires only for ~/lab/ sessions.
#
# Installed at the user's Claude Code settings via ~/.claude/settings.json
# under hooks.SessionStart. Lives in the lab repo so it tracks lab-side
# changes via git. See docs/runbooks/lab-session-start.md for install
# instructions and the surrounding rationale.
#
# Output contract: ≤ 10 lines on stdout, all `[lab] ...` lines. The
# Claude Code session start surfaces these in the transcript. Stays silent
# unless cwd is under ~/lab/ (or its /data/lab/code symlink target).
#
# Dependencies: psql, redis-cli, curl. Each section degrades gracefully
# if its dependency is missing or its service is down.

set -uo pipefail

# -- gate on cwd --------------------------------------------------------------
case "$PWD" in
    "$HOME/lab"|"$HOME/lab"/*|/data/lab|/data/lab/*) ;;
    *) exit 0 ;;
esac

LAB_DSN="${LAB_PG_DSN:-postgresql://m@/lab}"
RERANK_URL="${LAB_RERANK_URL:-http://127.0.0.1:8401}"
SWEEP_PIDS_DIR="${LAB_SWEEP_PIDS_DIR:-/data/lab/services/sweep-pids}"
REDIS_URL="${LAB_REDIS_URL:-redis://localhost:6379/0}"
TIMEOUT="${LAB_HOOK_TIMEOUT:-1}"

# -- 1. Most recent running experiment ---------------------------------------
active_exp="$(timeout "$TIMEOUT" psql -tAX -d lab -c \
    "SELECT slug FROM experiments WHERE status='running' \
     ORDER BY started_at DESC NULLS LAST LIMIT 1;" 2>/dev/null || true)"
if [[ -n "$active_exp" ]]; then
    echo "[lab] active experiment: $active_exp"
else
    echo "[lab] active experiment: (none running)"
fi

# -- 2. Last 3 findings (importance-sorted) ----------------------------------
# Schema (per /data/lab/code/packages/lab-core/src/lab/migrations) has no importance column —
# fall back to most-recent finding_id. Future schema additions: ORDER BY
# importance DESC, finding_id DESC.
findings="$(timeout "$TIMEOUT" psql -tAX -F'|' -d lab -c \
    "SELECT slug, confidence FROM findings \
     ORDER BY finding_id DESC LIMIT 3;" 2>/dev/null || true)"
if [[ -n "$findings" ]]; then
    summary="$(echo "$findings" | tr '\n' ' ' | sed 's/  */ /g' | sed 's/ $//')"
    echo "[lab] last 3 findings: $summary"
else
    echo "[lab] last 3 findings: (db unreachable or empty)"
fi

# -- 3. Running sweep PIDs (from pidfile dir) --------------------------------
if [[ -d "$SWEEP_PIDS_DIR" ]]; then
    pidfiles=("$SWEEP_PIDS_DIR"/*.pid)
    if [[ -e "${pidfiles[0]}" ]]; then
        names=()
        for p in "${pidfiles[@]}"; do
            names+=("$(basename "$p" .pid)")
        done
        echo "[lab] sweeps running: ${names[*]}"
    else
        echo "[lab] sweeps running: (none)"
    fi
else
    echo "[lab] sweeps running: (pidfile dir missing)"
fi

# -- 4. GPU lease state ------------------------------------------------------
lease="$(timeout "$TIMEOUT" redis-cli -u "$REDIS_URL" get "lab:gpu:lease:0" 2>/dev/null || true)"
if [[ -n "$lease" && "$lease" != "(nil)" ]]; then
    ttl="$(timeout "$TIMEOUT" redis-cli -u "$REDIS_URL" ttl "lab:gpu:lease:0" 2>/dev/null || echo "?")"
    echo "[lab] gpu lease: $lease (ttl ${ttl}s)"
else
    echo "[lab] gpu lease: (free)"
fi

# -- 5. Rerank service health ------------------------------------------------
if curl -sf --max-time "$TIMEOUT" "$RERANK_URL/healthz" >/dev/null 2>&1; then
    echo "[lab] rerank service: up ($RERANK_URL)"
else
    echo "[lab] rerank service: down or unreachable"
fi

# -- 6. doc-graph (Phase 14.4) -----------------------------------------------
# Three-line summary derived directly from the SQLite DB so the hook stays
# well under the 200ms budget (no python/uv cold-start). Falls back to a
# single helpful line if the DB doesn't exist yet.

DOCS_DB="${LAB_DOCS_DB:-$HOME/db/m/docs.db}"
if [[ ! -f "$DOCS_DB" ]]; then
    echo "[lab] doc-graph: not initialised — run \`m docs scan\`"
else
    status_line="$(timeout "$TIMEOUT" sqlite3 "$DOCS_DB" \
        "SELECT 'active: ' || COALESCE(SUM(status='active'),0) || \
                ', draft: '  || COALESCE(SUM(status='draft'),0) || \
                ', archived: ' || COALESCE(SUM(status='archived'),0) \
         FROM docs;" 2>/dev/null || true)"
    if [[ -n "$status_line" ]]; then
        echo "[lab] doc-graph status: $status_line"
    else
        echo "[lab] doc-graph status: (db unreachable)"
    fi

    gap_count="$(timeout "$TIMEOUT" sqlite3 "$DOCS_DB" \
        "SELECT COUNT(*) FROM docs \
         WHERE last_verified IS NULL \
            OR last_verified < date('now', '-30 days');" 2>/dev/null || echo "?")"
    echo "[lab] doc-graph gap (>30d): ${gap_count:-?}"

    orphan_count="$(timeout "$TIMEOUT" sqlite3 "$DOCS_DB" \
        "SELECT COUNT(*) FROM docs d \
         WHERE NOT EXISTS (SELECT 1 FROM doc_deps WHERE doc_id = d.doc_id) \
           AND NOT EXISTS (SELECT 1 FROM doc_deps WHERE dep_kind='doc' AND dep_target = d.doc_id);" \
        2>/dev/null || echo "?")"
    echo "[lab] doc-graph orphans: ${orphan_count:-?}"
fi
