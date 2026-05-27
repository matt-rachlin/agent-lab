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
# Schema (per /data/lab/code/src/lab/migrations) has no importance column —
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

# -- 6. doc-graph placeholder (Phase 14) -------------------------------------
echo "[lab] doc-graph: Phase 14 pending"
