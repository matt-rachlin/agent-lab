#!/usr/bin/env bash
# roll_daily_log.sh — auto-commit today's daily log if it has real content.
# Triggered by lab-daily-log-roll.timer at 23:50 local time.
set -euo pipefail

cd /data/lab/code

date_str=$(date +%Y-%m-%d)
f="docs/log/$date_str.md"

# Nothing to do if the file doesn't exist yet.
if [[ ! -f "$f" ]]; then
    echo "roll_daily_log: $f does not exist, nothing to commit."
    exit 0
fi

wc=$(wc -l < "$f")

# Untouched template has 6 lines or fewer — skip.
if [[ "$wc" -lt 6 ]]; then
    echo "roll_daily_log: $f has only $wc lines (looks like an empty template), skipping."
    exit 0
fi

git add "$f"

# Check if there's actually a staged change to commit (idempotent on re-runs).
if git diff --cached --quiet; then
    echo "roll_daily_log: $f already committed or unchanged, nothing to do."
    exit 0
fi

git -c user.name="lab-daily-log-roll" \
    -c user.email="lab@localhost" \
    commit -m "log: $date_str (auto-roll)"

echo "roll_daily_log: committed $f"
