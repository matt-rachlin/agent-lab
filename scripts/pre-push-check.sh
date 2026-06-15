#!/usr/bin/env bash
# pre-push-check.sh — G2 gate: runs as a pre-push hook via pre-commit.
# Exits 1 with a clear message if the push should be blocked.
set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

BLOCKED=0
REASONS=()

# 1. Untracked .md files in strategic doc directories
untracked_docs=$(git ls-files --others --exclude-standard \
    docs/adr/ docs/exp/ docs/findings/ docs/log/ docs/writeups/ docs/protocols/ \
    2>/dev/null | grep '\.md$' || true)
if [[ -n "$untracked_docs" ]]; then
    BLOCKED=1
    count=$(echo "$untracked_docs" | wc -l)
    REASONS+=("  untracked strategic .md files ($count) — stage or add to .gitignore:")
    while IFS= read -r f; do REASONS+=("    $f"); done <<< "$untracked_docs"
fi

# 2. Untracked .yaml files under conf/sweep/
untracked_sweep=$(git ls-files --others --exclude-standard conf/sweep/ \
    2>/dev/null | grep '\.yaml$' || true)
if [[ -n "$untracked_sweep" ]]; then
    BLOCKED=1
    count=$(echo "$untracked_sweep" | wc -l)
    REASONS+=("  untracked sweep configs ($count) — stage or add to .gitignore:")
    while IFS= read -r f; do REASONS+=("    $f"); done <<< "$untracked_sweep"
fi

# 3. Tracked EXP docs containing the placeholder SHA
placeholder_hits=$(git ls-files docs/exp/ | xargs grep -l \
    '<commit SHA filled by lab exp register at registration time>' 2>/dev/null || true)
if [[ -n "$placeholder_hits" ]]; then
    BLOCKED=1
    count=$(echo "$placeholder_hits" | wc -l)
    REASONS+=("  EXP docs with placeholder SHA ($count) — run 'lab exp register <slug>':")
    while IFS= read -r f; do REASONS+=("    $f"); done <<< "$placeholder_hits"
fi

# 4. Any tracked *.bak* files (defense in depth alongside commit-time gate)
tracked_baks=$(git ls-files | grep '\.bak' || true)
if [[ -n "$tracked_baks" ]]; then
    BLOCKED=1
    count=$(echo "$tracked_baks" | wc -l)
    REASONS+=("  tracked .bak* files ($count) — remove from index with 'git rm --cached':")
    while IFS= read -r f; do REASONS+=("    $f"); done <<< "$tracked_baks"
fi

if [[ "$BLOCKED" -eq 1 ]]; then
    echo ""
    echo "pre-push BLOCKED because:"
    for r in "${REASONS[@]}"; do
        echo "$r"
    done
    echo ""
    echo "Fix the above and retry. (Skip: SKIP=pre-push-check git push)"
    exit 1
fi

exit 0
