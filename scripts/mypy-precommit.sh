#!/usr/bin/env bash
# Phase 15.1: lab is a PEP 420 namespace split across src/lab and
# packages/lab-*/src/lab. mypy needs every populated source root passed
# explicitly so it can resolve cross-package imports without bouncing
# off the namespace duplicates that would result from passing the whole
# repo to a single mypy invocation.
#
# During the 15.1.2 - 15.1.9 file-move sequence some packages contain
# only a placeholder `__init__.py` (which would otherwise collide with
# the still-in-src/ real module). We pass `--exclude` for those until
# the corresponding move lands.
set -euo pipefail

cd "$(dirname "$0")/.."

# Skeleton-with-placeholder packages that haven't been populated yet.
# Each entry is the path of a placeholder __init__.py that would
# collide with the same module still living under src/lab/.
SKELETON_INITS=(
    packages/lab-inspect/src/lab/inspect_bridge/__init__.py
    packages/lab-sweep/src/lab/sweep/__init__.py
    packages/lab-observability/src/lab/observability/__init__.py
)
EXCLUDE_PATTERNS=()
for f in "${SKELETON_INITS[@]}"; do
    # If src/lab/<name>/ still has a real __init__.py, the placeholder is
    # the duplicate and must be excluded from mypy's discovery.
    name="$(dirname "$f" | sed 's|.*/src/lab/||')"
    if [[ -f "src/lab/$name/__init__.py" ]]; then
        EXCLUDE_PATTERNS+=("--exclude" "^$f$")
    fi
done

exec uv run mypy "${EXCLUDE_PATTERNS[@]}" src packages
