#!/usr/bin/env bash
# Pre-commit wrapper for `m docs lint`.
#
# Pre-commit passes the staged markdown filenames as positional args.
# We forward each as a `--path <file>` flag to `m docs lint`, which
# parses frontmatter and exits non-zero on missing-required-fields or
# invalid-frontmatter, but tolerates files with no frontmatter at all
# (we're still mid-backfill — see Phase 14.5).
#
# To run on demand: `pre-commit run --hook-stage manual docs-lint`
# Or directly: `m docs lint --path <file> [--path <file2> ...]`

set -uo pipefail

if [[ $# -eq 0 ]]; then
    exit 0
fi

args=()
for f in "$@"; do
    # Resolve to absolute path so the fallback `uv --directory` invocation
    # (which cd's into the m-cli repo) can still find the file.
    if [[ "$f" == /* ]]; then
        args+=("--path" "$f")
    else
        args+=("--path" "$PWD/$f")
    fi
done

# Try `m` from PATH first (installed via pipx / pyproject script); fall
# back to running it from the m-cli repo via uv if not installed or if
# the installed copy predates Phase 14 (no `docs` subcommand yet).
if command -v m >/dev/null 2>&1; then
    if m docs --help >/dev/null 2>&1; then
        exec m docs lint "${args[@]}"
    fi
fi

MCLI_ROOT="${MCLI_ROOT:-$HOME/lab/home-reorg/m-cli}"
if [[ -d "$MCLI_ROOT" ]]; then
    exec uv --directory "$MCLI_ROOT" run m docs lint "${args[@]}"
fi

echo "docs-lint: cannot find \`m docs\` binary or m-cli repo at $MCLI_ROOT" >&2
exit 2
