#!/usr/bin/env bash
# Phase 15.1: lab is a PEP 420 namespace assembled from
# packages/lab-*/src/lab. mypy needs every populated source root passed
# explicitly so it can resolve cross-package imports without bouncing
# off the namespace duplicates that would result from passing the whole
# repo to a single mypy invocation.
set -euo pipefail

cd "$(dirname "$0")/.."

exec uv run mypy packages
