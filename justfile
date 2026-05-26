# lab — task runner. `just` to list, `just <recipe>` to run.

set dotenv-load := true
set positional-arguments := true

default:
    @just --list --unsorted

# --- bootstrap / setup ---

# Install all dependencies (creates .venv)
bootstrap:
    uv sync --all-extras

# Create the Postgres `lab` database
db-create:
    createdb lab || echo "lab DB already exists"

# Apply all SQL migrations
db-migrate:
    @for f in src/lab/migrations/*.sql; do echo "applying $f"; psql -d lab -f "$f"; done

# Full DB init: create + migrate
db-init: db-create db-migrate

# Drop and recreate the DB (DANGEROUS)
db-reset:
    dropdb lab && createdb lab && just db-migrate

# --- services (Podman containers) ---

# Bring up MinIO + MLflow + LiteLLM proxy
services-up:
    podman-compose -f services/compose.yml up -d

services-down:
    podman-compose -f services/compose.yml down

services-logs:
    podman-compose -f services/compose.yml logs -f

services-status:
    podman ps --filter "label=app=lab"

# --- agent sandbox (Phase 6b) ---

# Build the per-cell agent sandbox image (Fedora minimal + python3.13 + uv +
# ripgrep, runs as uid 10001 in /workspace). Writes the image digest to
# conf/sandbox-image.sha so experiment_runs can record the hash it executed
# against.
sandbox-build:
    uv run lab agent sandbox build

# Smoke-test the sandbox under gVisor; prints "ok" or fails loudly.
sandbox-smoke:
    podman run --rm --runtime=runsc --security-opt label=disable --runtime-flag=ignore-cgroups \
        --network=none lab-agent-sandbox:0.1 python3 -c 'print("ok")'

# --- rerank server (Phase 7.1) ---

# Install + enable the host-side rerank server as a systemd user unit.
# Copies services/rerank.service to ~/.config/systemd/user/ so
# `systemctl --user` can manage it; then daemon-reload + enable + start.
rerank-install:
    install -D -m 0644 services/rerank.service \
        ~/.config/systemd/user/rerank.service
    systemctl --user daemon-reload
    systemctl --user enable --now rerank.service
    @echo "rerank.service installed; healthz:"
    @sleep 1 && curl -sS http://127.0.0.1:8401/healthz || \
        echo "(not yet responsive — check 'just rerank-status')"

# Stop + disable + remove the systemd unit (leaves the source in repo).
rerank-uninstall:
    -systemctl --user disable --now rerank.service
    -rm -f ~/.config/systemd/user/rerank.service
    systemctl --user daemon-reload

# Show service status + recent journal.
rerank-status:
    @systemctl --user status rerank.service --no-pager || true
    @echo "--- last 20 log lines ---"
    @journalctl --user -u rerank.service -n 20 --no-pager || true

# Restart after a code change (no need to reinstall unless service file changed).
rerank-restart:
    systemctl --user restart rerank.service
    @sleep 1 && curl -sS http://127.0.0.1:8401/healthz

# --- backup ---

# Nightly snapshot: 3 PG dumps + 2 MinIO bucket mirrors + git bundle, to /mnt/backup/lab
backup:
    bash scripts/backup.sh

# What's in the backup tree?
backup-status:
    @echo "=== latest 5 daily snapshots ==="
    @ls -lat /mnt/backup/lab/daily/ 2>/dev/null | head -6
    @echo "=== total backup size ==="
    @du -sh /mnt/backup/lab 2>/dev/null || echo "(no backups yet)"

# --- model management ---

# Pull a curated initial set of local Ollama models
models-pull:
    @for m in qwen3:14b-q4_K_M qwen3:8b llama3.1:8b-instruct-q4_K_M phi4:latest gemma3:12b-it-q4_K_M; do \
        echo "pulling $m"; ollama pull "$m" || true; \
    done

# Register pulled models (local + cloud) into the lab.models table
models-register:
    uv run python -m lab.models.register

# --- sanity tests ---

# Test that we can write a manifest and store it
manifest-test:
    uv run python -m lab.manifest --test

# Test the Valkey-backed GPU lease
gpu-lease-test:
    uv run python -m lab.gpu_lease --test

# Test Ollama Cloud auth
cloud-test:
    uv run python -m lab.models.cloud_test

# --- quality gates ---

lint:
    uv run ruff check .

fmt:
    uv run ruff format .

fmt-check:
    uv run ruff format --check .

types:
    uv run mypy src

test:
    uv run pytest tests/unit -q

test-integration:
    uv run pytest tests/integration -q -m integration

test-int:
    uv run pytest tests/ -q -m integration

check: lint fmt-check types test
    @echo "all clean"

# --- docs ---

docs-serve:
    uv run mkdocs serve

docs-build:
    uv run mkdocs build

# --- experiment ergonomics ---

today:
    @date_str=$(date +%Y-%m-%d); \
    f="docs/log/$date_str.md"; \
    if [ ! -f "$f" ]; then \
        echo "# $date_str\n\n## Intent today\n\n## Did\n\n## Stuck on / questions\n\n## Notes\n\n## Tomorrow\n" > "$f"; \
        echo "created $f"; \
    fi; \
    ${EDITOR:-vi} "$f"

# Scaffold a new experiment plan: just exp 042 "qwen quant sweep"
exp NUM TITLE:
    @slug=$(echo "{{TITLE}}" | tr ' ' '-' | tr '[:upper:]' '[:lower:]'); \
    f="docs/exp/EXP-{{NUM}}-$slug.md"; \
    cp docs/_templates/experiment.md "$f"; \
    echo "created $f"

# Scaffold a new ADR
adr NUM TITLE:
    @slug=$(echo "{{TITLE}}" | tr ' ' '-' | tr '[:upper:]' '[:lower:]'); \
    f="docs/adr/ADR-{{NUM}}-$slug.md"; \
    cp docs/_templates/adr.md "$f"; \
    echo "created $f"
