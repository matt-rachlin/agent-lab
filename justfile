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

# --- model management ---

# Pull a curated initial set of local Ollama models
models-pull:
    @for m in qwen3:14b-q4_K_M qwen3:8b-q5_K_M llama3.1:8b-instruct-q4_K_M phi-4:14b-q4_K_M gemma3:12b-it-q4_K_M; do \
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
    uv run pytest tests/ -q

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
