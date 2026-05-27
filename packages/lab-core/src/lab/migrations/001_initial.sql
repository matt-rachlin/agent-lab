-- lab — initial schema. Applied to a fresh `lab` Postgres DB.

CREATE EXTENSION IF NOT EXISTS pgcrypto;        -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS vector;          -- pgvector

-- ------------------------------------------------------------
-- Models (local + cloud, all routed through LiteLLM)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS models (
    model_id        SERIAL PRIMARY KEY,
    publisher       TEXT NOT NULL,                 -- 'qwen', 'meta', 'microsoft', 'openai', etc.
    name            TEXT NOT NULL,                 -- 'qwen3', 'llama3.1', 'gpt-oss'
    variant         TEXT,                          -- '14b', '8b', '120b'
    quant           TEXT,                          -- 'Q4_K_M', 'Q5_K_M', 'fp16', 'cloud'
    backend         TEXT NOT NULL,                 -- 'ollama-local', 'ollama-cloud', 'vllm', 'llama.cpp'
    litellm_id      TEXT NOT NULL,                 -- the canonical name we use everywhere
    source_url      TEXT,
    source_sha256   TEXT,                          -- digest if known
    ollama_tag      TEXT,                          -- 'qwen3:14b-q4_K_M' or 'gpt-oss:120b-cloud'
    vram_gb         NUMERIC,                       -- NULL for cloud
    context_max     INTEGER,
    output_max      INTEGER,                       -- 16384 for ollama cloud per spec
    license         TEXT,
    capabilities    TEXT[],                        -- ['tool_call', 'vision', 'reasoning', 'json']
    notes           TEXT,
    pulled_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    retired_at      TIMESTAMPTZ,
    UNIQUE (litellm_id)
);

CREATE INDEX IF NOT EXISTS idx_models_backend ON models(backend) WHERE retired_at IS NULL;

-- ------------------------------------------------------------
-- Tasks (one row per eval task)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tasks (
    task_id         SERIAL PRIMARY KEY,
    suite           TEXT NOT NULL,                 -- 'PBS-v0.1', 'BFCL-v3', 'tau-bench-airline'
    external_id     TEXT,                          -- upstream id if borrowed
    slug            TEXT NOT NULL,                 -- short stable identifier
    category        TEXT,                          -- 'tool_call', 'reasoning', 'desktop', 'research_workflow'
    difficulty      TEXT CHECK (difficulty IN ('easy','medium','hard')),
    payload         JSONB NOT NULL,                -- input, gold answer, rubric, tools, etc.
    added_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    retired_at      TIMESTAMPTZ,
    UNIQUE (suite, slug)
);

CREATE INDEX IF NOT EXISTS idx_tasks_suite ON tasks(suite) WHERE retired_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_tasks_category ON tasks(category) WHERE retired_at IS NULL;

-- ------------------------------------------------------------
-- Prompts (versioned, content-addressable)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prompts (
    prompt_id       SERIAL PRIMARY KEY,
    family          TEXT NOT NULL,                 -- 'system-default', 'tool-aware-v2', 'verbose-tool'
    version         TEXT NOT NULL,
    content_sha256  TEXT NOT NULL UNIQUE,
    content         TEXT NOT NULL,
    notes           TEXT,
    registered_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prompts_family ON prompts(family);

-- ------------------------------------------------------------
-- Experiments (planned investigations)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id   SERIAL PRIMARY KEY,
    slug            TEXT NOT NULL UNIQUE,          -- 'EXP-001-twelve-gb-agent'
    title           TEXT NOT NULL,
    hypothesis      TEXT,
    status          TEXT NOT NULL DEFAULT 'planned' CHECK (status IN
        ('planned','running','analyzed','written_up','abandoned')),
    plan_path       TEXT NOT NULL,                 -- 'docs/exp/EXP-001-...md'
    plan_git_sha    TEXT,                          -- the commit at which plan was registered
    pre_registered_at TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);

-- ------------------------------------------------------------
-- Manifests (env capture per run)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS manifests (
    manifest_sha    TEXT PRIMARY KEY,              -- sha256 of the manifest content
    s3_path         TEXT NOT NULL,                 -- where the json lives
    captured_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    git_sha         TEXT NOT NULL,
    git_dirty       BOOLEAN NOT NULL,
    python_version  TEXT NOT NULL,
    deps_sha256     TEXT NOT NULL,                 -- hash of uv pip freeze
    nvidia_driver   TEXT,
    cuda_version    TEXT,
    gpu_name        TEXT,
    payload         JSONB NOT NULL                 -- the full manifest blob for queryability
);

CREATE INDEX IF NOT EXISTS idx_manifests_git_sha ON manifests(git_sha);

-- ------------------------------------------------------------
-- Experiment runs (one per (model, config, task, seed))
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id          TEXT PRIMARY KEY,              -- deterministic hash of (experiment + model + config + task + seed)
    experiment_id   INT REFERENCES experiments(experiment_id),
    model_id        INT NOT NULL REFERENCES models(model_id),
    prompt_id       INT REFERENCES prompts(prompt_id),
    task_id         INT NOT NULL REFERENCES tasks(task_id),
    config_hash     TEXT NOT NULL,
    config          JSONB NOT NULL,
    seed            INT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'queued' CHECK (status IN
        ('queued','running','done','error','killed','skipped')),
    manifest_sha    TEXT REFERENCES manifests(manifest_sha),
    trace_path      TEXT,                          -- 's3://lab/runs/...trace.jsonl'
    tokens_in       INT,
    tokens_out      INT,
    latency_ms      INT,
    cost_usd        NUMERIC(10, 6),
    cost_gpu_sec    NUMERIC(10, 3),                -- ollama cloud bills gpu-sec
    error           TEXT,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_runs_experiment ON experiment_runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_runs_model_task ON experiment_runs(model_id, task_id);
CREATE INDEX IF NOT EXISTS idx_runs_started ON experiment_runs(started_at);
CREATE INDEX IF NOT EXISTS idx_runs_status ON experiment_runs(status);

-- ------------------------------------------------------------
-- Evaluators (versioned)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evaluators (
    evaluator_id    SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    version         TEXT NOT NULL,
    category        TEXT NOT NULL CHECK (category IN ('deterministic','llm_judge','human','external')),
    module_path     TEXT NOT NULL,
    rubric_path     TEXT,                          -- pre-registered rubric file for llm_judge
    judge_model_id  INT REFERENCES models(model_id),
    threshold       NUMERIC,
    registered_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (name, version)
);

-- ------------------------------------------------------------
-- Eval results
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_results (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id          TEXT NOT NULL REFERENCES experiment_runs(run_id),
    evaluator_id    INT NOT NULL REFERENCES evaluators(evaluator_id),
    score           NUMERIC NOT NULL,
    passed          BOOLEAN,
    raw             JSONB,                         -- judge reasoning, sub-scores
    evaluated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, evaluator_id)
);

CREATE INDEX IF NOT EXISTS idx_eval_results_evaluator ON eval_results(evaluator_id, evaluated_at);

-- ------------------------------------------------------------
-- Findings (mirror of docs/findings/)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS findings (
    finding_id      SERIAL PRIMARY KEY,
    slug            TEXT NOT NULL UNIQUE,
    claim           TEXT NOT NULL,
    confidence      TEXT NOT NULL CHECK (confidence IN ('low','medium','high')),
    source_exp      INT REFERENCES experiments(experiment_id),
    doc_path        TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'logged',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    superseded_by   INT REFERENCES findings(finding_id)
);

-- ------------------------------------------------------------
-- Datasets (registry)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id      SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    version         TEXT NOT NULL,
    source_url      TEXT,
    s3_path         TEXT NOT NULL,                 -- MinIO bucket path
    size_bytes      BIGINT,
    rows            BIGINT,
    schema_hash     TEXT,
    datasheet_path  TEXT,                          -- docs/datasets/DS-NNN.md
    license         TEXT,
    added_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (name, version)
);

-- ------------------------------------------------------------
-- Migration tracking
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_migrations (
    version         TEXT PRIMARY KEY,
    applied_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO schema_migrations (version) VALUES ('001_initial') ON CONFLICT DO NOTHING;
