-- lab — migration 002: agent harness (Phase 6).
-- Adds per-run agent metrics to experiment_runs and a new agent_logs table
-- holding the Inspect-side log pointer + per-turn instrumentation.

ALTER TABLE experiment_runs ADD COLUMN IF NOT EXISTS actual_turns INT;
ALTER TABLE experiment_runs ADD COLUMN IF NOT EXISTS tool_call_count INT;
ALTER TABLE experiment_runs ADD COLUMN IF NOT EXISTS sandbox_image_hash TEXT;

CREATE TABLE IF NOT EXISTS agent_logs (
    run_id            TEXT PRIMARY KEY REFERENCES experiment_runs(run_id),
    inspect_log_path  TEXT,
    turns             JSONB,
    inserted_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO schema_migrations (version) VALUES ('002_agent') ON CONFLICT DO NOTHING;
