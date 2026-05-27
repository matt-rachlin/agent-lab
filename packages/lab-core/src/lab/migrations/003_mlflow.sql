-- lab — migration 003: MLflow additive mirror (Phase 15.2).
--
-- Adds id columns so each Postgres row can point at its MLflow mirror.
-- Postgres remains the canonical source of truth; MLflow is additive only.
-- See ~/docs/plans/2026-05-26-lab-master-roadmap.md §15.2.

ALTER TABLE experiment_runs ADD COLUMN IF NOT EXISTS mlflow_run_id TEXT;
ALTER TABLE experiments     ADD COLUMN IF NOT EXISTS mlflow_experiment_id TEXT;
ALTER TABLE findings        ADD COLUMN IF NOT EXISTS mlflow_run_id TEXT;
ALTER TABLE models          ADD COLUMN IF NOT EXISTS mlflow_model_uri TEXT;

CREATE INDEX IF NOT EXISTS idx_runs_mlflow_run_id ON experiment_runs(mlflow_run_id);
CREATE INDEX IF NOT EXISTS idx_experiments_mlflow_id ON experiments(mlflow_experiment_id);

INSERT INTO schema_migrations (version) VALUES ('003_mlflow') ON CONFLICT DO NOTHING;
