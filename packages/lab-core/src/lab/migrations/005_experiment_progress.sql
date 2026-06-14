-- lab — migration 005: experiment progress.
--
-- total_cells lets the dashboard show done/total progress instead of just a
-- running count of created rows. Set by run_sweep at start (it computes the
-- matrix size). Experiment-level run-status is never persisted ('running' rows
-- don't exist — see migration 004), so the dashboard derives "running" from
-- total - done - errors; this column is what makes that possible.
ALTER TABLE experiments ADD COLUMN IF NOT EXISTS total_cells INTEGER;
