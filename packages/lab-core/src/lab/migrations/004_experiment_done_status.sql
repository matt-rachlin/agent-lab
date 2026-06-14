-- lab — migration 004: add 'done' to the experiment lifecycle.
--
-- The lifecycle was planned -> running -> analyzed -> written_up (+ abandoned),
-- with no stage for "finished executing but not yet analyzed". The sweep runner
-- left finished experiments stuck at 'running' forever (it stamped 'running' at
-- the END of run_sweep and never wrote a terminal state). 'done' fills that gap:
-- run_sweep now sets status='running'+started_at at start and 'done'+completed_at
-- on exit. Nothing filters experiments by this field (analyze/eval key on
-- experiment_runs.status), so widening the check is non-breaking.

ALTER TABLE experiments DROP CONSTRAINT IF EXISTS experiments_status_check;
ALTER TABLE experiments ADD CONSTRAINT experiments_status_check
    CHECK (status = ANY (ARRAY[
        'planned'::text, 'running'::text, 'done'::text,
        'analyzed'::text, 'written_up'::text, 'abandoned'::text
    ]));
