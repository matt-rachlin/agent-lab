-- 006_result_trust.sql
-- Stage 0a (ADR-008 / protocols/research-agent-stage0): result-trust substrate.
-- Additive + idempotent. trust_level is a CELL property on experiment_runs
-- (~22% of runs have no eval_results row; some have multiple evaluators), with
-- two gating flags (pre_registered, legacy). Append-only hash-chained transition
-- log; finding<->run link. Append-only enforcement + hash/sig logic land in 0b.

BEGIN;

-- 1. trust_level + gating flags on the cell
ALTER TABLE experiment_runs
  ADD COLUMN IF NOT EXISTS trust_level    text    NOT NULL DEFAULT 'raw',
  ADD COLUMN IF NOT EXISTS pre_registered boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS legacy         boolean NOT NULL DEFAULT false;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='experiment_runs_trust_level_check') THEN
    ALTER TABLE experiment_runs ADD CONSTRAINT experiment_runs_trust_level_check
      CHECK (trust_level IN ('raw','validity_passed','reliability_confirmed',
                             'verification_attempted','verified','finding'));
  END IF;
END $$;
CREATE INDEX IF NOT EXISTS idx_runs_trust_level ON experiment_runs(trust_level);

-- 2. append-only, hash-chained transition log (grants/hash/sig enforced in 0b)
CREATE TABLE IF NOT EXISTS trust_transitions (
  id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  run_id      text NOT NULL REFERENCES experiment_runs(run_id),
  from_level  text,
  to_level    text NOT NULL,
  actor       text NOT NULL,                 -- human id / agent id / 'system'
  is_human    boolean NOT NULL DEFAULT false,
  evidence    jsonb,                         -- gate report / verifier verdict pointer
  reason      text,
  prev_hash   text,                          -- chain: hash of prior row
  row_hash    text NOT NULL,                 -- H(prev_hash || canonical(row))
  signature   text,                          -- Ed25519; required for verified/finding (0b)
  created_at  timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT trust_transitions_to_level_check
    CHECK (to_level IN ('raw','validity_passed','reliability_confirmed',
                        'verification_attempted','verified','finding'))
);
CREATE INDEX IF NOT EXISTS idx_trust_transitions_run     ON trust_transitions(run_id);
CREATE INDEX IF NOT EXISTS idx_trust_transitions_created ON trust_transitions(created_at);

-- 3. finding <-> run link + min trust seen
ALTER TABLE findings
  ADD COLUMN IF NOT EXISTS source_run_id  text REFERENCES experiment_runs(run_id),
  ADD COLUMN IF NOT EXISTS min_trust_seen text;

-- 4. backfill: everything predating the lifecycle is raw + legacy (capped at validity_passed)
UPDATE experiment_runs SET legacy = true WHERE legacy = false;
UPDATE findings        SET min_trust_seen = 'legacy' WHERE min_trust_seen IS NULL;

INSERT INTO schema_migrations (version) VALUES ('006_result_trust') ON CONFLICT DO NOTHING;

COMMIT;
