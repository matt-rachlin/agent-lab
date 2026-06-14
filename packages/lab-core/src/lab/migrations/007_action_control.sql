-- 007_action_control.sql
-- Stage 0b #10: action-control substrate (ADR-008 / research-agent-stage0 D5).
-- Additive + idempotent. Creates an UNUSED least-privilege lab_agent role, the
-- control plane (kill switch + budgets), an append-only agent action audit log,
-- and promotion guards so the agent identity can never mint verified/finding.
-- Existing services (user 'm') are unaffected. Container + egress isolation are
-- operational steps handled outside this migration.

BEGIN;

-- control plane: singleton kill switch + budgets
CREATE TABLE IF NOT EXISTS agent_control (
  id                   boolean PRIMARY KEY DEFAULT true CHECK (id),
  killed               boolean NOT NULL DEFAULT false,
  killed_reason        text,
  killed_at            timestamptz,
  daily_usd_budget     numeric(10,2),
  daily_token_budget   bigint,
  daily_gpu_sec_budget bigint,
  updated_at           timestamptz NOT NULL DEFAULT now()
);
INSERT INTO agent_control (id) VALUES (true) ON CONFLICT DO NOTHING;

-- append-only audit of agent ACTIONS (distinct from result trust_transitions)
CREATE TABLE IF NOT EXISTS agent_action_log (
  id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  actor       text NOT NULL,
  action      text NOT NULL,
  args        jsonb,
  approved_by text,
  outcome     text,
  prev_hash   text,
  row_hash    text NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_agent_action_log_created ON agent_action_log(created_at);

-- least-privilege role (NOLOGIN; a login mechanism is wired when the agent ships)
DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'lab_agent') THEN
    CREATE ROLE lab_agent NOLOGIN;
  END IF;
END $$;
GRANT USAGE ON SCHEMA public TO lab_agent;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO lab_agent;
GRANT INSERT ON experiment_runs, eval_results, agent_logs, trust_transitions, agent_action_log TO lab_agent;
GRANT UPDATE ON experiment_runs TO lab_agent;
-- append-only: the agent can never rewrite/erase the audit trail or the chain
REVOKE UPDATE, DELETE ON trust_transitions, agent_action_log FROM lab_agent;

-- promotion guard: lab_agent may propose up to reliability_confirmed, never mint
-- verified/finding (ADR-008 §3). Trusted system (user 'm') is unaffected.
CREATE OR REPLACE FUNCTION _trust_promotion_guard() RETURNS trigger AS $$
BEGIN
  IF current_user = 'lab_agent' AND NEW.to_level IN ('verified', 'finding') THEN
    RAISE EXCEPTION 'lab_agent may not mint % (ADR-008: verified/finding are human/system-minted)', NEW.to_level;
  END IF;
  RETURN NEW;
END $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trust_promotion_guard ON trust_transitions;
CREATE TRIGGER trust_promotion_guard BEFORE INSERT ON trust_transitions
  FOR EACH ROW EXECUTE FUNCTION _trust_promotion_guard();

CREATE OR REPLACE FUNCTION _run_trust_level_guard() RETURNS trigger AS $$
BEGIN
  IF current_user = 'lab_agent' AND NEW.trust_level IN ('verified', 'finding')
     AND NEW.trust_level IS DISTINCT FROM OLD.trust_level THEN
    RAISE EXCEPTION 'lab_agent may not set trust_level=% (ADR-008)', NEW.trust_level;
  END IF;
  RETURN NEW;
END $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS run_trust_level_guard ON experiment_runs;
CREATE TRIGGER run_trust_level_guard BEFORE UPDATE ON experiment_runs
  FOR EACH ROW EXECUTE FUNCTION _run_trust_level_guard();

INSERT INTO schema_migrations (version) VALUES ('007_action_control') ON CONFLICT DO NOTHING;
COMMIT;
