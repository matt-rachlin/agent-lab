-- 008_scout_recommendations.sql
-- Research-scout (ADR-010) recommendation queue. Additive. source_url UNIQUE = dedup.
BEGIN;
CREATE TABLE IF NOT EXISTS scout_recommendations (
  id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  source_url   text NOT NULL UNIQUE,
  title        text NOT NULL,
  category     text NOT NULL,   -- model|architecture|software|paper|method|benchmark
  why_relevant text NOT NULL,
  confidence   text NOT NULL DEFAULT 'medium',  -- low|medium|high
  status       text NOT NULL DEFAULT 'new',      -- new|triaged|actioned|rejected
  found_at     timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT scout_confidence_check CHECK (confidence IN ('low','medium','high')),
  CONSTRAINT scout_status_check CHECK (status IN ('new','triaged','actioned','rejected'))
);
CREATE INDEX IF NOT EXISTS idx_scout_status ON scout_recommendations(status);
INSERT INTO schema_migrations (version) VALUES ('008_scout_recommendations') ON CONFLICT DO NOTHING;
COMMIT;
