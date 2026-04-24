-- ============================================================
-- PredMaint Schema
-- Paste into Supabase SQL Editor and click Run
-- ============================================================

-- audit_logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    id          BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id     TEXT NOT NULL DEFAULT 'system',
    action      TEXT NOT NULL,
    entity      TEXT,
    entity_id   TEXT,
    payload     JSONB,
    ip_address  TEXT
);

-- predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id            BIGSERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    asset_tag     TEXT,
    machine_type  TEXT,
    prediction    INTEGER,
    probability   DOUBLE PRECISION,
    features      JSONB,
    model_version TEXT NOT NULL DEFAULT '1.0'
);

-- alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id          BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    asset_tag   TEXT,
    channel     TEXT,
    message     TEXT,
    status      TEXT NOT NULL DEFAULT 'sent'
);

-- Enable Row Level Security
ALTER TABLE audit_logs  ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts      ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY audit_logs_all
    ON audit_logs
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

CREATE POLICY predictions_all
    ON predictions
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

CREATE POLICY alerts_all
    ON alerts
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);
