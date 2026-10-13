-- backend/migrations.sql
-- TimescaleDB initialization script for edge‑gateway architecture
-- -----------------------------------------------------------
-- 1. Create core tables
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    asset_id VARCHAR(50) NOT NULL,
    machine_type VARCHAR(50),
    telemetry JSONB,               -- raw sensor payload
    health_score REAL,
    ttf_seconds BIGINT,
    shap_values JSONB,
    model_version TEXT DEFAULT '1.0'
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    asset_id VARCHAR(50) NOT NULL,
    user_id TEXT,
    action TEXT NOT NULL,
    entity TEXT,
    entity_id TEXT,
    payload JSONB,
    ip_address TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    asset_id VARCHAR(50) NOT NULL,
    alert_type TEXT NOT NULL,
    severity TEXT CHECK (severity IN ('info','warning','critical')),
    message TEXT,
    metadata JSONB
);

-- -----------------------------------------------------------
-- 2. Convert to TimescaleDB hypertables
-- -----------------------------------------------------------
SELECT create_hypertable('predictions', 'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('audit_logs', 'timestamp', if_not_exists => TRUE);
-- alerts are low‑volume; keep as regular table (optional hypertable)

-- High-frequency sensor events and causal rolling features used by Feast.
CREATE TABLE IF NOT EXISTS sensor_telemetry (
    event_timestamp TIMESTAMPTZ NOT NULL,
    asset_id TEXT NOT NULL,
    plant_id TEXT NOT NULL,
    vibration DOUBLE PRECISION NOT NULL,
    thermal DOUBLE PRECISION NOT NULL,
    acoustic DOUBLE PRECISION NOT NULL,
    vibration_rms DOUBLE PRECISION NOT NULL,
    vibration_kurtosis DOUBLE PRECISION NOT NULL,
    vibration_crest_factor DOUBLE PRECISION NOT NULL,
    thermal_rms DOUBLE PRECISION NOT NULL,
    thermal_kurtosis DOUBLE PRECISION NOT NULL,
    thermal_crest_factor DOUBLE PRECISION NOT NULL,
    acoustic_rms DOUBLE PRECISION NOT NULL,
    acoustic_kurtosis DOUBLE PRECISION NOT NULL,
    acoustic_crest_factor DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (event_timestamp, asset_id)
);
SELECT create_hypertable('sensor_telemetry', 'event_timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_sensor_telemetry_asset_ts
    ON sensor_telemetry (asset_id, event_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sensor_telemetry_plant_ts
    ON sensor_telemetry (plant_id, event_timestamp DESC);

-- -----------------------------------------------------------
-- 3. Indexes for fast lookup
-- -----------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_predictions_asset_ts ON predictions (asset_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_asset_ts ON audit_logs (asset_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_asset_ts ON alerts (asset_id, timestamp DESC);

-- -----------------------------------------------------------
-- 4. Retention & compression policies
-- -----------------------------------------------------------
-- Enable compression for hypertable chunks older than 7 days
ALTER TABLE predictions SET (timescaledb.compress, timescaledb.compress_segmentby = 'asset_id');
SELECT add_compression_policy('predictions', INTERVAL '7 days');
ALTER TABLE audit_logs SET (timescaledb.compress, timescaledb.compress_segmentby = 'asset_id');
SELECT add_compression_policy('audit_logs', INTERVAL '7 days');

-- Retention: drop chunks older than 30 days (adjust as needed)
SELECT add_retention_policy('predictions', INTERVAL '30 days');
SELECT add_retention_policy('audit_logs', INTERVAL '30 days');

-- End of migrations.sql
