import os
import asyncio
import logging
from typing import Any, Dict
import asyncpg

logger = logging.getLogger(__name__)

# Global connection pool (asyncpg)
_pool: asyncpg.Pool | None = None

async def init_pool() -> None:
    """Initialise the asyncpg connection pool.
    Reads DSN from TIMESCALEDB_DSN env var, e.g.:
    postgres://user:password@localhost:5432/yourdb
    """
    global _pool
    if _pool is not None:
        return
    dsn = os.getenv("TIMESCALEDB_DSN")
    if not dsn:
        logger.error("TIMESCALEDB_DSN not set; TimescaleDB disabled.")
        return
    try:
        _pool = await asyncpg.create_pool(dsn, min_size=1, max_size=10)
        logger.info("TimescaleDB connection pool initialised.")
    except Exception as e:
        logger.error(f"Failed to create TimescaleDB pool: {e}")
        _pool = None

async def close_pool() -> None:
    """Close the connection pool on shutdown."""
    global _pool
    if _pool is None:
        return
    await _pool.close()
    logger.info("TimescaleDB pool closed.")
    _pool = None

async def _execute(query: str, *args: Any) -> None:
    """Execute a query using the pool.
    Helper that logs errors.
    """
    if _pool is None:
        logger.warning("TimescaleDB pool not initialized; query ignored.")
        return
    async with _pool.acquire() as conn:
        try:
            await conn.execute(query, *args)
        except Exception as e:
            logger.error(f"TimescaleDB query failed: {e}")

async def insert_audit_log(action: str, entity: str | None = None,
                           entity_id: str | None = None, payload: Dict | None = None,
                           user_id: str = "system", ip: str | None = None) -> None:
    """Insert an audit log entry into TimescaleDB.
    Table schema (simplified):
    CREATE TABLE audit_logs (
        id SERIAL PRIMARY KEY,
        created_at TIMESTAMPTZ DEFAULT now(),
        user_id TEXT,
        action TEXT NOT NULL,
        entity TEXT,
        entity_id TEXT,
        payload JSONB,
        ip_address TEXT
    );
    """
    query = """
        INSERT INTO audit_logs (user_id, action, entity, entity_id, payload, ip_address)
        VALUES ($1, $2, $3, $4, $5, $6)
    """
    await _execute(query, user_id, action, entity, entity_id, payload or {}, ip)

async def insert_prediction(record: Dict) -> None:
    """Insert a prediction record.
    Expected keys: asset_tag, machine_type, prediction, probability, features, model_version (optional).
    Table schema (simplified):
    CREATE TABLE predictions (
        id SERIAL PRIMARY KEY,
        created_at TIMESTAMPTZ DEFAULT now(),
        asset_tag TEXT,
        machine_type TEXT,
        prediction INT,
        probability FLOAT,
        features JSONB,
        model_version TEXT DEFAULT '1.0'
    );
    """
    query = """
        INSERT INTO predictions (asset_tag, machine_type, prediction, probability, features, model_version)
        VALUES ($1, $2, $3, $4, $5, $6)
    """
    await _execute(query,
                    record.get("asset_tag"),
                    record.get("machine_type"),
                    record.get("prediction"),
                    record.get("probability"),
                    record.get("features"),
                    record.get("model_version", "1.0"))
