"""Supabase client + table helpers."""
from supabase import create_client, Client
from backend.config import get_settings
import logging

logger = logging.getLogger(__name__)

_client: Client | None = None


def get_supabase() -> Client:
    global _client
    if _client is None:
        s = get_settings()
        if s.supabase_url and s.supabase_key:
            _client = create_client(s.supabase_url, s.supabase_key)
        else:
            logger.warning("Supabase credentials not set – DB features disabled.")
    return _client


# ── DDL helpers (run once to bootstrap tables) ──────────────────────────────

SCHEMA_SQL = """
create table if not exists audit_logs (
    id          bigserial primary key,
    created_at  timestamptz default now(),
    user_id     text,
    action      text not null,
    entity      text,
    entity_id   text,
    payload     jsonb,
    ip_address  text
);

create table if not exists predictions (
    id              bigserial primary key,
    created_at      timestamptz default now(),
    asset_tag       text,
    machine_type    text,
    prediction      int,
    probability     float,
    features        jsonb,
    model_version   text default '1.0'
);

create table if not exists alerts (
    id          bigserial primary key,
    created_at  timestamptz default now(),
    asset_tag   text,
    channel     text,
    message     text,
    status      text default 'sent'
);
"""


async def insert_audit_log(action: str, entity: str = None,
                           entity_id: str = None, payload: dict = None,
                           user_id: str = "system", ip: str = None):
    db = get_supabase()
    if db is None:
        return
    try:
        db.table("audit_logs").insert({
            "user_id": user_id,
            "action": action,
            "entity": entity,
            "entity_id": entity_id,
            "payload": payload or {},
            "ip_address": ip,
        }).execute()
    except Exception as e:
        logger.error(f"Audit log insert failed: {e}")


async def insert_prediction(record: dict):
    db = get_supabase()
    if db is None:
        return
    try:
        db.table("predictions").insert(record).execute()
    except Exception as e:
        logger.error(f"Prediction insert failed: {e}")


async def insert_alert(record: dict):
    db = get_supabase()
    if db is None:
        return
    try:
        db.table("alerts").insert(record).execute()
    except Exception as e:
        logger.error(f"Alert insert failed: {e}")
