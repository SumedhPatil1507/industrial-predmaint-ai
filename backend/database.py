"""Supabase client + table helpers with full persistence."""
from supabase import create_client, Client
from backend.config import get_settings
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

_client: Client | None = None


def get_supabase() -> Client | None:
    global _client
    if _client is None:
        s = get_settings()
        if s.supabase_url and s.supabase_key:
            try:
                _client = create_client(s.supabase_url, s.supabase_key)
            except Exception as e:
                logger.warning(f"Supabase init failed: {e}")
    return _client


def is_connected() -> bool:
    return get_supabase() is not None


async def insert_audit_log(action: str, entity: str = None,
                           entity_id: str = None, payload: dict = None,
                           user_id: str = "system", ip: str = None):
    db = get_supabase()
    if db is None:
        return
    try:
        db.table("audit_logs").insert({
            "user_id": user_id, "action": action,
            "entity": entity, "entity_id": entity_id,
            "payload": payload or {}, "ip_address": ip,
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


def get_prediction_history(asset_tag: str = None, limit: int = 100) -> list[dict]:
    db = get_supabase()
    if db is None:
        return []
    try:
        q = db.table("predictions").select("*").order("created_at", desc=True).limit(limit)
        if asset_tag:
            q = q.eq("asset_tag", asset_tag)
        return q.execute().data or []
    except Exception as e:
        logger.error(f"Prediction history fetch failed: {e}")
        return []


def get_alert_history(limit: int = 50) -> list[dict]:
    db = get_supabase()
    if db is None:
        return []
    try:
        return db.table("alerts").select("*").order("created_at", desc=True).limit(limit).execute().data or []
    except Exception as e:
        logger.error(f"Alert history fetch failed: {e}")
        return []



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
