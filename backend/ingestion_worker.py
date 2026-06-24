import os
import json
import asyncio
import logging
from typing import List, Dict

from aiomqtt import Client as MQTTClient

# Local imports – adjust if package name differs
from backend import sensor_validator
from backend import timescaledb

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Environment configuration
MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
MQTT_TOPIC = "factory/sensors/#"

# Flush thresholds – can be overridden via env vars
BUFFER_MAX = int(os.getenv("TIMESCALEDB_BATCH_SIZE", "200"))
FLUSH_INTERVAL = float(os.getenv("INGESTION_FLUSH_INTERVAL_SECS", "0.5"))

async def _ensure_db() -> None:
    """Initialize the TimescaleDB connection pool if not already ready."""
    await timescaledb.init_pool()

async def _process_payload(payload: str, buffer: List[Dict]) -> None:
    """Validate a single MQTT payload and, if valid, append to the in‑memory buffer.

    The payload is expected to be a JSON object that matches the sensor schema used by
    ``sensor_validator.validate_reading``.
    """
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        log.warning("Received malformed JSON – ignored")
        return

    # ``prev_reading`` is not available in the streaming context, so we pass ``None``.
    result = sensor_validator.validate_reading(data, None)
    if result.is_valid and result.cleaned_reading:
        buffer.append(result.cleaned_reading)
    else:
        log.info("Invalid sensor reading dropped – %s", result.message)

async def _flush_buffer(buffer: List[Dict]) -> None:
    """Write all buffered records to TimescaleDB in a single batch.

    Errors from the DB layer are caught – the buffer is retained so that a subsequent
    retry can resend the same records once connectivity is restored.
    """
    if not buffer:
        return
    try:
        # In this implementation we treat each validated reading as an audit log entry.
        # Adjust the target table or helper function if a different schema is desired.
        await asyncio.gather(
            *[
                timescaledb.insert_audit_log(
                    action="sensor_ingest",
                    entity="sensor",
                    entity_id=rec.get("asset_tag"),
                    payload=rec,
                    user_id="ingestion_worker",
                )
                for rec in buffer
            ],
            return_exceptions=True,
        )
        log.info("Flushed %d records to TimescaleDB", len(buffer))
        buffer.clear()
    except Exception as exc:
        log.error("Failed to write batch to TimescaleDB – will retry later: %s", exc)
        # Do not clear the buffer; it will be retried on the next flush cycle.

async def ingestion_worker() -> None:
    """Main asynchronous loop that consumes MQTT messages, validates them, and writes
    to TimescaleDB using a time‑ or count‑based flushing strategy.
    """
    await _ensure_db()
    buffer: List[Dict] = []
    last_flush = asyncio.get_event_loop().time()

    async with MQTTClient(
        hostname=MQTT_BROKER_HOST,
        port=MQTT_BROKER_PORT,
        username=MQTT_USER,
        password=MQTT_PASSWORD,
    ) as client:
        await client.subscribe(MQTT_TOPIC)
        log.info("Subscribed to MQTT topic %s on %s:%s", MQTT_TOPIC, MQTT_BROKER_HOST, MQTT_BROKER_PORT)
        async for message in client.messages:
            payload = message.payload.decode()
            await _process_payload(payload, buffer)
            now = asyncio.get_event_loop().time()
            if len(buffer) >= BUFFER_MAX or (now - last_flush) >= FLUSH_INTERVAL:
                await _flush_buffer(buffer)
                last_flush = now

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(ingestion_worker())
    except KeyboardInterrupt:
        log.info("Ingestion worker stopped by user")

if __name__ == "__main__":
    main()
