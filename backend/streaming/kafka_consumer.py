"""Async Kafka ingestion with causal sensor features and a TimescaleDB sink."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping

import asyncpg
from aiokafka import AIOKafkaConsumer

LOGGER = logging.getLogger(__name__)
TOPIC = "industrial.sensors.v1"
SENSOR_FIELDS = ("vibration", "thermal", "acoustic")
RecordHandler = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class RollingFeatures:
    """Bounded, per-asset rolling windows for RMS, kurtosis and crest factor."""

    window_size: int = 120

    def __post_init__(self) -> None:
        if self.window_size < 4:
            raise ValueError("window_size must be at least 4 for kurtosis")
        self._windows: dict[tuple[str, str], deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    def update(self, asset_id: str, sensors: Mapping[str, float]) -> dict[str, float]:
        output: dict[str, float] = {}
        for sensor in SENSOR_FIELDS:
            values = self._windows[(asset_id, sensor)]
            value = float(sensors[sensor])
            if not math.isfinite(value):
                raise ValueError(f"{sensor} must be finite")
            values.append(value)
            mean = sum(values) / len(values)
            centered = [sample - mean for sample in values]
            second = sum(item * item for item in centered) / len(values)
            fourth = sum(item ** 4 for item in centered) / len(values)
            rms = math.sqrt(sum(item * item for item in values) / len(values))
            output[f"{sensor}_rms"] = rms
            output[f"{sensor}_kurtosis"] = fourth / (second * second) if second else 0.0
            output[f"{sensor}_crest_factor"] = max(map(abs, values)) / rms if rms else 0.0
        return output


def normalize_payload(payload: bytes | str, features: RollingFeatures) -> dict[str, Any]:
    """Validate an event and attach causal rolling features."""
    try:
        raw = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ValueError("message must be valid JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("message must be a JSON object")
    asset_id, plant_id = raw.get("asset_id"), raw.get("plant_id")
    if not isinstance(asset_id, str) or not asset_id.strip():
        raise ValueError("asset_id is required")
    if not isinstance(plant_id, str) or not plant_id.strip():
        raise ValueError("plant_id is required")
    timestamp = raw.get("event_timestamp", raw.get("timestamp"))
    if not isinstance(timestamp, str):
        raise ValueError("event_timestamp (ISO-8601) is required")
    try:
        event_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("event_timestamp must be ISO-8601") from exc
    if event_time.tzinfo is None:
        raise ValueError("event_timestamp must include a timezone")
    sensors: dict[str, float] = {}
    for field in SENSOR_FIELDS:
        try:
            sensors[field] = float(raw[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be numeric") from exc
        if not math.isfinite(sensors[field]):
            raise ValueError(f"{field} must be finite")
    asset_id, plant_id = asset_id.strip(), plant_id.strip()
    return {
        "asset_id": asset_id,
        "plant_id": plant_id,
        "event_timestamp": event_time.astimezone(timezone.utc),
        **sensors,
        **features.update(asset_id, sensors),
    }


async def insert_sensor_record(record: dict[str, Any], pool: asyncpg.Pool | None = None) -> None:
    """Persist an event and its point-in-time features; failures propagate for Kafka retry."""
    own_pool = pool is None
    if own_pool:
        dsn = os.getenv("TIMESCALEDB_DSN")
        if not dsn:
            raise RuntimeError("TIMESCALEDB_DSN is required for Kafka ingestion")
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
    assert pool is not None
    try:
        columns = SENSOR_FIELDS + tuple(
            f"{sensor}_{stat}" for sensor in SENSOR_FIELDS
            for stat in ("rms", "kurtosis", "crest_factor")
        )
        names = ", ".join(columns)
        placeholders = ", ".join(f"${index}" for index in range(1, len(columns) + 4))
        values = [record["event_timestamp"], record["asset_id"], record["plant_id"]]
        values.extend(record[column] for column in columns)
        async with pool.acquire() as connection:
            await connection.execute(
                f"INSERT INTO sensor_telemetry (event_timestamp, asset_id, plant_id, {names}) "
                f"VALUES ({placeholders}) ON CONFLICT (event_timestamp, asset_id) DO NOTHING",
                *values,
            )
    finally:
        if own_pool:
            await pool.close()


async def consume(
    record_handler: RecordHandler = insert_sensor_record,
    *, bootstrap_servers: str | None = None,
    topic: str = TOPIC,
    group_id: str = "predmaint-feature-ingestion",
    window_size: int = 120,
) -> None:
    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        group_id=group_id,
        enable_auto_commit=False,
        auto_offset_reset="latest",
    )
    await consumer.start()
    rolling = RollingFeatures(window_size=window_size)
    try:
        async for message in consumer:
            try:
                record = normalize_payload(message.value, rolling)
            except ValueError as exc:
                LOGGER.warning("Dropping invalid telemetry from %s: %s", topic, exc)
                await consumer.commit()
                continue
            await record_handler(record)
            await consumer.commit()
    finally:
        await consumer.stop()


async def main() -> None:
    await consume()


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    asyncio.run(main())
