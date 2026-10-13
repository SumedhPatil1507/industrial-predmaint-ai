"""Async Kafka ingestion and rolling sensor feature extraction.

Messages are acknowledged only after ``record_handler`` has accepted the normalized
record. The handler is the integration point for a durable feature sink (for example,
an online store or a stream processing service).
"""
from __future__ import annotations

import json
import logging
import math
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping

from aiokafka import AIOKafkaConsumer

LOGGER = logging.getLogger(__name__)
TOPIC = "industrial.sensors.v1"
SENSOR_FIELDS = ("vibration", "temperature", "pressure", "acoustic")
RecordHandler = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class RollingFeatures:
    """Bounded, per-asset sample windows for the four sensor channels."""

    window_size: int = 120

    def __post_init__(self) -> None:
        if self.window_size < 4:
            raise ValueError("window_size must be at least 4 for kurtosis")
        self._windows: dict[tuple[str, str], deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    def update(self, asset_id: str, plant_id: str, sensors: Mapping[str, float]) -> dict[str, float]:
        output: dict[str, float] = {}
        for sensor in SENSOR_FIELDS:
            values = self._windows[(asset_id, f"{plant_id}:{sensor}")]
            value = float(sensors[sensor])
            if not math.isfinite(value):
                raise ValueError(f"{sensor} must be finite")
            values.append(value)
            mean = sum(values) / len(values)
            centered = [sample - mean for sample in values]
            second = sum(item * item for item in centered) / len(values)
            fourth = sum(item ** 4 for item in centered) / len(values)
            prefix = sensor
            output[f"{prefix}_rms"] = math.sqrt(sum(item * item for item in values) / len(values))
            output[f"{prefix}_kurtosis"] = fourth / (second * second) if second else 0.0
            output[f"{prefix}_crest_factor"] = max(abs(item) for item in values) / output[f"{prefix}_rms"] if output[f"{prefix}_rms"] else 0.0
        return output


def normalize_payload(payload: bytes | str, features: RollingFeatures) -> dict[str, Any]:
    """Decode and validate a sensor event, then attach causal rolling features."""
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
    event_time = raw.get("event_timestamp", raw.get("timestamp"))
    if not isinstance(event_time, str):
        raise ValueError("event_timestamp (ISO-8601) is required")
    try:
        parsed_time = datetime.fromisoformat(event_time.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("event_timestamp must be ISO-8601") from exc
    if parsed_time.tzinfo is None:
        raise ValueError("event_timestamp must include a timezone")
    sensor_values: dict[str, float] = {}
    for field in SENSOR_FIELDS:
        try:
            value = float(raw[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
        sensor_values[field] = value
    return {
        "asset_id": asset_id.strip(),
        "plant_id": plant_id.strip(),
        "event_timestamp": parsed_time.astimezone(timezone.utc),
        **sensor_values,
        **features.update(asset_id.strip(), plant_id.strip(), sensor_values),
    }


async def consume(
    record_handler: RecordHandler,
    *,
    bootstrap_servers: str | None = None,
    topic: str = TOPIC,
    group_id: str = "predmaint-feature-ingestion",
    window_size: int = 120,
) -> None:
    """Consume Kafka records, extract rolling features, and call a durable sink."""
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
                # Poison records are logged and committed so a malformed message cannot
                # block its partition. Sink failures are not swallowed or committed.
                LOGGER.warning("Dropping invalid telemetry from %s: %s", topic, exc)
                await consumer.commit()
                continue
            # Sink failures deliberately escape so Kafka can redeliver the message.
            await record_handler(record)
            await consumer.commit()
    finally:
        await consumer.stop()


async def main() -> None:
    async def log_record(record: dict[str, Any]) -> None:
        LOGGER.info("Ingested telemetry for %s at %s", record["asset_id"], record["event_timestamp"])

    await consume(log_record)


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    asyncio.run(main())
