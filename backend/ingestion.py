"""
Industrial Protocol Ingestion Layer
Handles MQTT streams and OPC-UA data frames.
Normalises raw protocol payloads into the standard SensorReading format.

Supported protocols:
  - MQTT (via paho-mqtt or simulated JSON payload)
  - OPC-UA (via opcua library or raw NodeId/Value frame parsing)
  - REST POST (standard JSON — already handled by main.py)
  - CSV/Parquet batch (handled by file_parser.py)

On Streamlit Cloud: MQTT/OPC-UA clients are not available.
The ingestion layer still parses and validates the payload format.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from typing import Optional
from backend.sensor_validator import validate_reading, ValidationResult

logger = logging.getLogger(__name__)


# ── MQTT payload normaliser ───────────────────────────────────────────────────

# Standard MQTT topic pattern: predmaint/{plant_id}/{asset_tag}/sensors
MQTT_TOPIC_PATTERN = "predmaint/{plant_id}/{asset_tag}/sensors"

# Supported MQTT payload schemas
MQTT_SCHEMA_V1 = {
    "t_brg": "temp_bearing_degC",
    "t_mtr": "temp_motor_degC",
    "vib_h": "vibration_h_mms",
    "vib_v": "vibration_v_mms",
    "p_oil": "oil_pressure_bar",
    "load":  "load_pct",
    "rpm":   "shaft_rpm",
    "pwr":   "power_consumption_kw",
}

MQTT_SCHEMA_V2 = {
    "bearing_temperature": "temp_bearing_degC",
    "motor_temperature":   "temp_motor_degC",
    "horizontal_vibration": "vibration_h_mms",
    "vertical_vibration":   "vibration_v_mms",
    "oil_pressure":         "oil_pressure_bar",
    "load_percent":         "load_pct",
    "shaft_speed_rpm":      "shaft_rpm",
    "active_power_kw":      "power_consumption_kw",
}


def parse_mqtt_payload(
    topic: str,
    payload: bytes | str | dict,
    machine_type: str = "CNC Lathe",
    prev_reading: Optional[dict] = None,
) -> dict:
    """
    Parse an MQTT message into a validated SensorReading dict.

    Args:
        topic:        MQTT topic string (e.g. predmaint/plant1/CNC-001/sensors)
        payload:      Raw bytes, JSON string, or already-parsed dict
        machine_type: Machine type for validation context
        prev_reading: Previous reading for rate-of-change checks

    Returns:
        dict with keys: asset_tag, machine_type, sensors..., validation_status,
                        malfunction (bool), malfunction_message
    """
    # Parse payload
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as e:
            return _malfunction_response("UNKNOWN", machine_type,
                                         f"MQTT payload JSON parse error: {e}")

    # Extract asset tag from topic
    parts = topic.split("/")
    asset_tag = parts[2] if len(parts) >= 3 else payload.get("asset_tag", "UNKNOWN")

    # Detect schema version and normalise
    normalised = {"asset_tag": asset_tag, "machine_type": machine_type,
                  "timestamp": payload.get("ts", datetime.utcnow().isoformat())}

    # Try V1 schema first, then V2, then direct field names
    mapped = False
    for schema in [MQTT_SCHEMA_V1, MQTT_SCHEMA_V2]:
        if any(k in payload for k in schema):
            for src, dst in schema.items():
                if src in payload:
                    try:
                        normalised[dst] = float(payload[src])
                    except (TypeError, ValueError):
                        pass
            mapped = True
            break

    if not mapped:
        # Assume direct field names (REST-style payload over MQTT)
        for field in ["temp_bearing_degC", "temp_motor_degC", "vibration_h_mms",
                      "vibration_v_mms", "oil_pressure_bar", "load_pct",
                      "shaft_rpm", "power_consumption_kw"]:
            if field in payload:
                try:
                    normalised[field] = float(payload[field])
                except (TypeError, ValueError):
                    pass

    # Validate
    result = validate_reading(normalised, prev_reading)
    normalised["validation_status"] = "OK" if result.is_valid else "MALFUNCTION"
    normalised["malfunction"] = result.malfunction
    normalised["malfunction_type"] = result.malfunction_type.value if result.malfunction_type else None
    normalised["malfunction_message"] = result.message if result.malfunction else ""

    if not result.is_valid:
        logger.warning(f"[MQTT] {asset_tag}: {result.message}")

    return normalised


# ── OPC-UA frame parser ───────────────────────────────────────────────────────

# OPC-UA NodeId to sensor field mapping
# Format: {NodeId_string: sensor_field}
OPCUA_NODE_MAP = {
    "ns=2;s=CNC001.BearingTemp":    "temp_bearing_degC",
    "ns=2;s=CNC001.MotorTemp":      "temp_motor_degC",
    "ns=2;s=CNC001.VibrationH":     "vibration_h_mms",
    "ns=2;s=CNC001.VibrationV":     "vibration_v_mms",
    "ns=2;s=CNC001.OilPressure":    "oil_pressure_bar",
    "ns=2;s=CNC001.LoadPct":        "load_pct",
    "ns=2;s=CNC001.ShaftRPM":       "shaft_rpm",
    "ns=2;s=CNC001.PowerKW":        "power_consumption_kw",
    # Add more NodeIds for other machines as needed
}


def parse_opcua_frame(
    nodes: list[dict],
    asset_tag: str = "CNC-001",
    machine_type: str = "CNC Lathe",
    prev_reading: Optional[dict] = None,
) -> dict:
    """
    Parse an OPC-UA data frame (list of {NodeId, Value, StatusCode} dicts).

    Args:
        nodes:        List of OPC-UA node readings
                      e.g. [{"NodeId": "ns=2;s=CNC001.BearingTemp",
                              "Value": 68.5, "StatusCode": "Good"}]
        asset_tag:    Asset identifier
        machine_type: Machine type for validation
        prev_reading: Previous reading for rate-of-change checks

    Returns:
        Validated SensorReading dict with validation_status field
    """
    reading = {"asset_tag": asset_tag, "machine_type": machine_type,
               "timestamp": datetime.utcnow().isoformat()}

    for node in nodes:
        node_id = node.get("NodeId", "")
        value   = node.get("Value")
        status  = node.get("StatusCode", "Good")

        # Skip bad-quality OPC-UA readings
        if status not in ("Good", "Uncertain"):
            logger.warning(f"[OPC-UA] Bad status for {node_id}: {status}")
            continue

        field = OPCUA_NODE_MAP.get(node_id)
        if field and value is not None:
            try:
                reading[field] = float(value)
            except (TypeError, ValueError):
                pass

    result = validate_reading(reading, prev_reading)
    reading["validation_status"] = "OK" if result.is_valid else "MALFUNCTION"
    reading["malfunction"] = result.malfunction
    reading["malfunction_type"] = result.malfunction_type.value if result.malfunction_type else None
    reading["malfunction_message"] = result.message if result.malfunction else ""

    if not result.is_valid:
        logger.warning(f"[OPC-UA] {asset_tag}: {result.message}")

    return reading


# ── Ingestion router ──────────────────────────────────────────────────────────

def ingest(source: str, payload: dict | bytes | str,
           asset_tag: str = "UNKNOWN", machine_type: str = "CNC Lathe",
           prev_reading: Optional[dict] = None) -> dict:
    """
    Unified ingestion entry point.
    source: 'mqtt' | 'opcua' | 'rest' | 'batch'
    """
    if source == "mqtt":
        topic = payload.get("topic", f"predmaint/plant1/{asset_tag}/sensors") \
            if isinstance(payload, dict) else f"predmaint/plant1/{asset_tag}/sensors"
        return parse_mqtt_payload(topic, payload, machine_type, prev_reading)
    elif source == "opcua":
        nodes = payload if isinstance(payload, list) else [payload]
        return parse_opcua_frame(nodes, asset_tag, machine_type, prev_reading)
    else:
        # REST / direct dict
        if isinstance(payload, dict):
            payload.setdefault("asset_tag", asset_tag)
            payload.setdefault("machine_type", machine_type)
        result = validate_reading(payload if isinstance(payload, dict) else {}, prev_reading)
        out = payload if isinstance(payload, dict) else {}
        out["validation_status"] = "OK" if result.is_valid else "MALFUNCTION"
        out["malfunction"] = result.malfunction
        out["malfunction_type"] = result.malfunction_type.value if result.malfunction_type else None
        out["malfunction_message"] = result.message if result.malfunction else ""
        return out


def _malfunction_response(asset_tag: str, machine_type: str, message: str) -> dict:
    return {
        "asset_tag": asset_tag, "machine_type": machine_type,
        "timestamp": datetime.utcnow().isoformat(),
        "validation_status": "MALFUNCTION", "malfunction": True,
        "malfunction_type": "DATA_DROP", "malfunction_message": message,
    }
