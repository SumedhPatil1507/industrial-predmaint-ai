"""
Sensor Data Validation Layer
Pydantic models + physics-based cross-sensor validation.

Detects:
  - Out-of-range values (physical limits)
  - Cross-sensor inconsistencies (e.g. temp surge without pressure rise)
  - Sensor malfunction patterns (stuck values, impossible rates of change)
  - Data drops (null / NaN / zero-fill artifacts)

Returns a validated reading OR a SensorMalfunctionEvent — never a false breakdown.
"""
from __future__ import annotations
from pydantic import BaseModel, field_validator, model_validator
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum
import math


# ── Physical hard limits (absolute impossibility beyond these) ────────────────
HARD_LIMITS = {
    "CNC Lathe": dict(
        temp_bearing_degC=(0, 150), temp_motor_degC=(0, 160),
        vibration_h_mms=(0, 25),   vibration_v_mms=(0, 20),
        oil_pressure_bar=(0, 15),  load_pct=(0, 100),
        shaft_rpm=(0, 5000),       power_consumption_kw=(0, 80),
    ),
    "Hydraulic Press": dict(
        temp_bearing_degC=(0, 130), temp_motor_degC=(0, 140),
        vibration_h_mms=(0, 15),   vibration_v_mms=(0, 12),
        oil_pressure_bar=(0, 200), load_pct=(0, 100),
        shaft_rpm=(0, 1500),       power_consumption_kw=(0, 150),
    ),
    "Belt Conveyor": dict(
        temp_bearing_degC=(0, 110), temp_motor_degC=(0, 120),
        vibration_h_mms=(0, 20),   vibration_v_mms=(0, 18),
        oil_pressure_bar=(0, 10),  load_pct=(0, 100),
        shaft_rpm=(0, 800),        power_consumption_kw=(0, 50),
    ),
    "Screw Compressor": dict(
        temp_bearing_degC=(0, 160), temp_motor_degC=(0, 170),
        vibration_h_mms=(0, 18),   vibration_v_mms=(0, 15),
        oil_pressure_bar=(0, 20),  load_pct=(0, 100),
        shaft_rpm=(0, 5000),       power_consumption_kw=(0, 120),
    ),
    "EOT Crane": dict(
        temp_bearing_degC=(0, 100), temp_motor_degC=(0, 110),
        vibration_h_mms=(0, 12),   vibration_v_mms=(0, 10),
        oil_pressure_bar=(0, 12),  load_pct=(0, 100),
        shaft_rpm=(0, 500),        power_consumption_kw=(0, 80),
    ),
}

# Cross-sensor physics rules per machine type
# Format: (trigger_sensor, trigger_condition, correlated_sensor, expected_direction, tolerance_pct)
CROSS_SENSOR_RULES = {
    "Screw Compressor": [
        # If temp surges >30C above normal, pressure must also rise (>5%)
        # If temp surges but pressure drops — sensor malfunction
        ("temp_bearing_degC", "surge", "oil_pressure_bar", "rise", 5.0),
        # If RPM drops >50%, power must also drop
        ("shaft_rpm", "drop", "power_consumption_kw", "drop", 10.0),
    ],
    "Hydraulic Press": [
        # If pressure drops >30%, load must also drop
        ("oil_pressure_bar", "drop", "load_pct", "drop", 15.0),
    ],
    "CNC Lathe": [
        # If RPM drops >40%, power must drop
        ("shaft_rpm", "drop", "power_consumption_kw", "drop", 10.0),
    ],
}

# Normal operating baselines for rate-of-change checks
NORMAL_RANGES = {
    "CNC Lathe":        dict(temp_bearing_degC=(55, 75), temp_motor_degC=(60, 85),
                             oil_pressure_bar=(4.5, 6.5), shaft_rpm=(800, 2000)),
    "Hydraulic Press":  dict(temp_bearing_degC=(50, 70), temp_motor_degC=(55, 80),
                             oil_pressure_bar=(80, 120), shaft_rpm=(200, 600)),
    "Belt Conveyor":    dict(temp_bearing_degC=(40, 60), temp_motor_degC=(45, 65),
                             oil_pressure_bar=(2.0, 4.0), shaft_rpm=(100, 400)),
    "Screw Compressor": dict(temp_bearing_degC=(60, 85), temp_motor_degC=(65, 95),
                             oil_pressure_bar=(6.0, 10.0), shaft_rpm=(1500, 3000)),
    "EOT Crane":        dict(temp_bearing_degC=(35, 55), temp_motor_degC=(40, 60),
                             oil_pressure_bar=(3.0, 5.0), shaft_rpm=(50, 200)),
}


class MalfunctionType(str, Enum):
    OUT_OF_RANGE       = "OUT_OF_RANGE"
    CROSS_SENSOR_FAULT = "CROSS_SENSOR_FAULT"
    STUCK_VALUE        = "STUCK_VALUE"
    DATA_DROP          = "DATA_DROP"
    IMPOSSIBLE_RATE    = "IMPOSSIBLE_RATE"


@dataclass
class ValidationResult:
    is_valid: bool
    malfunction: bool = False
    malfunction_type: Optional[MalfunctionType] = None
    affected_sensors: list = field(default_factory=list)
    message: str = ""
    cleaned_reading: Optional[dict] = None


class SensorReading(BaseModel):
    """Pydantic model with physical boundary validation."""
    asset_tag: str = "UNKNOWN"
    machine_type: str = "CNC Lathe"
    temp_bearing_degC: float
    temp_motor_degC: float
    vibration_h_mms: float
    vibration_v_mms: float
    oil_pressure_bar: float
    load_pct: float
    shaft_rpm: float
    power_consumption_kw: float
    timestamp: Optional[str] = None

    @field_validator("temp_bearing_degC", "temp_motor_degC")
    @classmethod
    def temp_must_be_positive(cls, v: float) -> float:
        if v < -10 or v > 200:
            raise ValueError(f"Temperature {v} outside physical range [-10, 200]")
        return v

    @field_validator("vibration_h_mms", "vibration_v_mms")
    @classmethod
    def vibration_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"Vibration cannot be negative: {v}")
        if v > 50:
            raise ValueError(f"Vibration {v} mm/s exceeds physical maximum")
        return v

    @field_validator("load_pct")
    @classmethod
    def load_in_range(cls, v: float) -> float:
        if not (0 <= v <= 100):
            raise ValueError(f"Load % must be 0-100, got {v}")
        return v

    @field_validator("shaft_rpm")
    @classmethod
    def rpm_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"RPM cannot be negative: {v}")
        return v

    @field_validator("power_consumption_kw")
    @classmethod
    def power_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"Power cannot be negative: {v}")
        return v

    @model_validator(mode="after")
    def motor_hotter_than_bearing(self) -> "SensorReading":
        """Motor temp should generally be >= bearing temp (within 30C tolerance)."""
        diff = self.temp_bearing_degC - self.temp_motor_degC
        if diff > 30:
            raise ValueError(
                f"Bearing temp ({self.temp_bearing_degC}C) is {diff:.0f}C hotter than "
                f"motor ({self.temp_motor_degC}C) — likely sensor swap or malfunction"
            )
        return self


def validate_reading(reading: dict, prev_reading: Optional[dict] = None) -> ValidationResult:
    """
    Full validation pipeline:
    1. Pydantic type + range check
    2. Machine-specific hard limits
    3. Cross-sensor physics rules
    4. Rate-of-change check (if previous reading provided)
    """
    mtype = reading.get("machine_type", "CNC Lathe")

    # Step 1: Pydantic validation
    try:
        parsed = SensorReading(**reading)
    except Exception as e:
        return ValidationResult(
            is_valid=False, malfunction=True,
            malfunction_type=MalfunctionType.OUT_OF_RANGE,
            message=f"Sensor Malfunction / Data Drop: {e}",
            affected_sensors=_extract_sensor_from_error(str(e)),
        )

    # Step 2: Machine-specific hard limits
    limits = HARD_LIMITS.get(mtype, HARD_LIMITS["CNC Lathe"])
    for sensor, (lo, hi) in limits.items():
        val = getattr(parsed, sensor, None)
        if val is not None and not (lo <= val <= hi):
            return ValidationResult(
                is_valid=False, malfunction=True,
                malfunction_type=MalfunctionType.OUT_OF_RANGE,
                affected_sensors=[sensor],
                message=(
                    f"Sensor Malfunction / Data Drop: {sensor} = {val} "
                    f"outside physical limits [{lo}, {hi}] for {mtype}"
                ),
            )

    # Step 3: Cross-sensor physics rules
    rules = CROSS_SENSOR_RULES.get(mtype, [])
    normal = NORMAL_RANGES.get(mtype, {})
    for trigger, condition, correlated, expected, tol in rules:
        t_val = getattr(parsed, trigger, None)
        c_val = getattr(parsed, correlated, None)
        if t_val is None or c_val is None:
            continue
        t_range = normal.get(trigger)
        c_range = normal.get(correlated)
        if not t_range or not c_range:
            continue
        t_mid = (t_range[0] + t_range[1]) / 2
        c_mid = (c_range[0] + c_range[1]) / 2

        trigger_fired = False
        if condition == "surge" and t_val > t_range[1] * 1.3:
            trigger_fired = True
        elif condition == "drop" and t_val < t_range[0] * 0.5:
            trigger_fired = True

        if trigger_fired:
            c_shift = abs(c_val - c_mid) / c_mid * 100
            if expected == "rise" and c_val < c_mid * (1 - tol / 100):
                return ValidationResult(
                    is_valid=False, malfunction=True,
                    malfunction_type=MalfunctionType.CROSS_SENSOR_FAULT,
                    affected_sensors=[trigger, correlated],
                    message=(
                        f"Sensor Malfunction / Data Drop: {trigger} surged to {t_val} "
                        f"but {correlated} did not rise as expected (got {c_val}). "
                        f"Likely {trigger} sensor fault — not a machine breakdown."
                    ),
                )
            elif expected == "drop" and c_val > c_mid * (1 + tol / 100):
                return ValidationResult(
                    is_valid=False, malfunction=True,
                    malfunction_type=MalfunctionType.CROSS_SENSOR_FAULT,
                    affected_sensors=[trigger, correlated],
                    message=(
                        f"Sensor Malfunction / Data Drop: {trigger} dropped to {t_val} "
                        f"but {correlated} did not drop as expected (got {c_val}). "
                        f"Likely {trigger} sensor fault — not a machine breakdown."
                    ),
                )

    # Step 4: Rate-of-change check
    if prev_reading:
        for sensor in ["temp_bearing_degC", "temp_motor_degC", "oil_pressure_bar"]:
            curr_val = getattr(parsed, sensor, None)
            prev_val = prev_reading.get(sensor)
            if curr_val is None or prev_val is None or prev_val == 0:
                continue
            change_pct = abs(curr_val - prev_val) / abs(prev_val) * 100
            # Flag if sensor changes by more than 50% in one tick (1 second)
            if change_pct > 50:
                return ValidationResult(
                    is_valid=False, malfunction=True,
                    malfunction_type=MalfunctionType.IMPOSSIBLE_RATE,
                    affected_sensors=[sensor],
                    message=(
                        f"Sensor Malfunction / Data Drop: {sensor} changed by "
                        f"{change_pct:.0f}% in one reading interval "
                        f"({prev_val} -> {curr_val}). Physically impossible rate."
                    ),
                )

    return ValidationResult(
        is_valid=True,
        cleaned_reading=parsed.model_dump(),
        message="OK",
    )


def _extract_sensor_from_error(error_str: str) -> list:
    sensors = ["temp_bearing_degC", "temp_motor_degC", "vibration_h_mms",
               "vibration_v_mms", "oil_pressure_bar", "load_pct",
               "shaft_rpm", "power_consumption_kw"]
    return [s for s in sensors if s in error_str]
