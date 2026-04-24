"""
Machine Health Score Engine
Computes a 0-100 health score per asset using weighted sensor deviation.
Tracks score history and generates trend alerts.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Normal operating thresholds per machine type
THRESHOLDS = {
    "CNC Lathe":        dict(temp_bearing_degC=75, temp_motor_degC=85, vibration_h_mms=4.0, vibration_v_mms=3.5, oil_pressure_bar=(4.5, 6.5), load_pct=90, power_consumption_kw=35),
    "Hydraulic Press":  dict(temp_bearing_degC=70, temp_motor_degC=80, vibration_h_mms=2.5, vibration_v_mms=2.0, oil_pressure_bar=(80, 120), load_pct=95, power_consumption_kw=80),
    "Belt Conveyor":    dict(temp_bearing_degC=60, temp_motor_degC=65, vibration_h_mms=5.0, vibration_v_mms=4.0, oil_pressure_bar=(2.0, 4.0), load_pct=80, power_consumption_kw=20),
    "Screw Compressor": dict(temp_bearing_degC=85, temp_motor_degC=95, vibration_h_mms=3.5, vibration_v_mms=3.0, oil_pressure_bar=(6.0, 10.0), load_pct=100, power_consumption_kw=55),
    "EOT Crane":        dict(temp_bearing_degC=55, temp_motor_degC=60, vibration_h_mms=2.0, vibration_v_mms=1.8, oil_pressure_bar=(3.0, 5.0), load_pct=70, power_consumption_kw=40),
}

WEIGHTS = {
    "temp_bearing_degC": 0.20,
    "temp_motor_degC":   0.15,
    "vibration_h_mms":   0.25,
    "vibration_v_mms":   0.15,
    "oil_pressure_bar":  0.15,
    "power_consumption_kw": 0.10,
}


@dataclass
class HealthResult:
    asset_tag: str
    machine_type: str
    health_score: float       # 0-100 (100 = perfect)
    status: str               # Healthy / Warning / Critical
    component_scores: dict
    recommendations: list[str]


def compute_health_score(reading: dict) -> HealthResult:
    mtype = reading.get("machine_type", "CNC Lathe")
    thresh = THRESHOLDS.get(mtype, THRESHOLDS["CNC Lathe"])
    component_scores = {}
    weighted_penalty = 0.0

    for sensor, weight in WEIGHTS.items():
        val = reading.get(sensor)
        if val is None:
            component_scores[sensor] = 100.0
            continue

        limit = thresh.get(sensor)
        if isinstance(limit, tuple):
            lo, hi = limit
            if lo <= val <= hi:
                penalty = 0.0
            elif val < lo:
                penalty = min(1.0, (lo - val) / lo)
            else:
                penalty = min(1.0, (val - hi) / hi)
        else:
            penalty = min(1.0, max(0.0, (val - limit) / limit)) if val > limit else 0.0

        score = round((1 - penalty) * 100, 1)
        component_scores[sensor] = score
        weighted_penalty += penalty * weight

    health = round(max(0.0, (1 - weighted_penalty) * 100), 1)

    if health >= 75:
        status = "Healthy"
    elif health >= 50:
        status = "Warning"
    else:
        status = "Critical"

    recs = _generate_recommendations(component_scores, reading)

    return HealthResult(
        asset_tag=reading.get("asset_tag", "UNKNOWN"),
        machine_type=mtype,
        health_score=health,
        status=status,
        component_scores=component_scores,
        recommendations=recs,
    )


def compute_fleet_health(df: pd.DataFrame) -> pd.DataFrame:
    """Compute health score for each row in a dataframe."""
    results = []
    for _, row in df.iterrows():
        r = compute_health_score(row.to_dict())
        results.append({
            "asset_tag": r.asset_tag,
            "machine_type": r.machine_type,
            "health_score": r.health_score,
            "status": r.status,
        })
    return pd.DataFrame(results)


def _generate_recommendations(scores: dict, reading: dict) -> list[str]:
    recs = []
    if scores.get("temp_bearing_degC", 100) < 70:
        recs.append("Inspect bearing lubrication – temperature elevated")
    if scores.get("temp_motor_degC", 100) < 70:
        recs.append("Check motor cooling system and ventilation")
    if scores.get("vibration_h_mms", 100) < 70:
        recs.append("Check shaft alignment and balance – high vibration detected")
    if scores.get("oil_pressure_bar", 100) < 70:
        recs.append("Inspect hydraulic seals and oil level")
    if scores.get("power_consumption_kw", 100) < 70:
        recs.append("Investigate abnormal power draw – possible electrical fault")
    if not recs:
        recs.append("All parameters within normal range – continue scheduled PM")
    return recs
