"""
Hardware Stress-Testing Harness
Simulates industrial sensor streams for validation, load testing, and
pipeline verification. NOT a production data source.

Stress test modes:
  NORMAL      — Steady-state operation within normal ranges
  DEGRADING   — Gradual wear simulation (default)
  SPIKE       — Sudden sensor spike events
  MALFUNCTION — Intentional cross-sensor inconsistency injection
  FAILURE     — Imminent breakdown scenario

Usage:
  from backend.iot_simulator import StressTestHarness
  harness = StressTestHarness(mode="DEGRADING")
  reading = harness.generate_reading("CNC-001")
"""
import asyncio
import random
import json
from datetime import datetime
from enum import Enum
from typing import Optional

MACHINES = [
    {"asset_tag": "CNC-001", "machine_type": "CNC Lathe"},
    {"asset_tag": "HYD-001", "machine_type": "Hydraulic Press"},
    {"asset_tag": "BLT-001", "machine_type": "Belt Conveyor"},
    {"asset_tag": "CMP-001", "machine_type": "Screw Compressor"},
    {"asset_tag": "EOT-001", "machine_type": "EOT Crane"},
]

RANGES = {
    "CNC Lathe":        dict(tb=(55,75), tm=(60,85), vh=(1.5,4.0), vv=(1.0,3.5),
                             op=(4.5,6.5), lp=(40,90), rpm=(800,2000), pw=(15,35)),
    "Hydraulic Press":  dict(tb=(50,70), tm=(55,80), vh=(0.5,2.5), vv=(0.5,2.0),
                             op=(80,120), lp=(50,95), rpm=(200,600), pw=(30,80)),
    "Belt Conveyor":    dict(tb=(40,60), tm=(45,65), vh=(2.0,5.0), vv=(1.5,4.0),
                             op=(2.0,4.0), lp=(30,80), rpm=(100,400), pw=(5,20)),
    "Screw Compressor": dict(tb=(60,85), tm=(65,95), vh=(1.0,3.5), vv=(0.8,3.0),
                             op=(6.0,10.0), lp=(60,100), rpm=(1500,3000), pw=(20,55)),
    "EOT Crane":        dict(tb=(35,55), tm=(40,60), vh=(0.5,2.0), vv=(0.5,1.8),
                             op=(3.0,5.0), lp=(20,70), rpm=(50,200), pw=(10,40)),
}


class StressMode(str, Enum):
    NORMAL      = "NORMAL"
    DEGRADING   = "DEGRADING"
    SPIKE       = "SPIKE"
    MALFUNCTION = "MALFUNCTION"
    FAILURE     = "FAILURE"


class StressTestHarness:
    """
    Hardware Stress-Testing Harness.
    Generates synthetic sensor streams for pipeline validation.
    """

    def __init__(self, mode: StressMode = StressMode.DEGRADING):
        self.mode = mode
        self._state = {
            m["asset_tag"]: {"degradation": 0.0, "tick": 0}
            for m in MACHINES
        }

    def generate_reading(self, asset_tag: str) -> dict:
        machine = next((m for m in MACHINES if m["asset_tag"] == asset_tag), MACHINES[0])
        return self._generate(machine)

    def generate_all(self) -> list[dict]:
        return [self._generate(m) for m in MACHINES]

    def _generate(self, machine: dict) -> dict:
        tag   = machine["asset_tag"]
        mtype = machine["machine_type"]
        r     = RANGES[mtype]
        s     = self._state[tag]
        s["tick"] += 1

        if self.mode == StressMode.NORMAL:
            deg = 0.0
        elif self.mode == StressMode.DEGRADING:
            if random.random() < 0.002:
                s["degradation"] = 0.0
            s["degradation"] = min(1.0, s["degradation"] + random.uniform(0, 0.005))
            deg = s["degradation"]
        elif self.mode == StressMode.FAILURE:
            s["degradation"] = min(1.0, s["degradation"] + random.uniform(0.01, 0.03))
            deg = s["degradation"]
        else:
            deg = s["degradation"]

        def noisy(lo, hi, bias=0.0):
            return round(random.uniform(lo, hi) +
                         random.gauss(0, (hi-lo)*0.05) + bias*(hi-lo), 2)

        tb  = noisy(*r["tb"],  bias=deg*0.4)
        tm  = noisy(*r["tm"],  bias=deg*0.5)
        vh  = max(0, noisy(*r["vh"], bias=deg*0.6))
        vv  = max(0, noisy(*r["vv"], bias=deg*0.5))
        op  = max(0, noisy(*r["op"], bias=-deg*0.3))
        lp  = max(0, min(100, noisy(*r["lp"])))
        rpm = max(0, noisy(*r["rpm"], bias=-deg*0.1))
        pw  = max(0, noisy(*r["pw"],  bias=deg*0.2))

        # Spike mode: inject sudden sensor spikes
        if self.mode == StressMode.SPIKE or random.random() < 0.01*(1+deg*3):
            spike_sensor = random.choice(["vh", "tb"])
            if spike_sensor == "vh":
                vh *= random.uniform(2.0, 3.5)
            else:
                tb *= random.uniform(1.2, 1.5)

        # Malfunction mode: inject cross-sensor inconsistency
        # (temp surge without pressure rise — should be caught by validator)
        if self.mode == StressMode.MALFUNCTION:
            tb = r["tb"][1] * 1.8   # 80% above max
            op = r["op"][0] * 0.5   # pressure drops instead of rising

        return {
            "timestamp":            datetime.utcnow().isoformat(),
            "asset_tag":            tag,
            "machine_type":         mtype,
            "stress_mode":          self.mode.value,
            "temp_bearing_degC":    round(tb, 2),
            "temp_motor_degC":      round(tm, 2),
            "vibration_h_mms":      round(vh, 2),
            "vibration_v_mms":      round(vv, 2),
            "oil_pressure_bar":     round(op, 2),
            "load_pct":             round(lp, 1),
            "shaft_rpm":            round(rpm, 0),
            "power_consumption_kw": round(pw, 2),
            "degradation_index":    round(deg, 3),
        }


# ── Default harness instance (used by main.py WebSocket) ─────────────────────
_default_harness = StressTestHarness(mode=StressMode.DEGRADING)


def _generate_reading(machine: dict) -> dict:
    """Backward-compatible wrapper for existing code."""
    return _default_harness._generate(machine)


async def stream_sensor_data(websocket, interval: float = 1.0):
    """Stream validated sensor readings via WebSocket."""
    from backend.sensor_validator import validate_reading
    prev_readings: dict = {}

    try:
        while True:
            readings = _default_harness.generate_all()
            validated = []
            for r in readings:
                tag = r["asset_tag"]
                result = validate_reading(r, prev_readings.get(tag))
                r["validation_status"] = "OK" if result.is_valid else "MALFUNCTION"
                r["malfunction"] = result.malfunction
                r["malfunction_message"] = result.message if result.malfunction else ""
                if result.is_valid:
                    prev_readings[tag] = r
                validated.append(r)
            await websocket.send_text(json.dumps(validated))
            await asyncio.sleep(interval)
    except Exception:
        pass
