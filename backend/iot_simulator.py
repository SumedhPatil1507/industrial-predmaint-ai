"""Live IoT sensor simulation – streams realistic sensor data via WebSocket."""
import asyncio
import random
import json
import math
from datetime import datetime

MACHINES = [
    {"asset_tag": "CNC-001", "machine_type": "CNC Lathe"},
    {"asset_tag": "HYD-001", "machine_type": "Hydraulic Press"},
    {"asset_tag": "BLT-001", "machine_type": "Belt Conveyor"},
    {"asset_tag": "CMP-001", "machine_type": "Screw Compressor"},
    {"asset_tag": "EOT-001", "machine_type": "EOT Crane"},
]

# Normal operating ranges per machine type
RANGES = {
    "CNC Lathe":        dict(tb=(55, 75), tm=(60, 85), vh=(1.5, 4.0), vv=(1.0, 3.5), op=(4.5, 6.5), lp=(40, 90), rpm=(800, 2000), pw=(15, 35)),
    "Hydraulic Press":  dict(tb=(50, 70), tm=(55, 80), vh=(0.5, 2.5), vv=(0.5, 2.0), op=(80, 120), lp=(50, 95), rpm=(200, 600), pw=(30, 80)),
    "Belt Conveyor":    dict(tb=(40, 60), tm=(45, 65), vh=(2.0, 5.0), vv=(1.5, 4.0), op=(2.0, 4.0), lp=(30, 80), rpm=(100, 400), pw=(5, 20)),
    "Screw Compressor": dict(tb=(60, 85), tm=(65, 95), vh=(1.0, 3.5), vv=(0.8, 3.0), op=(6.0, 10.0), lp=(60, 100), rpm=(1500, 3000), pw=(20, 55)),
    "EOT Crane":        dict(tb=(35, 55), tm=(40, 60), vh=(0.5, 2.0), vv=(0.5, 1.8), op=(3.0, 5.0), lp=(20, 70), rpm=(50, 200), pw=(10, 40)),
}

# Degradation state per machine (simulates gradual wear)
_state: dict[str, dict] = {
    m["asset_tag"]: {"degradation": 0.0, "tick": 0}
    for m in MACHINES
}


def _generate_reading(machine: dict) -> dict:
    tag = machine["asset_tag"]
    mtype = machine["machine_type"]
    r = RANGES[mtype]
    state = _state[tag]

    state["tick"] += 1
    # Slowly increase degradation, reset after "repair"
    if random.random() < 0.002:
        state["degradation"] = 0.0  # simulated repair
    state["degradation"] = min(1.0, state["degradation"] + random.uniform(0, 0.005))
    deg = state["degradation"]

    def noisy(lo, hi, bias=0.0):
        base = random.uniform(lo, hi)
        noise = random.gauss(0, (hi - lo) * 0.05)
        return round(base + noise + bias * (hi - lo), 2)

    # Degradation pushes temps and vibration up, pressure down
    tb = noisy(*r["tb"], bias=deg * 0.4)
    tm = noisy(*r["tm"], bias=deg * 0.5)
    vh = noisy(*r["vh"], bias=deg * 0.6)
    vv = noisy(*r["vv"], bias=deg * 0.5)
    op = noisy(*r["op"], bias=-deg * 0.3)
    lp = noisy(*r["lp"])
    rpm = noisy(*r["rpm"], bias=-deg * 0.1)
    pw = noisy(*r["pw"], bias=deg * 0.2)

    # Spike events
    if random.random() < 0.01 * (1 + deg * 3):
        vh *= random.uniform(1.5, 2.5)
        tb *= random.uniform(1.1, 1.3)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "asset_tag": tag,
        "machine_type": mtype,
        "temp_bearing_degC": round(tb, 2),
        "temp_motor_degC": round(tm, 2),
        "vibration_h_mms": round(max(0, vh), 2),
        "vibration_v_mms": round(max(0, vv), 2),
        "oil_pressure_bar": round(max(0, op), 2),
        "load_pct": round(max(0, min(100, lp)), 1),
        "shaft_rpm": round(max(0, rpm), 0),
        "power_consumption_kw": round(max(0, pw), 2),
        "degradation_index": round(deg, 3),
    }


async def stream_sensor_data(websocket, interval: float = 1.0):
    """Continuously stream sensor readings for all machines."""
    try:
        while True:
            readings = [_generate_reading(m) for m in MACHINES]
            await websocket.send_text(json.dumps(readings))
            await asyncio.sleep(interval)
    except Exception:
        pass  # client disconnected
