"""
Data Engine – generates live synthetic data in-memory.
No file upload needed. Works on Streamlit Cloud.
"""
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from collections import deque
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.iot_simulator import MACHINES, RANGES

# ── Synthetic historical dataset (generated once per session) ─────────────────

def generate_historical_data(n_days: int = 365) -> pd.DataFrame:
    """Generate n_days of synthetic sensor data for all 5 machines."""
    rows = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)

    for machine in MACHINES:
        tag   = machine["asset_tag"]
        mtype = machine["machine_type"]
        r     = RANGES[mtype]
        degradation = 0.0

        for day in range(n_days):
            date = start_date + timedelta(days=day)
            degradation += random.uniform(0, 0.004)
            if random.random() < 0.005:
                degradation = 0.0
            degradation = min(1.0, degradation)
            deg = degradation

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

            bd_prob = 0.02 + deg * 0.25
            breakdown = int(random.random() < bd_prob)

            wo_type = None
            qty, issue_val = 0, 0.0
            if breakdown:
                wo_type = "BD"
                qty = random.randint(1, 5)
                issue_val = round(random.uniform(5000, 50000), 2)
            elif random.random() < 0.05:
                wo_type = "PM"
                qty = random.randint(1, 3)
                issue_val = round(random.uniform(1000, 15000), 2)

            criticality = {"CNC Lathe":"A","Hydraulic Press":"A",
                           "Belt Conveyor":"B","Screw Compressor":"B","EOT Crane":"C"}[mtype]

            rows.append({
                "transaction_date": date.strftime("%Y-%m-%d"),
                "asset_tag": tag, "machine_type": mtype,
                "criticality": criticality,
                "temp_bearing_degC": tb, "temp_motor_degC": tm,
                "vibration_h_mms": round(vh,2), "vibration_v_mms": round(vv,2),
                "oil_pressure_bar": op, "load_pct": round(lp,1),
                "shaft_rpm": round(rpm,0), "power_consumption_kw": pw,
                "qty_issued": qty, "issue_value_inr": issue_val,
                "wo_type": wo_type, "breakdown_flag": breakdown,
            })

    df = pd.DataFrame(rows)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["is_breakdown"] = df["breakdown_flag"].map({0:"Normal",1:"Breakdown"})
    df["month"] = df["transaction_date"].dt.month
    df["day_of_week"] = df["transaction_date"].dt.dayofweek
    return df


# ── Live tick generator ───────────────────────────────────────────────────────

_sim_state = {m["asset_tag"]: {"degradation": 0.0, "tick": 0} for m in MACHINES}


def generate_live_tick() -> list[dict]:
    """Generate one tick of live sensor readings for all machines."""
    readings = []
    for machine in MACHINES:
        tag   = machine["asset_tag"]
        mtype = machine["machine_type"]
        r     = RANGES[mtype]
        s     = _sim_state[tag]
        s["tick"] += 1
        if random.random() < 0.003:
            s["degradation"] = 0.0
        s["degradation"] = min(1.0, s["degradation"] + random.uniform(0, 0.006))
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

        if random.random() < 0.01*(1+deg*3):
            vh *= random.uniform(1.5, 2.5)
            tb *= random.uniform(1.1, 1.3)

        readings.append({
            "timestamp": datetime.utcnow().isoformat(),
            "asset_tag": tag, "machine_type": mtype,
            "temp_bearing_degC": round(tb,2), "temp_motor_degC": round(tm,2),
            "vibration_h_mms": round(vh,2),   "vibration_v_mms": round(vv,2),
            "oil_pressure_bar": round(op,2),   "load_pct": round(lp,1),
            "shaft_rpm": round(rpm,0),         "power_consumption_kw": round(pw,2),
            "degradation_index": round(deg,3),
        })
    return readings


# ── What-if simulator ─────────────────────────────────────────────────────────

def what_if_sensitivity(base_reading: dict, feature: str,
                        pct_changes: list[float]) -> pd.DataFrame:
    """Vary one feature by % and return predicted probabilities."""
    from backend.ml_engine import predict_single
    rows = []
    base_val = base_reading.get(feature, 0)
    for pct in pct_changes:
        modified = dict(base_reading)
        modified[feature] = round(base_val * (1 + pct/100), 2)
        try:
            result = predict_single(modified)
            prob = result["probability"]
        except Exception:
            prob = 0.0
        rows.append({"change_pct": pct, "value": modified[feature], "probability": prob})
    return pd.DataFrame(rows)
