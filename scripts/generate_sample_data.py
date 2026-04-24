"""
Generate a synthetic industrial machine dataset (219,000 rows).
Run: python scripts/generate_sample_data.py
Output: data/synthetic_industrial_machine_data.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

Path("data").mkdir(exist_ok=True)

MACHINES = [
    ("CNC-001", "CNC Lathe"),
    ("CNC-002", "CNC Lathe"),
    ("HYD-001", "Hydraulic Press"),
    ("HYD-002", "Hydraulic Press"),
    ("BLT-001", "Belt Conveyor"),
    ("BLT-002", "Belt Conveyor"),
    ("CMP-001", "Screw Compressor"),
    ("CMP-002", "Screw Compressor"),
    ("EOT-001", "EOT Crane"),
    ("EOT-002", "EOT Crane"),
]

START = datetime(2022, 1, 3)
END = datetime(2025, 1, 1)
DAYS = (END - START).days  # ~1094 days

CRITICALITY = {"CNC Lathe": "A", "Hydraulic Press": "A",
               "Belt Conveyor": "B", "Screw Compressor": "B", "EOT Crane": "C"}

RANGES = {
    "CNC Lathe":        dict(tb=(55,75), tm=(60,85), vh=(1.5,4.0), vv=(1.0,3.5), op=(4.5,6.5), lp=(40,90), rpm=(800,2000), pw=(15,35)),
    "Hydraulic Press":  dict(tb=(50,70), tm=(55,80), vh=(0.5,2.5), vv=(0.5,2.0), op=(80,120), lp=(50,95), rpm=(200,600), pw=(30,80)),
    "Belt Conveyor":    dict(tb=(40,60), tm=(45,65), vh=(2.0,5.0), vv=(1.5,4.0), op=(2.0,4.0), lp=(30,80), rpm=(100,400), pw=(5,20)),
    "Screw Compressor": dict(tb=(60,85), tm=(65,95), vh=(1.0,3.5), vv=(0.8,3.0), op=(6.0,10.0), lp=(60,100), rpm=(1500,3000), pw=(20,55)),
    "EOT Crane":        dict(tb=(35,55), tm=(40,60), vh=(0.5,2.0), vv=(0.5,1.8), op=(3.0,5.0), lp=(20,70), rpm=(50,200), pw=(10,40)),
}

rows = []
for asset_tag, machine_type in MACHINES:
    r = RANGES[machine_type]
    degradation = 0.0
    for day_offset in range(DAYS):
        date = START + timedelta(days=day_offset)

        # Degradation cycle
        degradation += np.random.uniform(0, 0.003)
        if degradation > 1.0 or np.random.random() < 0.005:
            degradation = 0.0  # repair event

        deg = min(degradation, 1.0)

        def noisy(lo, hi, bias=0.0):
            return round(np.random.uniform(lo, hi) + np.random.normal(0, (hi-lo)*0.05) + bias*(hi-lo), 2)

        tb = noisy(*r["tb"], bias=deg*0.4)
        tm = noisy(*r["tm"], bias=deg*0.5)
        vh = max(0, noisy(*r["vh"], bias=deg*0.6))
        vv = max(0, noisy(*r["vv"], bias=deg*0.5))
        op = max(0, noisy(*r["op"], bias=-deg*0.3))
        lp = max(0, min(100, noisy(*r["lp"])))
        rpm = max(0, noisy(*r["rpm"], bias=-deg*0.1))
        pw = max(0, noisy(*r["pw"], bias=deg*0.2))

        # Breakdown probability increases with degradation
        bd_prob = 0.02 + deg * 0.25
        breakdown = int(np.random.random() < bd_prob)

        wo_type = None
        qty = 0
        issue_val = 0
        if breakdown:
            wo_type = "BD"
            qty = np.random.randint(1, 5)
            issue_val = round(np.random.uniform(5000, 50000), 2)
        elif np.random.random() < 0.05:
            wo_type = "PM"
            qty = np.random.randint(1, 3)
            issue_val = round(np.random.uniform(1000, 15000), 2)

        rows.append({
            "transaction_date": date.strftime("%Y-%m-%d"),
            "asset_tag": asset_tag,
            "machine_type": machine_type,
            "criticality": CRITICALITY[machine_type],
            "temp_bearing_degC": tb,
            "temp_motor_degC": tm,
            "vibration_h_mms": round(vh, 2),
            "vibration_v_mms": round(vv, 2),
            "oil_pressure_bar": op,
            "load_pct": round(lp, 1),
            "shaft_rpm": round(rpm, 0),
            "power_consumption_kw": pw,
            "qty_issued": qty,
            "issue_value_inr": issue_val,
            "wo_type": wo_type,
            "breakdown_flag": breakdown,
        })

df = pd.DataFrame(rows)
out = "data/synthetic_industrial_machine_data.csv"
df.to_csv(out, index=False)
print(f"✅ Generated {len(df):,} rows → {out}")
print(f"   Breakdown rate: {df['breakdown_flag'].mean():.1%}")
print(f"   Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")
