"""Quick API smoke test. Run after starting the backend."""
import httpx
import json

BASE = "http://localhost:8000"

def test(name, fn):
    try:
        result = fn()
        print(f"✅ {name}: {json.dumps(result, indent=2)[:200]}")
    except Exception as e:
        print(f"❌ {name}: {e}")

test("Health", lambda: httpx.get(f"{BASE}/health").json())

test("Single Predict", lambda: httpx.post(f"{BASE}/predict", json={
    "asset_tag": "CNC-001",
    "machine_type": "CNC Lathe",
    "temp_bearing_degC": 82.0,
    "temp_motor_degC": 91.0,
    "vibration_h_mms": 6.5,
    "vibration_v_mms": 5.2,
    "oil_pressure_bar": 4.1,
    "load_pct": 88.0,
    "shaft_rpm": 1650.0,
    "power_consumption_kw": 38.0,
}, timeout=30).json())

test("Downtime Calc", lambda: httpx.post(f"{BASE}/downtime-calculator", json={
    "machine_type": "CNC Lathe",
    "annual_breakdowns": 15,
}).json())
