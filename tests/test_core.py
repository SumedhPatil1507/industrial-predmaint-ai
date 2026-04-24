"""Core unit tests – no external dependencies required."""
import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.downtime_calculator import calculate_downtime
from backend.health_score import compute_health_score
from backend.ttf_predictor import estimate_ttf
from backend.file_parser import parse_upload, validate_columns
from backend.model_registry import register_model, get_registry, compare_models


# ── Downtime Calculator ───────────────────────────────────────────────────────

def test_downtime_basic():
    r = calculate_downtime("CNC Lathe")
    assert r.total_cost_inr > 0
    assert r.total_cost_usd > 0
    assert r.savings_with_pdm_inr > 0


def test_downtime_custom_values():
    r = calculate_downtime("Hydraulic Press", mttr_hours=10,
                           hourly_production_inr=30000, repair_cost_inr=100000)
    assert r.production_loss_inr == 300000.0
    assert r.total_cost_inr == 400000.0


def test_downtime_roi_positive():
    r = calculate_downtime("CNC Lathe", annual_breakdowns=20, pdm_system_cost_inr=200000)
    assert r.roi_percent > 0


# ── Health Score ──────────────────────────────────────────────────────────────

def test_health_score_normal():
    reading = {
        "asset_tag": "CNC-001", "machine_type": "CNC Lathe",
        "temp_bearing_degC": 65.0, "temp_motor_degC": 72.0,
        "vibration_h_mms": 2.5, "vibration_v_mms": 2.0,
        "oil_pressure_bar": 5.5, "load_pct": 70.0,
        "shaft_rpm": 1200.0, "power_consumption_kw": 25.0,
    }
    r = compute_health_score(reading)
    assert r.health_score >= 70
    assert r.status == "Healthy"


def test_health_score_critical():
    reading = {
        "asset_tag": "CNC-001", "machine_type": "CNC Lathe",
        "temp_bearing_degC": 110.0, "temp_motor_degC": 120.0,
        "vibration_h_mms": 12.0, "vibration_v_mms": 10.0,
        "oil_pressure_bar": 1.0, "load_pct": 99.0,
        "shaft_rpm": 2500.0, "power_consumption_kw": 50.0,
    }
    r = compute_health_score(reading)
    assert r.health_score < 50
    assert r.status == "Critical"
    assert len(r.recommendations) > 1


# ── TTF Predictor ─────────────────────────────────────────────────────────────

def test_ttf_insufficient_data():
    df = pd.DataFrame({"asset_tag": ["A"] * 5,
                       "transaction_date": pd.date_range("2024-01-01", periods=5),
                       "vibration_h_mms": [2.0] * 5,
                       "temp_bearing_degC": [60.0] * 5})
    r = estimate_ttf(df, "A")
    assert r.confidence == "Low"


def test_ttf_degrading_machine():
    dates = pd.date_range("2024-01-01", periods=60)
    df = pd.DataFrame({
        "asset_tag": ["X"] * 60,
        "transaction_date": dates,
        "temp_bearing_degC": np.linspace(55, 95, 60),
        "temp_motor_degC": np.linspace(60, 100, 60),
        "vibration_h_mms": np.linspace(1.5, 8.0, 60),
        "vibration_v_mms": np.linspace(1.0, 6.0, 60),
        "oil_pressure_bar": np.linspace(6.0, 3.0, 60),
        "power_consumption_kw": np.linspace(20, 40, 60),
        "breakdown_flag": [0] * 55 + [1] * 5,
    })
    r = estimate_ttf(df, "X")
    assert r.estimated_days < 60
    assert r.urgency in ("Immediate", "This Week", "This Month", "Routine")


# ── File Parser ───────────────────────────────────────────────────────────────

def test_parse_csv():
    csv_bytes = b"a,b,c\n1,2,3\n4,5,6"
    df = parse_upload("test.csv", csv_bytes)
    assert df.shape == (2, 3)


def test_parse_json():
    import json
    data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    df = parse_upload("test.json", json.dumps(data).encode())
    assert len(df) == 2


def test_unsupported_format():
    with pytest.raises(ValueError):
        parse_upload("test.pdf", b"dummy")


def test_validate_columns_pass():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    ok, missing = validate_columns(df, ["a", "b"])
    assert ok and missing == []


def test_validate_columns_fail():
    df = pd.DataFrame({"a": [1]})
    ok, missing = validate_columns(df, ["a", "b", "c"])
    assert not ok
    assert "b" in missing and "c" in missing


# ── Model Registry ────────────────────────────────────────────────────────────

def test_register_and_retrieve(tmp_path, monkeypatch):
    import backend.model_registry as mr
    monkeypatch.setattr(mr, "REGISTRY_FILE", tmp_path / "registry.json")
    monkeypatch.setattr(mr, "MODEL_DIR", tmp_path)

    v = register_model(
        metrics={"accuracy": 0.95, "auc": 0.98},
        params={"n_estimators": 100},
        dataset_info={"rows": 1000},
    )
    assert v == "v1.0"
    registry = get_registry()
    assert len(registry) == 1
    assert registry[0]["is_active"] is True

    v2 = register_model(
        metrics={"accuracy": 0.96, "auc": 0.99},
        params={"n_estimators": 200},
        dataset_info={"rows": 2000},
    )
    assert v2 == "v2.0"
    registry = get_registry()
    assert registry[0]["is_active"] is False
    assert registry[1]["is_active"] is True
