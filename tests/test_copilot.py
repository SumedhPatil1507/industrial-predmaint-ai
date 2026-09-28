"""
Comprehensive unit and integration tests for Maintenance Copilot (LangGraph + RAG + SHAP).
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.copilot.manuals_data import get_all_manuals, get_manuals_for_machine
from backend.copilot.rag_engine import get_rag_indexer, NOVELTY_SIMILARITY_THRESHOLD
from backend.copilot.shap_explainer import explain_reading_shap
from backend.copilot.agent import run_maintenance_copilot, get_copilot_graph


# ── Equipment Manuals & SOP Dataset ──────────────────────────────────────────

def test_equipment_manuals_coverage():
    manuals = get_all_manuals()
    assert len(manuals) >= 8
    
    machine_types = {m["machine_type"] for m in manuals}
    assert "CNC Lathe" in machine_types
    assert "Hydraulic Press" in machine_types
    assert "Milling Machine" in machine_types
    assert "Industrial Boiler" in machine_types
    assert "Centrifugal Pump" in machine_types

    for m in manuals:
        assert "manual_id" in m
        assert "sop_steps" in m and len(m["sop_steps"]) >= 3
        assert "safety_protocol" in m
        assert "required_parts" in m and len(m["required_parts"]) > 0
        assert m["estimated_mttr_hours"] > 0


def test_manuals_filter_by_machine():
    cnc_manuals = get_manuals_for_machine("CNC Lathe")
    assert len(cnc_manuals) >= 2
    for m in cnc_manuals:
        assert m["machine_type"] == "CNC Lathe"


# ── RAG Vector Store & Novelty Gating ────────────────────────────────────────

def test_rag_semantic_search_standard_sop():
    indexer = get_rag_indexer()
    res = indexer.search_manuals(
        machine_type="CNC Lathe",
        query="spindle bearing overheating high horizontal vibration grease lubrication",
        top_k=2
    )
    assert len(res["retrieved_docs"]) > 0
    assert res["top_similarity"] >= NOVELTY_SIMILARITY_THRESHOLD
    assert res["is_novel_anomaly"] is False
    assert "CNC-SOP-001" in [d["manual_id"] for d in res["retrieved_docs"]]


def test_rag_novel_anomaly_detection():
    indexer = get_rag_indexer()
    res = indexer.search_manuals(
        machine_type="CNC Lathe",
        query="completely bizarre quantum radioactive magnetic anomaly unknown plasma fluctuation",
        top_k=2,
        min_threshold=0.60
    )
    assert res["is_novel_anomaly"] is True
    assert res["novelty_reason"] is not None


# ── SHAP Local Feature Explanations ──────────────────────────────────────────

def test_shap_explanation_structure():
    telemetry = {
        "asset_tag": "CNC-001",
        "machine_type": "CNC Lathe",
        "temp_bearing_degC": 92.0,
        "temp_motor_degC": 75.0,
        "vibration_h_mms": 7.5,
        "vibration_v_mms": 3.0,
        "oil_pressure_bar": 5.0,
        "load_pct": 85.0,
        "shaft_rpm": 1800.0,
        "power_consumption_kw": 35.0,
    }
    exp = explain_reading_shap(telemetry)
    assert "all_contributions" in exp
    assert "top_features" in exp
    assert len(exp["top_features"]) >= 3
    assert "summary" in exp
    assert any("temp_bearing_degC" == f["feature"] or "vibration_h_mms" == f["feature"] for f in exp["top_features"])


# ── LangGraph Maintenance Copilot Agent ──────────────────────────────────────

def test_copilot_standard_triage():
    telemetry = {
        "asset_tag": "CNC-001",
        "machine_type": "CNC Lathe",
        "temp_bearing_degC": 88.0,
        "temp_motor_degC": 78.0,
        "vibration_h_mms": 7.0,
        "vibration_v_mms": 3.2,
        "oil_pressure_bar": 5.2,
        "load_pct": 80.0,
        "shaft_rpm": 1800.0,
        "power_consumption_kw": 32.0,
    }
    result = run_maintenance_copilot(telemetry, trigger_source="unit_test")
    
    assert "work_order_id" in result
    assert result["work_order_id"].startswith("WO-")
    assert "work_order_markdown" in result
    assert "work_order_structured" in result
    assert result["confidence_score"] > 0
    assert len(result["graph_trace"]) == 6
    assert result["escalation_tier"] in ("LEVEL_1_STANDARD_SOP", "LEVEL_2_SENIOR_TECH")


def test_copilot_novel_anomaly_escalation():
    # Extreme aberrant signature
    telemetry = {
        "asset_tag": "CNC-002",
        "machine_type": "CNC Lathe",
        "temp_bearing_degC": 125.0,
        "temp_motor_degC": 40.0,
        "vibration_h_mms": 16.5,
        "vibration_v_mms": 15.0,
        "oil_pressure_bar": 9.5,
        "load_pct": 20.0,
        "shaft_rpm": 250.0,
        "power_consumption_kw": 60.0,
    }
    result = run_maintenance_copilot(telemetry, trigger_source="unit_test_novel")
    
    assert result["work_order_id"].startswith("WO-")
    assert result["escalation_tier"] in ("LEVEL_2_SENIOR_TECH", "LEVEL_3_OEM_SPECIALIST")
    assert result["health_score"] < 50
    assert len(result["graph_trace"]) == 6
    wo_struct = result["work_order_structured"]
    assert wo_struct["priority"].startswith("P1") or wo_struct["priority"].startswith("P2")
