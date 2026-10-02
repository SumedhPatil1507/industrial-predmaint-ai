"""Quick integration smoke-test for the Maintenance Copilot pipeline."""
from backend.copilot import (
    run_maintenance_copilot,
    get_rag_indexer,
    get_all_manuals,
    WORK_ORDERS_HISTORY,
)
from frontend.charts import (
    copilot_langgraph_flow_chart,
    copilot_shap_waterfall_chart,
    copilot_confidence_gauge,
    copilot_radar_chart,
    copilot_retrieval_similarity_chart,
)

PASS = []
FAIL = []

def check(label, condition, detail=""):
    if condition:
        PASS.append(label)
        print(f"  OK  {label}")
    else:
        FAIL.append(label)
        print(f"FAIL  {label}  {detail}")

# --- Copilot agent ---
t1 = {"asset_tag": "CNC-001", "machine_type": "CNC Lathe",
      "temp_bearing_degC": 88.5, "temp_motor_degC": 78.0,
      "vibration_h_mms": 7.2,   "vibration_v_mms": 3.5,
      "oil_pressure_bar": 5.2,  "load_pct": 82.0,
      "shaft_rpm": 1800.0,      "power_consumption_kw": 34.0}
r1 = run_maintenance_copilot(t1)

check("WO ID generated",        bool(r1.get("work_order_id")))
check("6 graph nodes traced",   len(r1.get("graph_trace", [])) == 6,
      f"got {len(r1.get('graph_trace', []))}")
check("SHAP top features",      len(r1.get("shap_top_features", [])) > 0)
check("Retrieved manuals",      len(r1.get("retrieved_manuals", [])) > 0)
check("Work order markdown",    len(r1.get("work_order_markdown", "")) > 100)
check("Structured WO dict",     isinstance(r1.get("work_order_structured"), dict))
check("Escalation tier set",    r1.get("escalation_tier") in (
    "LEVEL_1_STANDARD_SOP", "LEVEL_2_SENIOR_TECH", "LEVEL_3_OEM_SPECIALIST"))

# Novel anomaly
t2 = {"asset_tag": "CNC-002", "machine_type": "CNC Lathe",
      "temp_bearing_degC": 104.0, "temp_motor_degC": 52.0,
      "vibration_h_mms": 14.5,   "vibration_v_mms": 12.8,
      "oil_pressure_bar": 8.9,   "load_pct": 30.0,
      "shaft_rpm": 350.0,        "power_consumption_kw": 58.0}
r2 = run_maintenance_copilot(t2)
check("Novel anomaly flagged",  r2.get("is_novel_anomaly") is True)
check("L3 escalation for novel", r2.get("escalation_tier") == "LEVEL_3_OEM_SPECIALIST")

# RAG
idx  = get_rag_indexer()
res  = idx.search_manuals("CNC Lathe", "spindle bearing overheat vibration", top_k=3)
check("RAG similarity > 0",     res["top_similarity"] > 0)
check("RAG returns 3 docs",     len(res["retrieved_docs"]) == 3)

all_m = get_all_manuals()
check("11 manuals indexed",     len(all_m) == 11)
check("5 machine types",        len(set(m["machine_type"] for m in all_m)) == 5)
check("History populated",      len(WORK_ORDERS_HISTORY) >= 2)

# Charts
mock_trace = [
    {"step": i+1, "node": f"Node {i+1}", "status": "COMPLETED",
     "timestamp": "2026-10-02T10:00:00", "summary": "done"}
    for i in range(6)
]
check("LangGraph flow chart",   len(copilot_langgraph_flow_chart(mock_trace).data) > 0)

mock_feats = [
    {"display_name": "Bearing Temp", "shap_impact":  0.42, "raw_value": 88.5, "direction": "elevated"},
    {"display_name": "H-Vibration",  "shap_impact":  0.31, "raw_value": 7.2,  "direction": "elevated"},
    {"display_name": "Oil Pressure", "shap_impact": -0.12, "raw_value": 5.2,  "direction": "reduced"},
]
check("SHAP waterfall chart",   len(copilot_shap_waterfall_chart(mock_feats).data) > 0)
check("Confidence gauge",       len(copilot_confidence_gauge(0.82).data) > 0)
check("Confidence gauge novel", len(copilot_confidence_gauge(0.31, is_novel=True).data) > 0)

mock_docs = [
    {"manual_id": "CNC-SOP-001", "title": "Spindle Bearing Overheating",
     "machine_type": "CNC Lathe", "subsystem": "Spindle",
     "similarity_score": 0.72, "estimated_mttr_hours": 3.5},
    {"manual_id": "CNC-SOP-002", "title": "Drive Motor Thermal Overload",
     "machine_type": "CNC Lathe", "subsystem": "Drive",
     "similarity_score": 0.48, "estimated_mttr_hours": 2.0},
]
check("RAG similarity chart",   len(copilot_retrieval_similarity_chart(mock_docs, 0.52).data) > 0)

radar_telem = {"temp_bearing_degC": 88.5, "vibration_h_mms": 7.2,
               "oil_pressure_bar": 5.2, "load_pct": 82.0,
               "shaft_rpm": 1800.0, "power_consumption_kw": 34.0,
               "temp_motor_degC": 78.0, "vibration_v_mms": 3.5}
check("Radar chart 3 traces",   len(copilot_radar_chart(radar_telem, "CNC Lathe").data) == 3)

# Summary
print()
print(f"Results: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    raise SystemExit(1)
