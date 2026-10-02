"""
LangGraph Maintenance Copilot Agent.
Orchestrates anomaly intake, SHAP local explanation, RAG manual retrieval,
confidence-gated novelty evaluation, and automated work order drafting.
"""

import os
import json
import logging
import datetime
from typing import Dict, Any, List, Optional

from langgraph.graph import StateGraph, END

from backend.config import get_settings
from backend.copilot.state import CopilotState
from backend.copilot.rag_engine import get_rag_indexer, NOVELTY_SIMILARITY_THRESHOLD
from backend.copilot.shap_explainer import explain_reading_shap
from backend.health_score import compute_health_score
from backend import ml_engine

logger = logging.getLogger(__name__)


# ── LangGraph Node Implementations ───────────────────────────────────────────

def intake_telemetry_node(state: CopilotState) -> Dict[str, Any]:
    """Node 1: Ingests sensor telemetry, assesses current health score and risk level."""
    telemetry = state.get("telemetry", {})
    asset_tag = state.get("asset_tag") or telemetry.get("asset_tag", "ASSET-001")
    machine_type = state.get("machine_type") or telemetry.get("machine_type", "CNC Lathe")

    # Compute health score if missing
    hs_res = compute_health_score(telemetry)
    health_score = state.get("health_score", hs_res.health_score)
    health_status = state.get("health_status", hs_res.status)

    # Compute ML prediction / risk if available
    prob = state.get("breakdown_probability", 0.0)
    anomaly_score = state.get("anomaly_score", 0.0)
    risk_level = state.get("risk_level", "NORMAL")

    if ml_engine.model_exists() and prob == 0.0:
        try:
            pred = ml_engine.predict_single(telemetry)
            prob = pred.get("probability", 0.0)
            risk_level = pred.get("risk_level", "NORMAL")
            anomaly_score = pred.get("anomaly_score", 0.0)
        except Exception as e:
            logger.warning(f"Prediction during copilot intake: {e}")

    step_trace = {
        "step": 1,
        "node": "Intake Telemetry & Health Assessment",
        "timestamp": datetime.datetime.now().isoformat(),
        "summary": f"Ingested telemetry for {asset_tag} ({machine_type}). Health Score: {health_score:.1f} ({health_status}), Risk: {risk_level}, Anomaly Score: {anomaly_score:.2f}.",
        "status": "COMPLETED",
    }

    traces = list(state.get("graph_trace", []))
    traces.append(step_trace)

    return {
        "asset_tag": asset_tag,
        "machine_type": machine_type,
        "health_score": health_score,
        "health_status": health_status,
        "breakdown_probability": prob,
        "risk_level": risk_level,
        "anomaly_score": anomaly_score,
        "graph_trace": traces,
    }


def shap_explainer_node(state: CopilotState) -> Dict[str, Any]:
    """Node 2: Extracts local SHAP feature contributions to identify failure drivers."""
    telemetry = state.get("telemetry", {})
    shap_res = explain_reading_shap(telemetry)

    top_features = shap_res["top_features"]
    shap_summary = shap_res["summary"]
    symptoms = shap_res["symptoms_text"]

    step_trace = {
        "step": 2,
        "node": "SHAP Explainability & Root Attribution",
        "timestamp": datetime.datetime.now().isoformat(),
        "summary": f"SHAP identified top failure drivers: {', '.join([f['display_name'] for f in top_features[:3]])}.",
        "status": "COMPLETED",
    }

    traces = list(state.get("graph_trace", []))
    traces.append(step_trace)

    return {
        "shap_top_features": top_features,
        "shap_summary": shap_summary,
        "anomaly_symptoms": symptoms,
        "graph_trace": traces,
    }


def rag_retrieval_node(state: CopilotState) -> Dict[str, Any]:
    """Node 3: Retrieves relevant equipment manual SOP sections from vector store."""
    machine_type = state.get("machine_type", "CNC Lathe")
    top_features = state.get("shap_top_features", [])
    telemetry    = state.get("telemetry", {})

    # Construct targeted RAG query including raw sensor values so extreme
    # out-of-envelope readings produce lower similarity (novelty detection)
    feature_names = " ".join([f["display_name"] for f in top_features])
    symptoms      = state.get("anomaly_symptoms", "")

    # Append raw sensor values: extreme values diverge from manual vocabulary
    sensor_value_str = " ".join([
        f"{k}={v:.1f}"
        for k, v in telemetry.items()
        if isinstance(v, (int, float)) and k not in ("asset_tag", "machine_type")
    ])
    query_text = f"{machine_type} failure {feature_names} {symptoms} {sensor_value_str}"

    indexer = get_rag_indexer()
    rag_result = indexer.search_manuals(
        machine_type=machine_type,
        query=query_text,
        top_k=3,
        min_threshold=NOVELTY_SIMILARITY_THRESHOLD,
    )

    step_trace = {
        "step": 3,
        "node": "RAG Manual & SOP Vector Retrieval",
        "timestamp": datetime.datetime.now().isoformat(),
        "summary": f"Retrieved {len(rag_result['retrieved_docs'])} SOP manual sections for {machine_type}. Top similarity: {rag_result['top_similarity']:.2f}.",
        "status": "COMPLETED",
    }

    traces = list(state.get("graph_trace", []))
    traces.append(step_trace)

    return {
        "query_text": query_text,
        "retrieved_manuals": rag_result["retrieved_docs"],
        "top_similarity": rag_result["top_similarity"],
        "is_novel_anomaly": rag_result["is_novel_anomaly"],
        "novelty_reason": rag_result["novelty_reason"],
        "graph_trace": traces,
    }


def confidence_gate_node(state: CopilotState) -> Dict[str, Any]:
    """Node 4: Computes confidence score and assigns escalation tier."""
    top_sim      = state.get("top_similarity", 0.0)
    is_novel     = state.get("is_novel_anomaly", False)   # from RAG node
    health_score = state.get("health_score", 100.0)
    risk_level   = state.get("risk_level", "NORMAL")
    top_features = state.get("shap_top_features", [])

    # Base confidence = RAG similarity boosted slightly
    base_confidence = min(1.0, top_sim * 1.15)

    # --- Novelty promotion rules ---
    # 1. Severe health drop with weak SOP match
    if health_score < 45 and top_sim < 0.60:
        is_novel = True
        base_confidence = max(0.20, base_confidence * 0.70)

    # 2. Multiple extreme SHAP drivers with weak RAG coverage
    extreme_drivers = sum(1 for f in top_features if abs(f.get("shap_impact", 0)) > 0.3)
    if extreme_drivers >= 2 and top_sim < 0.65:
        is_novel = True
        base_confidence = max(0.20, base_confidence * 0.80)

    # 3. Any RAG-flagged novelty is always honoured
    # (is_novel already True from RAG node if sim < NOVELTY_SIMILARITY_THRESHOLD)

    confidence_score = round(max(0.15, min(0.98, base_confidence)), 2)

    # Determine Escalation Tier
    if is_novel or confidence_score < 0.55:
        escalation_tier = "LEVEL_3_OEM_SPECIALIST"
        escalation_reason = (
            "Novel / Unprecedented Failure Signature: Telemetry divergence is not covered by "
            "standard OEM manual SOPs. Direct Level 3 escalation to Senior Reliability Lead & OEM Field Specialist."
        )
    elif confidence_score < 0.75 or risk_level == "CRITICAL":
        escalation_tier = "LEVEL_2_SENIOR_TECH"
        escalation_reason = (
            "Elevated Degradation / Moderate SOP Coverage: Standard procedure applies with mandatory "
            "Senior Technician supervision and precision clearance verification."
        )
    else:
        escalation_tier = "LEVEL_1_STANDARD_SOP"
        escalation_reason = (
            "High SOP Match Confidence: Standard Operating Procedure verified against OEM manual. "
            "Autonomous Level 1 Work Order dispatch to floor technician."
        )

    step_trace = {
        "step": 4,
        "node": "Confidence-Gated Escalation & Novelty Assessment",
        "timestamp": datetime.datetime.now().isoformat(),
        "summary": f"Confidence Score: {confidence_score:.0%}. Assigned Tier: {escalation_tier}. Novel anomaly: {is_novel}.",
        "status": "COMPLETED",
    }

    traces = list(state.get("graph_trace", []))
    traces.append(step_trace)

    return {
        "confidence_score": confidence_score,
        "escalation_tier": escalation_tier,
        "escalation_reason": escalation_reason,
        "is_novel_anomaly": is_novel,
        "graph_trace": traces,
    }


def work_order_drafter_node(state: CopilotState) -> Dict[str, Any]:
    """Node 5: Drafts industrial work order in natural language and structured format."""
    asset_tag = state.get("asset_tag", "ASSET-001")
    machine_type = state.get("machine_type", "CNC Lathe")
    health_score = state.get("health_score", 65.0)
    prob = state.get("breakdown_probability", 0.5)
    risk_level = state.get("risk_level", "HIGH")
    anomaly_score = state.get("anomaly_score", 0.0)
    shap_summary = state.get("shap_summary", "")
    top_features = state.get("shap_top_features", [])
    retrieved = state.get("retrieved_manuals", [])
    confidence = state.get("confidence_score", 0.85)
    escalation_tier = state.get("escalation_tier", "LEVEL_1_STANDARD_SOP")
    escalation_reason = state.get("escalation_reason", "")
    is_novel = state.get("is_novel_anomaly", False)

    # Generate unique Work Order ID
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    import random
    wo_id = f"WO-{date_str}-{asset_tag.replace('-', '')}-{random.randint(100, 999)}"
    
    # Priority determination
    priority = "P1 - CRITICAL EMERGENCY" if risk_level == "CRITICAL" or health_score < 40 else (
        "P2 - URGENT INTERVENTION" if risk_level == "HIGH" or health_score < 65 else "P3 - PREVENTIVE INSPECTION"
    )

    # Primary matched manual
    top_doc = retrieved[0] if retrieved else {
        "manual_id": "GEN-SOP-000",
        "title": "General Machine Inspection Procedure",
        "subsystem": "General Mechanics",
        "sop_steps": ["1. Conduct multi-point visual inspection.", "2. Check lubrication levels and thermal equilibrium."],
        "safety_protocol": "Standard LOTO Isolation.",
        "required_parts": ["General Fasteners & Seal Kit"],
        "estimated_mttr_hours": 2.0,
        "tolerances": "Standard OEM limits",
    }

    # Draft structured work order
    structured_wo = {
        "work_order_id": wo_id,
        "created_at": datetime.datetime.now().isoformat(),
        "asset_tag": asset_tag,
        "machine_type": machine_type,
        "priority": priority,
        "health_score": health_score,
        "breakdown_risk": risk_level,
        "breakdown_probability_pct": round(prob * 100, 1),
        "confidence_score_pct": round(confidence * 100, 1),
        "escalation_tier": escalation_tier,
        "is_novel_anomaly": is_novel,
        "matched_sop_id": top_doc.get("manual_id"),
        "matched_sop_title": top_doc.get("title"),
        "subsystem": top_doc.get("subsystem"),
        "estimated_mttr_hours": top_doc.get("estimated_mttr_hours", 2.5),
        "required_parts": top_doc.get("required_parts", []),
        "clearance_tolerances": top_doc.get("tolerances", "Refer to OEM manual"),
        "safety_protocol": top_doc.get("safety_protocol", "Standard LOTO Isolation"),
        "recommended_actions": top_doc.get("sop_steps", []),
        "shap_drivers": [f["display_name"] + f" ({f['shap_impact']:+.2f})" for f in top_features],
    }

    # Build detailed Markdown Work Order
    escalation_badge = ""
    if escalation_tier == "LEVEL_3_OEM_SPECIALIST":
        escalation_badge = "🚨 **LEVEL 3 ESCALATION REQUIRED — NOVEL ANOMALY DETECTED (OEM / Senior Specialist Dispatch)**"
    elif escalation_tier == "LEVEL_2_SENIOR_TECH":
        escalation_badge = "⚠️ **LEVEL 2 ESCALATION — SENIOR TECHNICIAN SUPERVISION MANDATED**"
    else:
        escalation_badge = "✅ **LEVEL 1 STANDARD SOP EXECUTION — CONFIDENCE GATED (Autonomous Dispatch)**"

    sop_steps_formatted = "\n".join([f"- {s}" for s in top_doc.get("sop_steps", [])])
    parts_formatted = "\n".join([f"- {p}" for p in top_doc.get("required_parts", [])])

    markdown_wo = f"""# 📋 INDUSTRIAL MAINTENANCE WORK ORDER: {wo_id}
**Asset Tag:** `{asset_tag}` | **Machine Type:** `{machine_type}` | **Priority:** `{priority}`
**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} | **Confidence Score:** `{confidence:.0%}`

---

### 🛡️ Escalation & Triage Status
{escalation_badge}
> **Routing Rationale:** {escalation_reason}

---

### 🔍 1. Anomaly Summary & Sensor Telemetry
- **Equipment Health Index:** `{health_score:.1f} / 100` (Degradation State: `{risk_level}`)
- **Calculated Failure Probability:** `{prob:.1%}` | **Anomaly Severity:** `{anomaly_score:.2f}`
- **Root-Cause Telemetry Drivers (SHAP Local Attribution):**
{chr(10).join([f"  - **{f['display_name']}:** Value = `{f['raw_value']}`, SHAP Impact = `{f['shap_impact']:+.3f}` ({f['direction']})" for f in top_features])}

---

### 📚 2. Cross-Referenced OEM Manual & SOP Reference
- **Matched Manual ID:** `{top_doc.get('manual_id')}` ({top_doc.get('title')})
- **Subsystem Focus:** `{top_doc.get('subsystem')}`
- **Clearance & Torque Tolerances:** `{top_doc.get('tolerances')}`
- **RAG Vector Match Similarity:** `{state.get('top_similarity', 0.0):.2f}`

---

### 🔧 3. Standard Operating Procedure (Remediation Steps)
{sop_steps_formatted}

---

### ⚠️ 4. Safety & Lockout/Tagout (LOTO) Protocol
> **CRITICAL SAFETY DIRECTIVE:** {top_doc.get('safety_protocol')}

---

### 📦 5. Required Spare Parts & Tooling (BOM)
{parts_formatted}

- **Estimated MTTR (Mean Time to Repair):** `{top_doc.get('estimated_mttr_hours', 2.5)} Hours`
- **Assigned Technician / Tier:** `{escalation_tier}`
"""

    step_trace = {
        "step": 5,
        "node": "Work Order Drafting & Knowledge Synthesis",
        "timestamp": datetime.datetime.now().isoformat(),
        "summary": f"Drafted Work Order {wo_id}. Priority: {priority}, Estimated MTTR: {top_doc.get('estimated_mttr_hours')}h.",
        "status": "COMPLETED",
    }

    traces = list(state.get("graph_trace", []))
    traces.append(step_trace)

    return {
        "work_order_id": wo_id,
        "work_order_title": f"Maintenance Work Order {wo_id} ({machine_type})",
        "work_order_markdown": markdown_wo,
        "work_order_structured": structured_wo,
        "graph_trace": traces,
    }


def escalation_router_node(state: CopilotState) -> Dict[str, Any]:
    """Node 6: Handles notification broadcast, alerts, and OT write-back triggers."""
    escalation_tier = state.get("escalation_tier", "LEVEL_1_STANDARD_SOP")
    wo_id = state.get("work_order_id", "WO-UNKNOWN")
    asset_tag = state.get("asset_tag", "ASSET-001")

    # Record notification log
    notification_msg = (
        f"Work Order {wo_id} routed to {escalation_tier} for Asset {asset_tag}."
    )
    logger.info(notification_msg)

    step_trace = {
        "step": 6,
        "node": "Escalation Routing & OT Notification Dispatch",
        "timestamp": datetime.datetime.now().isoformat(),
        "summary": notification_msg,
        "status": "COMPLETED",
    }

    traces = list(state.get("graph_trace", []))
    traces.append(step_trace)

    return {
        "graph_trace": traces,
    }


# ── LangGraph Workflow Construction ──────────────────────────────────────────

def build_maintenance_copilot_graph():
    """Constructs and compiles the stateful LangGraph Copilot workflow."""
    workflow = StateGraph(CopilotState)

    # Register Nodes
    workflow.add_node("intake_telemetry", intake_telemetry_node)
    workflow.add_node("shap_explainer", shap_explainer_node)
    workflow.add_node("rag_retrieval", rag_retrieval_node)
    workflow.add_node("confidence_gate", confidence_gate_node)
    workflow.add_node("work_order_drafter", work_order_drafter_node)
    workflow.add_node("escalation_router", escalation_router_node)

    # Set Entry Point
    workflow.set_entry_point("intake_telemetry")

    # Connect Edges
    workflow.add_edge("intake_telemetry", "shap_explainer")
    workflow.add_edge("shap_explainer", "rag_retrieval")
    workflow.add_edge("rag_retrieval", "confidence_gate")
    workflow.add_edge("confidence_gate", "work_order_drafter")
    workflow.add_edge("work_order_drafter", "escalation_router")
    workflow.add_edge("escalation_router", END)

    app = workflow.compile()
    return app


# Compiled graph singleton
_copilot_app = None


def get_copilot_graph():
    global _copilot_app
    if _copilot_app is None:
        _copilot_app = build_maintenance_copilot_graph()
    return _copilot_app


# In-memory work order cache for history display
WORK_ORDERS_HISTORY: List[Dict[str, Any]] = []


def run_maintenance_copilot(
    telemetry: Dict[str, Any],
    trigger_source: str = "manual_trigger"
) -> Dict[str, Any]:
    """
    Executes the LangGraph Maintenance Copilot on given telemetry.
    Returns complete state including work order, SHAP drivers, RAG docs, and trace.
    """
    graph = get_copilot_graph()

    initial_state: CopilotState = {
        "telemetry": telemetry,
        "asset_tag": telemetry.get("asset_tag", "CNC-001"),
        "machine_type": telemetry.get("machine_type", "CNC Lathe"),
        "trigger_source": trigger_source,
        "graph_trace": [],
    }

    result_state = graph.invoke(initial_state)

    # Save to history cache
    if "work_order_structured" in result_state:
        wo_record = result_state["work_order_structured"]
        WORK_ORDERS_HISTORY.insert(0, wo_record)
        # Keep recent 100
        if len(WORK_ORDERS_HISTORY) > 100:
            WORK_ORDERS_HISTORY.pop()

    return result_state
