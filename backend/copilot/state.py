"""
State schema for LangGraph Maintenance Copilot.
"""

from typing import TypedDict, Optional, Any, List, Dict


class CopilotState(TypedDict, total=False):
    # Input Telemetry
    asset_tag: str
    machine_type: str
    telemetry: Dict[str, Any]
    health_score: float
    health_status: str
    breakdown_probability: float
    risk_level: str
    anomaly_score: float
    trigger_source: str

    # SHAP Explanations
    shap_top_features: List[Dict[str, Any]]
    shap_summary: str
    anomaly_symptoms: str

    # RAG Retrieval
    query_text: str
    retrieved_manuals: List[Dict[str, Any]]
    top_similarity: float
    is_novel_anomaly: bool
    novelty_reason: Optional[str]

    # Confidence & Escalation
    confidence_score: float
    escalation_tier: str  # "LEVEL_1_STANDARD_SOP", "LEVEL_2_SENIOR_TECH", "LEVEL_3_OEM_SPECIALIST"
    escalation_reason: Optional[str]

    # Work Order Output
    work_order_id: str
    work_order_title: str
    work_order_markdown: str
    work_order_structured: Dict[str, Any]

    # LangGraph Execution Trace
    graph_trace: List[Dict[str, Any]]
    error: Optional[str]
