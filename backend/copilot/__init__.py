"""Maintenance Copilot Package."""
from backend.copilot.agent import run_maintenance_copilot, get_copilot_graph, WORK_ORDERS_HISTORY
from backend.copilot.rag_engine import get_rag_indexer, NOVELTY_SIMILARITY_THRESHOLD
from backend.copilot.shap_explainer import explain_reading_shap
from backend.copilot.manuals_data import get_all_manuals, get_manuals_for_machine

__all__ = [
    "run_maintenance_copilot",
    "get_copilot_graph",
    "WORK_ORDERS_HISTORY",
    "get_rag_indexer",
    "NOVELTY_SIMILARITY_THRESHOLD",
    "explain_reading_shap",
    "get_all_manuals",
    "get_manuals_for_machine",
]
