"""
Model Registry – version, compare, and manage trained models.
Stores metadata for every training run so you can compare and rollback.
"""
import json
import joblib
from pathlib import Path
from datetime import datetime
import pandas as pd

REGISTRY_FILE = Path("models/registry.json")
MODEL_DIR = Path("models")


def _load_registry() -> list[dict]:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text())
    return []


def _save_registry(registry: list[dict]):
    MODEL_DIR.mkdir(exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2))


def register_model(metrics: dict, params: dict, dataset_info: dict) -> str:
    """Save a training run to the registry. Returns version string."""
    registry = _load_registry()
    version = f"v{len(registry) + 1}.0"
    entry = {
        "version": version,
        "trained_at": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "params": params,
        "dataset": dataset_info,
        "is_active": True,
    }
    # Deactivate previous
    for r in registry:
        r["is_active"] = False
    registry.append(entry)
    _save_registry(registry)
    return version


def get_registry() -> list[dict]:
    return _load_registry()


def get_active_version() -> str:
    registry = _load_registry()
    for r in reversed(registry):
        if r.get("is_active"):
            return r["version"]
    return "v1.0"


def compare_models() -> pd.DataFrame:
    registry = _load_registry()
    if not registry:
        return pd.DataFrame()
    rows = []
    for r in registry:
        m = r.get("metrics", {})
        rows.append({
            "version": r["version"],
            "trained_at": r["trained_at"][:10],
            "accuracy": round(m.get("accuracy", 0), 4),
            "auc": round(m.get("auc", 0), 4),
            "precision": round(m.get("precision_breakdown", 0), 4),
            "recall": round(m.get("recall_breakdown", 0), 4),
            "f1": round(m.get("f1_breakdown", 0), 4),
            "n_train": m.get("n_train", 0),
            "active": r.get("is_active", False),
        })
    return pd.DataFrame(rows)
