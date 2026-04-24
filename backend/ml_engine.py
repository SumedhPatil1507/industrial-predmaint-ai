"""ML engine: train, predict, SHAP, anomaly detection."""
import numpy as np
import pandas as pd
import joblib
import shap
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import logging

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "temp_bearing_degC", "temp_motor_degC",
    "vibration_h_mms", "vibration_v_mms",
    "oil_pressure_bar", "load_pct",
    "shaft_rpm", "power_consumption_kw",
]

TARGET = "breakdown_flag"


# ── Feature engineering ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temp_diff"] = df["temp_motor_degC"] - df["temp_bearing_degC"]
    df["vibration_total"] = np.sqrt(df["vibration_h_mms"] ** 2 + df["vibration_v_mms"] ** 2)
    df["power_per_load"] = df["power_consumption_kw"] / (df["load_pct"].replace(0, 1))
    if "transaction_date" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        df["month"] = df["transaction_date"].dt.month
        df["day_of_week"] = df["transaction_date"].dt.dayofweek
    return df


def get_all_features(df: pd.DataFrame) -> list[str]:
    base = FEATURE_COLS.copy()
    extra = ["temp_diff", "vibration_total", "power_per_load"]
    optional = ["month", "day_of_week"]
    cols = base + extra + [c for c in optional if c in df.columns]
    return [c for c in cols if c in df.columns]


# ── Training ─────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame) -> dict:
    df = engineer_features(df)
    feat_cols = get_all_features(df)

    X = df[feat_cols].fillna(df[feat_cols].median())
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)

    # Anomaly detector on normal data only
    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X_train[y_train == 0])

    scaler = StandardScaler()
    scaler.fit(X_train)

    joblib.dump(rf, MODEL_DIR / "rf_model.pkl")
    joblib.dump(iso, MODEL_DIR / "iso_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(feat_cols, MODEL_DIR / "feature_cols.pkl")

    logger.info(f"Model trained. AUC={auc:.4f}")
    return {
        "accuracy": report["accuracy"],
        "auc": auc,
        "precision_breakdown": report.get("1", {}).get("precision", 0),
        "recall_breakdown": report.get("1", {}).get("recall", 0),
        "f1_breakdown": report.get("1", {}).get("f1-score", 0),
        "feature_cols": feat_cols,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# ── Inference ─────────────────────────────────────────────────────────────────

def load_artifacts():
    rf = joblib.load(MODEL_DIR / "rf_model.pkl")
    iso = joblib.load(MODEL_DIR / "iso_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    feat_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
    return rf, iso, scaler, feat_cols


def predict_single(row: dict) -> dict:
    rf, iso, scaler, feat_cols = load_artifacts()
    df_row = pd.DataFrame([row])
    df_row = engineer_features(df_row)
    X = df_row[feat_cols].fillna(0)
    prob = rf.predict_proba(X)[0][1]
    pred = int(prob >= 0.5)
    anomaly_score = float(-iso.decision_function(X)[0])
    return {
        "prediction": pred,
        "probability": round(float(prob), 4),
        "anomaly_score": round(anomaly_score, 4),
        "risk_level": _risk_level(prob),
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    rf, iso, scaler, feat_cols = load_artifacts()
    df = engineer_features(df)
    X = df[feat_cols].fillna(0)
    probs = rf.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    anomaly_scores = -iso.decision_function(X)
    df = df.copy()
    df["prediction"] = preds
    df["probability"] = probs.round(4)
    df["anomaly_score"] = anomaly_scores.round(4)
    df["risk_level"] = [_risk_level(p) for p in probs]
    return df


def _risk_level(prob: float) -> str:
    if prob >= 0.75:
        return "CRITICAL"
    elif prob >= 0.5:
        return "HIGH"
    elif prob >= 0.25:
        return "MEDIUM"
    return "LOW"


# ── SHAP ─────────────────────────────────────────────────────────────────────

def compute_shap(df: pd.DataFrame, max_rows: int = 500) -> dict:
    rf, _, _, feat_cols = load_artifacts()
    df = engineer_features(df)
    X = df[feat_cols].fillna(0).head(max_rows)
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X)
    # class 1 (breakdown)
    sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
    mean_abs = np.abs(sv).mean(axis=0)
    importance = dict(zip(feat_cols, mean_abs.round(4).tolist()))
    return {
        "feature_importance": importance,
        "shap_values": sv.tolist(),
        "feature_names": feat_cols,
        "sample_data": X.values.tolist(),
    }


def model_exists() -> bool:
    return (MODEL_DIR / "rf_model.pkl").exists()
