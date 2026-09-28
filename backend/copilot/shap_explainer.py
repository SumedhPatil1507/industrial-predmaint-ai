"""
Local SHAP feature explanation utilities for single sensor telemetry events.
"""

import numpy as np
import pandas as pd
import logging
from backend import ml_engine

logger = logging.getLogger(__name__)

FEATURE_DISPLAY_NAMES = {
    "temp_bearing_degC": "Bearing Temperature (°C)",
    "temp_motor_degC": "Motor Temperature (°C)",
    "vibration_h_mms": "Horizontal Vibration (mm/s)",
    "vibration_v_mms": "Vertical Vibration (mm/s)",
    "oil_pressure_bar": "Oil Pressure (bar)",
    "load_pct": "Machine Load (%)",
    "shaft_rpm": "Shaft Speed (RPM)",
    "power_consumption_kw": "Power Draw (kW)",
    "temp_diff": "Motor-Bearing ΔT (°C)",
    "vibration_total": "Total Resultant Vibration (mm/s)",
    "power_per_load": "Power-to-Load Ratio",
}

# Nominal industrial baselines for comparison
NOMINAL_BASELINES = {
    "temp_bearing_degC": (60.0, 75.0),
    "temp_motor_degC": (65.0, 80.0),
    "vibration_h_mms": (1.5, 3.5),
    "vibration_v_mms": (1.2, 3.0),
    "oil_pressure_bar": (4.5, 6.0),
    "load_pct": (50.0, 85.0),
    "shaft_rpm": (1000.0, 1800.0),
    "power_consumption_kw": (15.0, 30.0),
}


def explain_reading_shap(telemetry: dict) -> dict:
    """
    Computes local SHAP explanation and feature contribution breakdown for a single reading.
    """
    df_row = pd.DataFrame([telemetry])
    engineered = ml_engine.engineer_features(df_row)
    
    try:
        rf, iso, scaler, feat_cols = ml_engine.load_artifacts()
    except Exception as e:
        logger.warning(f"Could not load ML artifacts for SHAP: {e}")
        feat_cols = ml_engine.FEATURE_COLS

    for c in feat_cols:
        if c not in engineered.columns:
            engineered[c] = 0.0

    X = engineered[feat_cols].fillna(0.0)

    # Try SHAP TreeExplainer if available
    contributions = {}
    try:
        import shap
        explainer = shap.TreeExplainer(rf)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            sv = shap_vals[0, :, 1]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
            sv = shap_vals[0]
        else:
            sv = np.array(shap_vals).flatten()
        
        for name, val in zip(feat_cols, sv):
            contributions[name] = float(val)
    except Exception as e:
        logger.warning(f"SHAP TreeExplainer fallback: {e}")
        # Rule-based heuristic SHAP proxy based on normalized deviations
        for col in feat_cols:
            val = float(telemetry.get(col, 0))
            if col in NOMINAL_BASELINES:
                low, high = NOMINAL_BASELINES[col]
                if val > high:
                    contributions[col] = round((val - high) / (high * 0.5), 3)
                elif val < low and col == "oil_pressure_bar":
                    contributions[col] = round((low - val) / (low * 0.5), 3)
                else:
                    contributions[col] = 0.02
            else:
                contributions[col] = 0.05

    # Rank features by absolute impact
    sorted_features = sorted(
        contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True
    )

    top_features = []
    symptom_descriptions = []

    for name, impact in sorted_features[:4]:
        raw_val = telemetry.get(name, engineered.get(name, [0])[0] if name in engineered else 0)
        display_name = FEATURE_DISPLAY_NAMES.get(name, name)
        
        # Check deviation direction
        direction = "elevated" if impact > 0 else "reduced"
        top_features.append({
            "feature": name,
            "display_name": display_name,
            "raw_value": round(float(raw_val), 2) if isinstance(raw_val, (int, float)) else raw_val,
            "shap_impact": round(impact, 4),
            "direction": direction,
        })
        symptom_descriptions.append(f"{display_name} = {raw_val} (impact: {impact:+.2f})")

    summary_text = (
        f"Primary failure drivers: " + ", ".join([f"{f['display_name']} ({f['shap_impact']:+.2f})" for f in top_features[:3]])
    )

    return {
        "all_contributions": contributions,
        "top_features": top_features,
        "summary": summary_text,
        "symptoms_text": " | ".join(symptom_descriptions),
    }
