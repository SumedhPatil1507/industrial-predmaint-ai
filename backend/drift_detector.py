"""
Model Drift Detector
Compares current sensor distributions against training baseline.
Uses Population Stability Index (PSI) and KS test.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass, field

MODEL_DIR = Path("models")
BASELINE_FILE = MODEL_DIR / "baseline_stats.pkl"

SENSOR_COLS = [
    "temp_bearing_degC", "temp_motor_degC",
    "vibration_h_mms", "vibration_v_mms",
    "oil_pressure_bar", "load_pct",
    "shaft_rpm", "power_consumption_kw",
]


@dataclass
class DriftResult:
    feature: str
    psi: float
    ks_stat: float
    ks_pvalue: float
    drift_detected: bool
    severity: str  # None / Warning / Critical
    baseline_mean: float
    current_mean: float
    mean_shift_pct: float


@dataclass
class FleetDriftReport:
    results: list = field(default_factory=list)
    overall_drift: bool = False
    drift_score: float = 0.0
    features_drifted: list = field(default_factory=list)
    recommendation: str = ""


def save_baseline(df: pd.DataFrame):
    """Save training data statistics as baseline."""
    stats = {}
    for col in SENSOR_COLS:
        if col in df.columns:
            vals = df[col].dropna().values
            stats[col] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "percentiles": np.percentile(vals, np.arange(0, 101, 10)).tolist(),
                "samples": vals[:5000].tolist(),  # store sample for KS test
            }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(stats, BASELINE_FILE)
    return stats


def load_baseline() -> dict:
    if BASELINE_FILE.exists():
        return joblib.load(BASELINE_FILE)
    return {}


def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index. PSI < 0.1 = stable, 0.1-0.2 = warning, >0.2 = critical."""
    mn = min(expected.min(), actual.min())
    mx = max(expected.max(), actual.max())
    if mx == mn:
        return 0.0
    bins = np.linspace(mn, mx, buckets + 1)
    exp_pct = np.histogram(expected, bins=bins)[0] / len(expected)
    act_pct = np.histogram(actual, bins=bins)[0] / len(actual)
    exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-6, act_pct)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def detect_drift(current_df: pd.DataFrame) -> FleetDriftReport:
    from scipy import stats as scipy_stats

    baseline = load_baseline()
    if not baseline:
        return FleetDriftReport(recommendation="No baseline found. Train model first.")

    results = []
    for col in SENSOR_COLS:
        if col not in current_df.columns or col not in baseline:
            continue

        current_vals = current_df[col].dropna().values
        baseline_vals = np.array(baseline[col]["samples"])

        if len(current_vals) < 30:
            continue

        psi = _psi(baseline_vals, current_vals)
        ks_stat, ks_pval = scipy_stats.ks_2samp(baseline_vals, current_vals)

        drift = psi > 0.1 or ks_pval < 0.05
        if psi > 0.2:
            severity = "Critical"
        elif psi > 0.1:
            severity = "Warning"
        else:
            severity = "None"

        b_mean = baseline[col]["mean"]
        c_mean = float(np.mean(current_vals))
        shift = ((c_mean - b_mean) / b_mean * 100) if b_mean != 0 else 0.0

        results.append(DriftResult(
            feature=col, psi=round(psi, 4),
            ks_stat=round(ks_stat, 4), ks_pvalue=round(ks_pval, 4),
            drift_detected=drift, severity=severity,
            baseline_mean=round(b_mean, 3), current_mean=round(c_mean, 3),
            mean_shift_pct=round(shift, 1),
        ))

    drifted = [r.feature for r in results if r.drift_detected]
    drift_score = round(np.mean([r.psi for r in results]) if results else 0.0, 4)
    overall = len(drifted) > 0

    if len(drifted) >= 4:
        rec = "Significant data drift detected. Model retraining strongly recommended."
    elif len(drifted) >= 2:
        rec = "Moderate drift in multiple features. Consider retraining within 1 week."
    elif len(drifted) == 1:
        rec = f"Minor drift in {drifted[0]}. Monitor closely."
    else:
        rec = "No significant drift detected. Model is stable."

    return FleetDriftReport(
        results=results, overall_drift=overall,
        drift_score=drift_score, features_drifted=drifted,
        recommendation=rec,
    )
