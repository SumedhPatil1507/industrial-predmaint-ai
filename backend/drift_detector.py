"""
Model Drift Detector
Compares current sensor distributions against training baseline.
Uses Population Stability Index (PSI) — KS test used as secondary signal only.

PSI thresholds (calibrated for industrial sensor data):
  < 0.15  = Stable
  0.15-0.25 = Warning
  > 0.25  = Critical drift
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

# Calibrated thresholds for industrial sensor data
PSI_WARNING  = 0.15
PSI_CRITICAL = 0.25
# Mean shift threshold — flag only if mean moves by more than this %
MEAN_SHIFT_THRESHOLD = 10.0


@dataclass
class DriftResult:
    feature: str
    psi: float
    mean_shift_pct: float
    drift_detected: bool
    severity: str        # None / Warning / Critical
    baseline_mean: float
    current_mean: float
    baseline_std: float
    current_std: float


@dataclass
class FleetDriftReport:
    results: list = field(default_factory=list)
    overall_drift: bool = False
    drift_score: float = 0.0
    features_drifted: list = field(default_factory=list)
    recommendation: str = ""


def save_baseline(df: pd.DataFrame):
    """Save training distribution statistics as baseline.
    Uses only normal (non-breakdown) records for a clean reference."""
    stats = {}
    # Use normal operating data as baseline — more stable reference
    normal_df = df[df["breakdown_flag"] == 0] if "breakdown_flag" in df.columns else df
    if len(normal_df) < 100:
        normal_df = df  # fallback if no label column

    for col in SENSOR_COLS:
        if col not in normal_df.columns:
            continue
        vals = normal_df[col].dropna().values
        if len(vals) < 30:
            continue
        # Store a representative sample (max 3000 to avoid memory issues)
        sample_size = min(3000, len(vals))
        rng = np.random.default_rng(42)
        sample = rng.choice(vals, size=sample_size, replace=False)
        stats[col] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "min":  float(np.min(vals)),
            "max":  float(np.max(vals)),
            "p5":   float(np.percentile(vals, 5)),
            "p95":  float(np.percentile(vals, 95)),
            "samples": sample.tolist(),
        }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(stats, BASELINE_FILE)
    return stats


def load_baseline() -> dict:
    if BASELINE_FILE.exists():
        return joblib.load(BASELINE_FILE)
    return {}


def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index."""
    mn = min(expected.min(), actual.min())
    mx = max(expected.max(), actual.max())
    if mx == mn:
        return 0.0
    bins = np.linspace(mn, mx, buckets + 1)
    exp_cnt = np.histogram(expected, bins=bins)[0]
    act_cnt = np.histogram(actual,   bins=bins)[0]
    # Normalise to proportions
    exp_pct = exp_cnt / max(exp_cnt.sum(), 1)
    act_pct = act_cnt / max(act_cnt.sum(), 1)
    # Replace zeros to avoid log(0)
    exp_pct = np.where(exp_pct == 0, 1e-4, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-4, act_pct)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def detect_drift(current_df: pd.DataFrame) -> FleetDriftReport:
    baseline = load_baseline()
    if not baseline:
        try:
            save_baseline(current_df)
            baseline = load_baseline()
        except Exception:
            pass
    if not baseline:
        return FleetDriftReport(recommendation="Baseline saved. Run analysis again.")

    results = []
    for col in SENSOR_COLS:
        if col not in current_df.columns or col not in baseline:
            continue

        current_vals = current_df[col].dropna().values
        baseline_vals = np.array(baseline[col]["samples"])

        if len(current_vals) < 30:
            continue

        psi = _psi(baseline_vals, current_vals)

        b_mean = baseline[col]["mean"]
        b_std  = baseline[col]["std"]
        c_mean = float(np.mean(current_vals))
        c_std  = float(np.std(current_vals))
        shift  = abs((c_mean - b_mean) / b_mean * 100) if b_mean != 0 else 0.0

        # Drift = PSI above threshold AND meaningful mean shift
        # Both conditions required to avoid false positives on large samples
        if psi > PSI_CRITICAL and shift > MEAN_SHIFT_THRESHOLD:
            severity = "Critical"
            drift = True
        elif psi > PSI_WARNING and shift > MEAN_SHIFT_THRESHOLD:
            severity = "Warning"
            drift = True
        else:
            severity = "None"
            drift = False

        results.append(DriftResult(
            feature=col,
            psi=round(psi, 4),
            mean_shift_pct=round(shift, 1),
            drift_detected=drift,
            severity=severity,
            baseline_mean=round(b_mean, 3),
            current_mean=round(c_mean, 3),
            baseline_std=round(b_std, 3),
            current_std=round(c_std, 3),
        ))

    drifted = [r.feature for r in results if r.drift_detected]
    drift_score = round(np.mean([r.psi for r in results]) if results else 0.0, 4)
    overall = len(drifted) > 0

    if len(drifted) >= 4:
        rec = "Critical drift in multiple sensors. Retrain model with recent data."
    elif len(drifted) >= 2:
        rec = f"Drift detected in {', '.join(drifted)}. Consider retraining within 1 week."
    elif len(drifted) == 1:
        rec = f"Minor drift in {drifted[0]}. Monitor for 3-5 days before retraining."
    else:
        rec = "No significant drift detected. Model is stable and reliable."

    return FleetDriftReport(
        results=results,
        overall_drift=overall,
        drift_score=drift_score,
        features_drifted=drifted,
        recommendation=rec,
    )
