"""
Time-to-Failure (TTF) Predictor
Uses rolling degradation trend to estimate days until breakdown.
Implements a simple linear regression on the degradation slope
plus a survival-analysis-inspired hazard rate.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class TTFResult:
    asset_tag: str
    estimated_days: float        # days until predicted failure
    confidence: str              # High / Medium / Low
    degradation_rate: float      # % per day
    recommended_action_by: str   # date string
    urgency: str                 # Immediate / This Week / This Month / Routine


def estimate_ttf(df: pd.DataFrame, asset_tag: str) -> TTFResult:
    """
    Estimate time-to-failure for a specific asset using its historical data.
    df must have: transaction_date, asset_tag, breakdown_flag + sensor cols.
    """
    asset_df = df[df["asset_tag"] == asset_tag].copy()
    asset_df["transaction_date"] = pd.to_datetime(asset_df["transaction_date"])
    asset_df = asset_df.sort_values("transaction_date").reset_index(drop=True)

    if len(asset_df) < 14:
        return TTFResult(asset_tag, 30.0, "Low", 0.0, "N/A", "Routine")

    # Build a composite stress index from sensor readings
    asset_df["stress"] = (
        _normalize(asset_df, "temp_bearing_degC") * 0.25 +
        _normalize(asset_df, "vibration_h_mms") * 0.30 +
        _normalize(asset_df, "vibration_v_mms") * 0.20 +
        (1 - _normalize(asset_df, "oil_pressure_bar")) * 0.15 +
        _normalize(asset_df, "power_consumption_kw") * 0.10
    )

    # Rolling 7-day mean stress
    asset_df["stress_roll"] = asset_df["stress"].rolling(7, min_periods=1).mean()

    # Fit linear trend on last 30 days
    recent = asset_df.tail(30)
    x = np.arange(len(recent))
    y = recent["stress_roll"].values
    slope, intercept = np.polyfit(x, y, 1)

    # Days until stress hits 1.0 (failure threshold)
    current_stress = y[-1]
    if slope <= 0:
        # Improving or stable
        days = 90.0
        confidence = "High"
    else:
        remaining = (1.0 - current_stress) / slope
        days = max(1.0, min(365.0, float(remaining)))
        confidence = "High" if len(recent) >= 25 else "Medium" if len(recent) >= 14 else "Low"

    # Adjust by historical breakdown frequency
    bd_rate = asset_df["breakdown_flag"].mean() if "breakdown_flag" in asset_df.columns else 0.05
    hazard_adjustment = 1.0 + bd_rate * 5
    days = days / hazard_adjustment

    days = round(days, 1)
    deg_rate = round(slope * 100, 3)

    last_date = asset_df["transaction_date"].max()
    action_date = (last_date + pd.Timedelta(days=max(1, days * 0.7))).strftime("%Y-%m-%d")

    if days <= 3:
        urgency = "Immediate"
    elif days <= 7:
        urgency = "This Week"
    elif days <= 30:
        urgency = "This Month"
    else:
        urgency = "Routine"

    return TTFResult(
        asset_tag=asset_tag,
        estimated_days=days,
        confidence=confidence,
        degradation_rate=deg_rate,
        recommended_action_by=action_date,
        urgency=urgency,
    )


def fleet_ttf(df: pd.DataFrame) -> list[dict]:
    results = []
    for tag in df["asset_tag"].unique():
        r = estimate_ttf(df, tag)
        results.append(r.__dict__)
    return sorted(results, key=lambda x: x["estimated_days"])


def _normalize(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index)
    mn, mx = df[col].min(), df[col].max()
    if mx == mn:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return (df[col] - mn) / (mx - mn)
