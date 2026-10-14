"""Single public JSON contract for diagnosis and anomaly endpoints."""
from __future__ import annotations

import re
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict


class ExplainableWorkOrderResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: str
    reasoning: str
    recommendation: str
    next_steps: list[str]


def build_explainable_response(
    telemetry: Mapping[str, Any],
    *,
    copilot_result: Mapping[str, Any] | None = None,
    prediction: Mapping[str, Any] | None = None,
    shap_features: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a schema-constrained answer using only measured/model/source data."""
    copilot = copilot_result or {}
    predicted = prediction or {}
    work_order = copilot.get("work_order_structured") or {}
    risk = str(copilot.get("risk_level", predicted.get("risk_level", "NORMAL"))).upper()
    probability = float(copilot.get("breakdown_probability", predicted.get("probability", 0.0)) or 0.0)
    health_score = float(copilot.get("health_score", 100.0) or 0.0)

    if predicted.get("validation_status") == "MALFUNCTION":
        verdict = "SENSOR_MALFUNCTION_DETECTED"
    elif risk == "CRITICAL" or probability >= 0.8 or health_score < 40:
        verdict = "CRITICAL_ANOMALY_DETECTED"
    elif risk in {"HIGH", "MEDIUM"} or probability >= 0.5 or health_score < 70:
        verdict = "ANOMALY_DETECTED"
    else:
        verdict = "NO_ANOMALY_DETECTED"

    features = shap_features or copilot.get("shap_top_features") or []
    # The work-order node stores a compact, already formatted SHAP fallback.
    # Prefer structured SHAP records when present so impact values stay numeric.
    if not features:
        features = [
            {"display_name": item.rsplit(" (", 1)[0],
             "shap_impact": float(item.rsplit(" (", 1)[1].rstrip(")"))}
            for item in work_order.get("shap_drivers", [])
            if " (" in item and item.endswith(")")
        ]
    driver_text = "; ".join(
        f"{feature.get('display_name', feature.get('feature', 'sensor'))} "
        f"{float(feature.get('shap_impact', 0.0)):+.2f} SHAP"
        for feature in features[:3]
    ) or "SHAP feature attribution is unavailable"
    fft_peak = telemetry.get("fft_peak_hz", telemetry.get("spectral_peak_hz"))
    spectral = (
        f"FFT spectral resonance at {float(fft_peak) / 1000:.3g} kHz"
        if isinstance(fft_peak, (int, float)) and fft_peak > 0
        else "FFT spectral data is unavailable for this reading"
    )
    reasoning = f"Top SHAP sensor drivers ({driver_text}) combined with {spectral}."

    tier = copilot.get("escalation_tier", "LEVEL_1_STANDARD_SOP")
    tier_label = {
        "LEVEL_1_STANDARD_SOP": "L1",
        "LEVEL_2_SENIOR_TECH": "L2",
        "LEVEL_3_OEM_SPECIALIST": "L3",
    }.get(tier, "L3")
    sop_id = work_order.get("matched_sop_id")
    chunk_id = work_order.get("matched_sop_chunk_id")
    if verdict == "SENSOR_MALFUNCTION_DETECTED":
        recommendation = "Sensor validation failed; correct the telemetry fault before maintenance triage."
    elif verdict == "NO_ANOMALY_DETECTED":
        recommendation = "L1 routine monitoring; no maintenance escalation is indicated."
    elif sop_id:
        citation = f" (vector chunk {chunk_id})" if chunk_id else ""
        recommendation = f"{tier_label} escalation; follow OEM SOP {sop_id}{citation}."
    else:
        recommendation = f"{tier_label} escalation; no matching OEM SOP was retrieved, request specialist review."

    next_steps: list[str] = []
    if verdict == "SENSOR_MALFUNCTION_DETECTED":
        next_steps.append("Validate sensor calibration, wiring, units, and timestamp before rerunning diagnosis.")
    elif verdict == "NO_ANOMALY_DETECTED":
        next_steps.append("Continue routine monitoring; no manual-backed maintenance action is indicated.")
    else:
        safety = work_order.get("safety_protocol")
        if safety:
            next_steps.append(f"Apply the cited OEM safety protocol: {safety}")
        for action in work_order.get("recommended_actions", []):
            if action and action not in next_steps:
                # Manuals often number their own steps; the response numbers
                # each instruction uniformly below.
                clean_action = str(action).strip()
                clean_action = re.sub(r"^\d+[.)]\s*", "", clean_action)
                next_steps.append(clean_action)
        for part in work_order.get("required_parts", []):
            if part:
                next_steps.append(f"Verify and stage OEM-listed part: {part}")
        tolerances = work_order.get("clearance_tolerances")
        if tolerances:
            next_steps.append(f"Verify OEM tolerances: {tolerances}")
    if not next_steps:
        next_steps = ["Continue routine monitoring; no manual-backed maintenance action is indicated."]
    next_steps = [f"Step {index}: {step}" for index, step in enumerate(next_steps, start=1)]

    return ExplainableWorkOrderResponse(
        verdict=verdict,
        reasoning=reasoning,
        recommendation=recommendation,
        next_steps=next_steps,
    ).model_dump()
