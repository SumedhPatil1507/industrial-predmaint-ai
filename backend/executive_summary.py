"""
Executive Summary Generator
Produces a one-page decision-ready summary for plant managers.
No ML jargon — just: what needs attention, what it costs, what to do.
"""
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd


@dataclass
class ExecutiveSummary:
    generated_at: str
    total_assets: int
    critical_assets: list
    warning_assets: list
    healthy_assets: list
    total_risk_exposure_inr: float
    potential_savings_inr: float
    top_action: str
    actions: list
    oee_summary: list
    shift_recommendation: str


def generate_summary(
    fleet_health: list[dict],
    ttf_results: list[dict],
    oee_results: list = None,
    hourly_rate_inr: float = 25000,
) -> ExecutiveSummary:
    critical, warning, healthy = [], [], []
    risk_exposure = 0.0

    for asset in fleet_health:
        hs = asset.get("health_score", 100)
        tag = asset.get("asset_tag", "")
        mtype = asset.get("machine_type", "")

        ttf = next((t for t in ttf_results if t.get("asset_tag") == tag), {})
        days = ttf.get("estimated_days", 90)
        urgency = ttf.get("urgency", "Routine")

        entry = {
            "asset_tag": tag,
            "machine_type": mtype,
            "health_score": hs,
            "ttf_days": days,
            "urgency": urgency,
        }

        if hs < 50 or urgency == "Immediate":
            critical.append(entry)
            risk_exposure += hourly_rate_inr * 8 * 2  # 2 days downtime risk
        elif hs < 75 or urgency in ("This Week",):
            warning.append(entry)
            risk_exposure += hourly_rate_inr * 4
        else:
            healthy.append(entry)

    potential_savings = risk_exposure * 0.6  # 60% avoidable with PdM

    # Build prioritised action list
    actions = []
    for a in sorted(critical, key=lambda x: x["ttf_days"]):
        actions.append({
            "priority": "CRITICAL",
            "asset": a["asset_tag"],
            "action": f"Immediate inspection — {a['machine_type']} at {a['health_score']:.0f}/100 health",
            "deadline": "Today",
            "est_cost_inr": 15000,
        })
    for a in sorted(warning, key=lambda x: x["ttf_days"]):
        actions.append({
            "priority": "WARNING",
            "asset": a["asset_tag"],
            "action": f"Schedule PM within {a['ttf_days']:.0f} days — {a['machine_type']}",
            "deadline": f"Within {a['ttf_days']:.0f} days",
            "est_cost_inr": 8000,
        })

    top_action = actions[0]["action"] if actions else "All machines operating normally"

    # Shift recommendation
    if critical:
        tags = ", ".join(a["asset_tag"] for a in critical)
        shift_rec = f"ALERT: {len(critical)} machine(s) require immediate attention — {tags}. Do not run at full load."
    elif warning:
        tags = ", ".join(a["asset_tag"] for a in warning)
        shift_rec = f"CAUTION: {len(warning)} machine(s) showing early degradation — {tags}. Monitor closely."
    else:
        shift_rec = "All machines healthy. Proceed with normal production schedule."

    oee_summary = []
    if oee_results:
        for r in oee_results:
            oee_summary.append({
                "asset": r.asset_tag,
                "oee": r.oee,
                "rating": r.rating,
                "annual_loss": r.annual_loss_inr,
            })

    return ExecutiveSummary(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        total_assets=len(fleet_health),
        critical_assets=critical,
        warning_assets=warning,
        healthy_assets=healthy,
        total_risk_exposure_inr=round(risk_exposure, 0),
        potential_savings_inr=round(potential_savings, 0),
        top_action=top_action,
        actions=actions,
        oee_summary=oee_summary,
        shift_recommendation=shift_rec,
    )
