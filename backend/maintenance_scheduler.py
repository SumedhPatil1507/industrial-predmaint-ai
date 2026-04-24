"""
Maintenance Schedule Generator
Given TTF predictions, generates an optimal weekly maintenance schedule.
Outputs prioritized work orders with estimated durations and spare parts.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field


MAINTENANCE_TASKS = {
    "CNC Lathe": {
        "daily":   ["Visual inspection", "Lubrication check", "Temperature log"],
        "weekly":  ["Bearing inspection", "Alignment check", "Coolant level"],
        "monthly": ["Full bearing replacement check", "Spindle calibration", "Belt tension"],
        "parts":   ["Bearing set (6205-2RS)", "Coolant filter", "Drive belt"],
        "mttr_hours": 8,
    },
    "Hydraulic Press": {
        "daily":   ["Oil pressure check", "Seal visual inspection", "Cylinder stroke test"],
        "weekly":  ["Hydraulic fluid sample", "Pressure relief valve test", "Filter check"],
        "monthly": ["Full seal replacement check", "Pump inspection", "Valve calibration"],
        "parts":   ["Hydraulic seals kit", "Oil filter", "Pressure gauge"],
        "mttr_hours": 12,
    },
    "Belt Conveyor": {
        "daily":   ["Belt tension check", "Roller visual", "Drive motor temp"],
        "weekly":  ["Belt alignment", "Idler roller lubrication", "Drive chain tension"],
        "monthly": ["Belt wear measurement", "Full roller inspection", "Motor coupling check"],
        "parts":   ["Conveyor belt section", "Idler roller", "Drive chain link"],
        "mttr_hours": 6,
    },
    "Screw Compressor": {
        "daily":   ["Oil level check", "Discharge temp log", "Pressure ratio check"],
        "weekly":  ["Air filter inspection", "Oil sample", "Valve leak test"],
        "monthly": ["Oil change", "Screw element inspection", "Safety valve test"],
        "parts":   ["Oil filter", "Air filter element", "Shaft seal"],
        "mttr_hours": 10,
    },
    "EOT Crane": {
        "daily":   ["Brake test", "Hook visual inspection", "Wire rope check"],
        "weekly":  ["Load test (10%)", "Limit switch test", "Lubrication"],
        "monthly": ["Full wire rope inspection", "Brake lining measurement", "Motor insulation test"],
        "parts":   ["Wire rope section", "Brake lining", "Hook block assembly"],
        "mttr_hours": 8,
    },
}

URGENCY_PRIORITY = {"Immediate": 1, "This Week": 2, "This Month": 3, "Routine": 4}


@dataclass
class WorkOrder:
    wo_id: str
    asset_tag: str
    machine_type: str
    priority: int
    urgency: str
    scheduled_date: str
    task_type: str
    tasks: list
    spare_parts: list
    estimated_hours: float
    estimated_cost_inr: float
    ttf_days: float
    health_score: float = 0.0


def generate_schedule(
    ttf_results: list[dict],
    health_scores: dict = None,
    start_date: datetime = None,
    horizon_days: int = 30,
) -> list[WorkOrder]:
    if start_date is None:
        start_date = datetime.now()

    health_scores = health_scores or {}
    work_orders = []

    for i, asset in enumerate(ttf_results):
        tag = asset["asset_tag"]
        mtype = _infer_machine_type(tag)
        urgency = asset.get("urgency", "Routine")
        ttf = asset.get("estimated_days", 30)
        tasks_info = MAINTENANCE_TASKS.get(mtype, MAINTENANCE_TASKS["CNC Lathe"])
        hs = health_scores.get(tag, 75.0)

        # Determine task type and schedule date
        if urgency == "Immediate" or ttf <= 3:
            task_type = "Emergency Inspection"
            tasks = tasks_info["daily"] + tasks_info["weekly"]
            sched = start_date
            est_hours = tasks_info["mttr_hours"]
            cost = est_hours * 2500 + 15000
        elif urgency == "This Week" or ttf <= 7:
            task_type = "Preventive Maintenance"
            tasks = tasks_info["weekly"] + tasks_info["monthly"]
            sched = start_date + timedelta(days=min(3, int(ttf * 0.5)))
            est_hours = tasks_info["mttr_hours"] * 0.6
            cost = est_hours * 2000 + 8000
        elif urgency == "This Month" or ttf <= 30:
            task_type = "Scheduled Inspection"
            tasks = tasks_info["monthly"]
            sched = start_date + timedelta(days=min(14, int(ttf * 0.6)))
            est_hours = tasks_info["mttr_hours"] * 0.4
            cost = est_hours * 1500 + 5000
        else:
            task_type = "Routine PM"
            tasks = tasks_info["daily"] + tasks_info["weekly"]
            sched = start_date + timedelta(days=min(horizon_days - 1, int(ttf * 0.7)))
            est_hours = tasks_info["mttr_hours"] * 0.3
            cost = est_hours * 1200 + 3000

        # Skip if scheduled beyond horizon
        if (sched - start_date).days > horizon_days:
            continue

        work_orders.append(WorkOrder(
            wo_id=f"WO-{start_date.strftime('%Y%m%d')}-{i+1:03d}",
            asset_tag=tag,
            machine_type=mtype,
            priority=URGENCY_PRIORITY.get(urgency, 4),
            urgency=urgency,
            scheduled_date=sched.strftime("%Y-%m-%d"),
            task_type=task_type,
            tasks=tasks,
            spare_parts=tasks_info["parts"],
            estimated_hours=round(est_hours, 1),
            estimated_cost_inr=round(cost, 0),
            ttf_days=ttf,
            health_score=hs,
        ))

    return sorted(work_orders, key=lambda x: (x.priority, x.scheduled_date))


def schedule_to_dataframe(work_orders: list[WorkOrder]) -> pd.DataFrame:
    rows = []
    for wo in work_orders:
        rows.append({
            "WO ID": wo.wo_id,
            "Asset": wo.asset_tag,
            "Machine": wo.machine_type,
            "Priority": wo.priority,
            "Urgency": wo.urgency,
            "Scheduled Date": wo.scheduled_date,
            "Task Type": wo.task_type,
            "Est. Hours": wo.estimated_hours,
            "Est. Cost (Rs.)": f"{wo.estimated_cost_inr:,.0f}",
            "TTF (days)": wo.ttf_days,
            "Health Score": wo.health_score,
            "Spare Parts": ", ".join(wo.spare_parts),
        })
    return pd.DataFrame(rows)


def _infer_machine_type(asset_tag: str) -> str:
    mapping = {
        "CNC": "CNC Lathe", "HYD": "Hydraulic Press",
        "BLT": "Belt Conveyor", "CMP": "Screw Compressor", "EOT": "EOT Crane",
    }
    for prefix, mtype in mapping.items():
        if asset_tag.startswith(prefix):
            return mtype
    return "CNC Lathe"
