"""Downtime cost calculator – INR and USD."""
from dataclasses import dataclass


MACHINE_DEFAULTS = {
    "CNC Lathe":        dict(hourly_production_inr=25000, repair_cost_inr=80000,  mttr_hours=8),
    "Hydraulic Press":  dict(hourly_production_inr=35000, repair_cost_inr=120000, mttr_hours=12),
    "Belt Conveyor":    dict(hourly_production_inr=15000, repair_cost_inr=50000,  mttr_hours=6),
    "Screw Compressor": dict(hourly_production_inr=20000, repair_cost_inr=90000,  mttr_hours=10),
    "EOT Crane":        dict(hourly_production_inr=18000, repair_cost_inr=70000,  mttr_hours=8),
}

USD_RATE = 83.5  # INR per USD (update as needed)


@dataclass
class DowntimeResult:
    machine_type: str
    mttr_hours: float
    production_loss_inr: float
    repair_cost_inr: float
    total_cost_inr: float
    total_cost_usd: float
    annual_bd_cost_inr: float
    savings_with_pdm_inr: float  # 50% reduction assumption
    roi_percent: float


def calculate_downtime(
    machine_type: str,
    mttr_hours: float | None = None,
    hourly_production_inr: float | None = None,
    repair_cost_inr: float | None = None,
    annual_breakdowns: int = 12,
    pdm_system_cost_inr: float = 500000,
) -> DowntimeResult:
    defaults = MACHINE_DEFAULTS.get(machine_type, MACHINE_DEFAULTS["CNC Lathe"])

    mttr = mttr_hours or defaults["mttr_hours"]
    hourly = hourly_production_inr or defaults["hourly_production_inr"]
    repair = repair_cost_inr or defaults["repair_cost_inr"]

    prod_loss = mttr * hourly
    total_per_event = prod_loss + repair
    annual_cost = total_per_event * annual_breakdowns
    savings = annual_cost * 0.50  # conservative 50% reduction
    roi = ((savings - pdm_system_cost_inr) / pdm_system_cost_inr) * 100

    return DowntimeResult(
        machine_type=machine_type,
        mttr_hours=mttr,
        production_loss_inr=round(prod_loss, 2),
        repair_cost_inr=round(repair, 2),
        total_cost_inr=round(total_per_event, 2),
        total_cost_usd=round(total_per_event / USD_RATE, 2),
        annual_bd_cost_inr=round(annual_cost, 2),
        savings_with_pdm_inr=round(savings, 2),
        roi_percent=round(roi, 1),
    )
