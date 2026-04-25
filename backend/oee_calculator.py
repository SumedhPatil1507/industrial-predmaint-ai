"""
OEE (Overall Equipment Effectiveness) Calculator
OEE = Availability x Performance x Quality
Industry standard KPI used by every manufacturing plant.
"""
from dataclasses import dataclass
import pandas as pd
import numpy as np


# World-class OEE benchmark = 85%
WORLD_CLASS_OEE = 85.0

OEE_BENCHMARKS = {
    "CNC Lathe":        {"availability": 90, "performance": 88, "quality": 98},
    "Hydraulic Press":  {"availability": 88, "performance": 85, "quality": 97},
    "Belt Conveyor":    {"availability": 92, "performance": 90, "quality": 99},
    "Screw Compressor": {"availability": 87, "performance": 86, "quality": 98},
    "EOT Crane":        {"availability": 91, "performance": 89, "quality": 99},
}


@dataclass
class OEEResult:
    machine_type: str
    asset_tag: str
    availability: float       # %
    performance: float        # %
    quality: float            # %
    oee: float                # %
    world_class_gap: float    # % gap from 85%
    annual_loss_inr: float    # cost of OEE gap
    improvement_potential: str
    rating: str               # World Class / Good / Average / Poor


def calculate_oee(
    machine_type: str,
    asset_tag: str,
    planned_hours: float = 8.0,
    downtime_hours: float = 0.5,
    ideal_cycle_time: float = 1.0,
    actual_cycle_time: float = 1.1,
    total_parts: int = 400,
    good_parts: int = 392,
    hourly_production_inr: float = 25000,
) -> OEEResult:
    availability = ((planned_hours - downtime_hours) / planned_hours) * 100
    performance  = (ideal_cycle_time / actual_cycle_time) * 100
    quality      = (good_parts / total_parts) * 100 if total_parts > 0 else 100.0

    oee = (availability * performance * quality) / 10000
    gap = max(0.0, WORLD_CLASS_OEE - oee)

    # Annual loss from OEE gap
    annual_loss = (gap / 100) * hourly_production_inr * planned_hours * 250  # 250 working days

    if oee >= 85:
        rating = "World Class"
        potential = "Maintain current performance"
    elif oee >= 75:
        rating = "Good"
        potential = f"Improve availability by {gap:.1f}% to reach world class"
    elif oee >= 65:
        rating = "Average"
        potential = "Focus on reducing unplanned downtime and speed losses"
    else:
        rating = "Poor"
        potential = "Immediate intervention required — high breakdown frequency"

    return OEEResult(
        machine_type=machine_type, asset_tag=asset_tag,
        availability=round(availability, 1), performance=round(performance, 1),
        quality=round(quality, 1), oee=round(oee, 1),
        world_class_gap=round(gap, 1),
        annual_loss_inr=round(annual_loss, 0),
        improvement_potential=potential, rating=rating,
    )


def oee_from_health_score(health_score: float, machine_type: str, asset_tag: str,
                           hourly_production_inr: float = 25000) -> OEEResult:
    """Estimate OEE from health score when actual production data isn't available."""
    bench = OEE_BENCHMARKS.get(machine_type, OEE_BENCHMARKS["CNC Lathe"])
    degradation = (100 - health_score) / 100

    availability = bench["availability"] - degradation * 15
    performance  = bench["performance"]  - degradation * 10
    quality      = bench["quality"]      - degradation * 3

    availability = max(50.0, min(100.0, availability))
    performance  = max(50.0, min(100.0, performance))
    quality      = max(90.0, min(100.0, quality))

    oee = (availability * performance * quality) / 10000
    gap = max(0.0, WORLD_CLASS_OEE - oee)
    annual_loss = (gap / 100) * hourly_production_inr * 8 * 250

    if oee >= 85:
        rating, potential = "World Class", "Maintain current performance"
    elif oee >= 75:
        rating, potential = "Good", f"Close {gap:.1f}% gap to world class"
    elif oee >= 65:
        rating, potential = "Average", "Reduce downtime and speed losses"
    else:
        rating, potential = "Poor", "Immediate intervention required"

    return OEEResult(
        machine_type=machine_type, asset_tag=asset_tag,
        availability=round(availability, 1), performance=round(performance, 1),
        quality=round(quality, 1), oee=round(oee, 1),
        world_class_gap=round(gap, 1), annual_loss_inr=round(annual_loss, 0),
        improvement_potential=potential, rating=rating,
    )
