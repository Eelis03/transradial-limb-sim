"""Analysis layer: efficiency chain, metrics, rating audit, sensitivity study, figures."""

from __future__ import annotations

from transradial_sim.analysis.energetics import (
    ChainStage,
    EnergyBudget,
    ForceChain,
    LossRow,
    energy_budget,
    force_chain,
    frictionless_tension_n,
)
from transradial_sim.analysis.metrics import (
    GraspSummary,
    PerformanceSummary,
    closing_time_s,
    free_running_speeds,
    grasp_summary,
    performance_summary,
    settling_index,
)
from transradial_sim.analysis.ratings import RatingAudit, RatingCheck, rating_audit
from transradial_sim.analysis.sensitivity import (
    FRICTION_VALUES,
    STIFFNESS_VALUES,
    SensitivityPoint,
    relative_spread,
    sweep_capstan_friction,
    sweep_tendon_stiffness,
)

__all__ = [
    "FRICTION_VALUES",
    "STIFFNESS_VALUES",
    "ChainStage",
    "EnergyBudget",
    "ForceChain",
    "GraspSummary",
    "LossRow",
    "PerformanceSummary",
    "RatingAudit",
    "RatingCheck",
    "SensitivityPoint",
    "closing_time_s",
    "energy_budget",
    "force_chain",
    "free_running_speeds",
    "frictionless_tension_n",
    "grasp_summary",
    "performance_summary",
    "rating_audit",
    "relative_spread",
    "settling_index",
    "sweep_capstan_friction",
    "sweep_tendon_stiffness",
]
