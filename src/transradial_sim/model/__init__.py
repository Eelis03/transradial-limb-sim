"""Physical model layer: pure functions and dataclasses, no input and no output.

Nothing in this package reads a file, prints, plots, or holds mutable global state. Every
function is a deterministic map from its arguments to its result, which is what makes the
invariant tests in ``tests/`` meaningful.
"""

from __future__ import annotations

from transradial_sim.model.contact import (
    CircularObject,
    ContactGeometry,
    ContactModel,
    ContactResult,
    FlatSurface,
    evaluate_contact,
)
from transradial_sim.model.finger import (
    INDEX_FINGER,
    FingerGeometry,
    Phalanx,
    mass_matrix,
    tendon_displacement,
)
from transradial_sim.model.gearbox import (
    MAXON_GP26B_84,
    GearboxParameters,
    current_limit_from_gearbox,
    lost_motion_m,
)
from transradial_sim.model.motor import (
    MAXON_RE25_118752,
    CatalogueOperatingPoint,
    MotorParameters,
    predicted_no_load_speed,
    predicted_stall_torque,
)
from transradial_sim.model.system import (
    PlantEvaluation,
    PowerSupply,
    StoredEnergy,
    SystemParameters,
    energy_residual,
    evaluate_plant,
    initial_state,
    state_derivative,
    stored_energy,
)
from transradial_sim.model.tendon import (
    PROSTHETIC_ROUTING,
    PROSTHETIC_TENDON,
    RoutingSegment,
    TendonParameters,
    TendonState,
    capstan_ratio,
    evaluate_tendon,
)

__all__ = [
    "INDEX_FINGER",
    "MAXON_GP26B_84",
    "MAXON_RE25_118752",
    "PROSTHETIC_ROUTING",
    "PROSTHETIC_TENDON",
    "CatalogueOperatingPoint",
    "CircularObject",
    "ContactGeometry",
    "ContactModel",
    "ContactResult",
    "FingerGeometry",
    "FlatSurface",
    "GearboxParameters",
    "MotorParameters",
    "Phalanx",
    "PlantEvaluation",
    "PowerSupply",
    "RoutingSegment",
    "StoredEnergy",
    "SystemParameters",
    "TendonParameters",
    "TendonState",
    "capstan_ratio",
    "current_limit_from_gearbox",
    "energy_residual",
    "evaluate_contact",
    "evaluate_plant",
    "evaluate_tendon",
    "initial_state",
    "lost_motion_m",
    "mass_matrix",
    "predicted_no_load_speed",
    "predicted_stall_torque",
    "state_derivative",
    "stored_energy",
    "tendon_displacement",
]
