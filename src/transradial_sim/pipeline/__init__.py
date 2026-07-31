"""Pipeline layer: simulation scenarios producing a structured trace."""

from __future__ import annotations

from transradial_sim.pipeline.scenario import (
    APPROACH_CURRENT_A,
    APPROACH_SPEED_LIMIT_RAD_S,
    COMPLIANT_CONTACT,
    FLAT_PLATE,
    LARGE_CYLINDER,
    PROSTHESIS_SUPPLY,
    SMALL_CYLINDER,
    SOFT_CONTACT,
    SQUEEZE_START_S,
    STIFF_CONTACT,
    TEST_OBJECTS,
    build_plant,
    closing_scenario,
    current_controller,
    grasp_controller,
    grasp_scenario,
    reference_current_limit_a,
    stall_scenario,
)
from transradial_sim.pipeline.simulate import (
    ScenarioConfig,
    SimulationController,
    fingertip_force,
    run_scenario,
)
from transradial_sim.pipeline.trace import SimulationTrace

__all__ = [
    "APPROACH_CURRENT_A",
    "APPROACH_SPEED_LIMIT_RAD_S",
    "COMPLIANT_CONTACT",
    "FLAT_PLATE",
    "LARGE_CYLINDER",
    "PROSTHESIS_SUPPLY",
    "SMALL_CYLINDER",
    "SOFT_CONTACT",
    "SQUEEZE_START_S",
    "STIFF_CONTACT",
    "TEST_OBJECTS",
    "ScenarioConfig",
    "SimulationController",
    "SimulationTrace",
    "build_plant",
    "closing_scenario",
    "current_controller",
    "fingertip_force",
    "grasp_controller",
    "grasp_scenario",
    "reference_current_limit_a",
    "run_scenario",
    "stall_scenario",
]
