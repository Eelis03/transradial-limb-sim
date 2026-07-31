"""Algorithm layer: controllers and the integrator formulation. No plotting, no input.

Both the controllers and the integrators satisfy the structural interfaces in
:mod:`transradial_sim.algorithm.protocols`, so the pipeline can be handed any combination
of the two without change.
"""

from __future__ import annotations

from transradial_sim.algorithm.controllers import (
    ConstantDuty,
    CurrentController,
    ForceController,
    PlantMeasurement,
    PositionController,
    ScheduledCurrentController,
    current_loop_gains,
)
from transradial_sim.algorithm.integrators import (
    ForwardEuler,
    RungeKutta4,
    integrate_fixed_step,
    richardson_order,
)
from transradial_sim.algorithm.protocols import Controller, Derivative, Integrator, Measurement

__all__ = [
    "ConstantDuty",
    "Controller",
    "CurrentController",
    "Derivative",
    "ForceController",
    "ForwardEuler",
    "Integrator",
    "Measurement",
    "PlantMeasurement",
    "PositionController",
    "RungeKutta4",
    "ScheduledCurrentController",
    "current_loop_gains",
    "integrate_fixed_step",
    "richardson_order",
]
