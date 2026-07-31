"""Physics simulation of transradial prosthesis actuation with tendon and motor models.

The package is arranged in five layers, each importing only from the ones before it.

``transradial_sim.model``
    Motor, gearbox, tendon, finger and contact as pure functions and dataclasses.
``transradial_sim.algorithm``
    Controllers and the integrator formulation, behind structural interfaces.
``transradial_sim.pipeline``
    Simulation scenarios that produce a structured trace.
``transradial_sim.analysis``
    Efficiency chain, performance metrics, sensitivity study and figures.
``examples/``
    Thin wiring scripts that contain no logic of their own.
"""

from __future__ import annotations

from transradial_sim.model.gearbox import MAXON_GP26B_84
from transradial_sim.model.motor import MAXON_RE25_118752
from transradial_sim.model.system import SystemParameters
from transradial_sim.model.tendon import PROSTHETIC_TENDON

__all__ = [
    "MAXON_GP26B_84",
    "MAXON_RE25_118752",
    "PROSTHETIC_TENDON",
    "SystemParameters",
    "__version__",
]

__version__ = "0.1.0"
