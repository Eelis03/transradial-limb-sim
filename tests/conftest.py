"""Shared fixtures and tolerance helpers for the test suite.

Every tolerance used anywhere in these tests is derived from a measurement scale, never
from an error that happened to be observed. The two scales that matter are the integration
step together with the order of the scheme, and the sample interval of a recorded trace.
Both are turned into numbers by the helpers below, and each test states which one it uses.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from transradial_sim.model.finger import INDEX_FINGER
from transradial_sim.model.gearbox import MAXON_GP26B_84
from transradial_sim.model.motor import MAXON_RE25_118752
from transradial_sim.model.system import PowerSupply, SystemParameters
from transradial_sim.model.tendon import PROSTHETIC_TENDON
from transradial_sim.pipeline.scenario import SOFT_CONTACT, build_plant

REFERENCE_STEP_S = 2.5e-5
"""Integration step used by the tests that are not themselves step studies."""

CONTROL_PERIOD_S = 1.0e-4
"""Controller sample period used throughout the suite."""


SCHEME_ORDER = 4
"""Order of the integrator every tolerance below is derived against."""


def truncation_bound(fastest_rate_rad_s: float, step_s: float) -> float:
    """Return the relative error bound of the scheme on a dissipative run.

    The global error of a scheme of order ``p`` is ``O((lambda h)^p)`` where ``lambda`` is
    the fastest rate the run resolves. On a dissipative system the error does not
    accumulate, because every mode is stable and forgets its own past, so the bound is the
    dimensionless step measure raised to the order and nothing else.

    Args:
        fastest_rate_rad_s: Fastest pole of the plant, in rad/s.
        step_s: Integration step, in s.

    Returns:
        A dimensionless relative error bound.
    """
    return (fastest_rate_rad_s * step_s) ** SCHEME_ORDER


def oscillatory_bound(
    frequency_rad_s: float, step_s: float, duration_s: float
) -> float:
    """Return the relative error bound of the scheme on a conservative run.

    A conservative system has no damping to absorb the local error, so the phase error of
    each mode accumulates linearly in the number of cycles. The bound is therefore the
    dissipative bound multiplied by ``frequency * duration``, which is that cycle count in
    radians.

    Args:
        frequency_rad_s: Natural frequency of the fastest mode, in rad/s.
        step_s: Integration step, in s.
        duration_s: Length of the run, in s.

    Returns:
        A dimensionless relative error bound.
    """
    return truncation_bound(frequency_rad_s, step_s) * frequency_rad_s * duration_s


def lossless_plant() -> SystemParameters:
    """Return a plant with every dissipative term removed.

    Winding resistance, motor friction, gearbox loss, capstan friction, tendon damping,
    joint damping, joint limit damping, battery resistance, bridge resistance and quiescent
    draw are all zero. What remains is an inductor, two inertias and three springs, which is
    a conservative system, so its total energy must not change.
    """
    motor = replace(
        MAXON_RE25_118752,
        resistance_ohm=0.0,
        viscous_friction_nms=0.0,
        coulomb_friction_nm=0.0,
    )
    gearbox = replace(MAXON_GP26B_84, efficiency=1.0)
    tendon = replace(PROSTHETIC_TENDON, friction_coefficient=0.0, damping_ns_per_m=0.0)
    finger = replace(
        INDEX_FINGER,
        phalanges=tuple(
            replace(phalanx, damping_nms_per_rad=0.0) for phalanx in INDEX_FINGER.phalanges
        ),
        limit_damping_nms_per_rad=0.0,
    )
    supply = PowerSupply(
        open_circuit_voltage_v=22.2,
        internal_resistance_ohm=0.0,
        driver_resistance_ohm=0.0,
        driver_quiescent_power_w=0.0,
        max_duty=0.98,
    )
    return SystemParameters(
        supply=supply,
        motor=motor,
        gearbox=gearbox,
        tendon=tendon,
        finger=finger,
        contact=SOFT_CONTACT,
        current_limit_a=1.0e6,
        obstacle=None,
    )


def lossless_natural_rate_rad_s(params: SystemParameters) -> float:
    """Return the fastest natural frequency of the lossless plant, in rad/s.

    The stiffest mode is the electromechanical resonance between the armature inductance
    and the rotor inertia through the torque constant, ``sqrt(k_t^2 / (L J))``.
    """
    torque_constant = params.motor.torque_constant_nm_per_a
    return math.sqrt(
        torque_constant * torque_constant
        / (params.motor.inductance_h * params.rotor_inertia_kgm2)
    )


@pytest.fixture
def reference_plant() -> SystemParameters:
    """The reference prosthesis with no object present."""
    return build_plant()
