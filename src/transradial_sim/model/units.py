"""Unit conversions and regularised sign functions used across the model layer.

Every quantity inside the model is stored in SI base units. Manufacturer catalogues
quote motor data in rpm, mNm/A and gcm^2, so the conversions live here and are used
once, at the point where catalogue data is entered.
"""

from __future__ import annotations

import math
from typing import Final

__all__ = [
    "GCM2_PER_KGM2",
    "RAD_S_PER_RPM",
    "gcm2_to_kgm2",
    "kgm2_to_gcm2",
    "mnm_per_a_to_nm_per_a",
    "nm_to_mnm",
    "rad_s_to_rpm",
    "rpm_to_rad_s",
    "smooth_sign",
]

RAD_S_PER_RPM: Final[float] = 2.0 * math.pi / 60.0
"""One revolution per minute expressed in rad/s."""

GCM2_PER_KGM2: Final[float] = 1.0e7
"""One kg m^2 expressed in g cm^2."""


def rpm_to_rad_s(speed_rpm: float) -> float:
    """Convert a rotational speed from rpm to rad/s."""
    return speed_rpm * RAD_S_PER_RPM


def rad_s_to_rpm(speed_rad_s: float) -> float:
    """Convert a rotational speed from rad/s to rpm."""
    return speed_rad_s / RAD_S_PER_RPM


def mnm_per_a_to_nm_per_a(constant_mnm_per_a: float) -> float:
    """Convert a torque constant from mNm/A to Nm/A."""
    return constant_mnm_per_a * 1.0e-3


def nm_to_mnm(torque_nm: float) -> float:
    """Convert a torque from Nm to mNm."""
    return torque_nm * 1.0e3


def gcm2_to_kgm2(inertia_gcm2: float) -> float:
    """Convert a mass moment of inertia from g cm^2 to kg m^2."""
    return inertia_gcm2 / GCM2_PER_KGM2


def kgm2_to_gcm2(inertia_kgm2: float) -> float:
    """Convert a mass moment of inertia from kg m^2 to g cm^2."""
    return inertia_kgm2 * GCM2_PER_KGM2


def smooth_sign(value: float, scale: float) -> float:
    """Return a smooth approximation of ``sign(value)`` saturating over ``scale``.

    Coulomb friction, gearbox loss and capstan friction all reverse direction with the
    sign of a velocity. The discontinuity destroys the order of any explicit integrator
    and makes the energy balance depend on where the step boundaries fall, so each of
    those terms uses ``tanh(value / scale)`` instead. The approximation is exact in the
    limit ``scale -> 0`` and remains dissipative for every ``scale > 0`` because the
    result always carries the sign of ``value``.

    Args:
        value: Velocity-like quantity whose sign selects the friction direction.
        scale: Positive velocity below which the friction is blended through zero.

    Returns:
        A value in ``(-1, 1)`` with the same sign as ``value``.

    Raises:
        ValueError: If ``scale`` is not strictly positive.
    """
    if scale <= 0.0:
        raise ValueError("scale must be strictly positive")
    return math.tanh(value / scale)
