"""Brushed direct current motor model with electrical and mechanical dynamics.

The model is the standard two-state lumped description of a permanent magnet brushed
motor: a resistive-inductive armature circuit coupled to a rotating inertia through the
torque constant and the back electromotive force constant.

Electrical:
    ``L di/dt = v - R i - k_e omega``

Mechanical:
    ``J domega/dt = k_t i - b omega - tau_c sign(omega) - tau_load``

The torque constant ``k_t`` in Nm/A and the back electromotive force constant ``k_e`` in
V s/rad are numerically equal for an ideal machine in SI units, which is the identity a
unit error in the catalogue conversion would break. The catalogue parameters are entered
once, in :data:`MAXON_RE25_118752`, and converted from catalogue units by
:mod:`transradial_sim.model.units`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from transradial_sim.model.units import (
    gcm2_to_kgm2,
    mnm_per_a_to_nm_per_a,
    rpm_to_rad_s,
    smooth_sign,
)

__all__ = [
    "MAXON_RE25_118752",
    "CatalogueOperatingPoint",
    "MotorParameters",
    "current_derivative",
    "electromagnetic_torque",
    "friction_torque",
    "predicted_no_load_speed",
    "predicted_stall_torque",
]


@dataclass(frozen=True, slots=True)
class CatalogueOperatingPoint:
    """Operating points quoted by the manufacturer, held for external validation.

    These values are never used by the dynamics. They exist so that the simulation can
    be checked against numbers the manufacturer measured, rather than only against
    itself.
    """

    part_number: str
    """Manufacturer order number."""

    nominal_voltage_v: float
    """Catalogue nominal voltage in V."""

    no_load_speed_rad_s: float
    """Catalogue no load speed in rad/s."""

    no_load_current_a: float
    """Catalogue no load current in A."""

    stall_torque_nm: float
    """Catalogue stall torque in Nm."""

    stall_current_a: float
    """Catalogue stall current in A."""

    max_continuous_current_a: float
    """Catalogue maximum continuous (nominal) current in A."""

    max_efficiency: float
    """Catalogue maximum efficiency as a fraction."""


@dataclass(frozen=True, slots=True)
class MotorParameters:
    """Physical parameters of a brushed direct current motor, in SI units.

    The viscous and Coulomb friction coefficients are not published separately by any
    manufacturer. They are derived from the catalogue no load current, which fixes their
    sum at the catalogue no load speed, together with an assumed Coulomb share. See
    :func:`friction_from_no_load_point`.
    """

    resistance_ohm: float
    """Terminal resistance of the armature circuit in ohm."""

    inductance_h: float
    """Terminal inductance of the armature circuit in H."""

    torque_constant_nm_per_a: float
    """Torque constant ``k_t`` in Nm/A."""

    rotor_inertia_kgm2: float
    """Rotor mass moment of inertia in kg m^2."""

    viscous_friction_nms: float
    """Viscous friction coefficient in Nm s/rad."""

    coulomb_friction_nm: float
    """Coulomb friction torque magnitude in Nm."""

    speed_regularisation_rad_s: float
    """Speed scale over which Coulomb friction is blended through zero, in rad/s."""

    catalogue: CatalogueOperatingPoint
    """Catalogue operating points for the same part, used only for validation."""

    @property
    def back_emf_constant_v_s_per_rad(self) -> float:
        """Back electromotive force constant ``k_e`` in V s/rad.

        For an ideal machine ``k_e`` and ``k_t`` are the same number in SI units because
        both express the same magnetic flux linkage. The model therefore does not carry
        a second parameter, and the SI consistency of the catalogue conversion is checked
        against the independently published speed constant.
        """
        return self.torque_constant_nm_per_a

    @property
    def electrical_time_constant_s(self) -> float:
        """Armature electrical time constant ``L / R`` in s."""
        return self.inductance_h / self.resistance_ohm


def friction_from_no_load_point(
    torque_constant_nm_per_a: float,
    no_load_current_a: float,
    no_load_speed_rad_s: float,
    coulomb_share: float,
) -> tuple[float, float]:
    """Split the catalogue no load torque into viscous and Coulomb components.

    At the catalogue no load point the electromagnetic torque exactly balances the
    internal friction, so ``k_t * i_0 = b * omega_0 + tau_c``. That single equation
    cannot separate the two terms, so ``coulomb_share`` fixes the fraction carried by
    the speed independent brush and bearing drag, and the viscous coefficient follows.

    Args:
        torque_constant_nm_per_a: Torque constant in Nm/A.
        no_load_current_a: Catalogue no load current in A.
        no_load_speed_rad_s: Catalogue no load speed in rad/s.
        coulomb_share: Fraction of the no load torque assigned to Coulomb friction.

    Returns:
        The viscous coefficient in Nm s/rad and the Coulomb torque in Nm.

    Raises:
        ValueError: If ``coulomb_share`` is outside ``[0, 1]`` or the speed is not
            strictly positive.
    """
    if not 0.0 <= coulomb_share <= 1.0:
        raise ValueError("coulomb_share must lie in [0, 1]")
    if no_load_speed_rad_s <= 0.0:
        raise ValueError("no_load_speed_rad_s must be strictly positive")
    no_load_torque = torque_constant_nm_per_a * no_load_current_a
    coulomb = coulomb_share * no_load_torque
    viscous = (no_load_torque - coulomb) / no_load_speed_rad_s
    return viscous, coulomb


_RE25_TORQUE_CONSTANT: Final[float] = mnm_per_a_to_nm_per_a(23.4)
_RE25_NO_LOAD_SPEED: Final[float] = rpm_to_rad_s(9560.0)
_RE25_NO_LOAD_CURRENT: Final[float] = 36.9e-3
_RE25_VISCOUS, _RE25_COULOMB = friction_from_no_load_point(
    torque_constant_nm_per_a=_RE25_TORQUE_CONSTANT,
    no_load_current_a=_RE25_NO_LOAD_CURRENT,
    no_load_speed_rad_s=_RE25_NO_LOAD_SPEED,
    coulomb_share=0.35,
)

MAXON_RE25_118752: Final[MotorParameters] = MotorParameters(
    resistance_ohm=2.32,
    inductance_h=0.238e-3,
    torque_constant_nm_per_a=_RE25_TORQUE_CONSTANT,
    rotor_inertia_kgm2=gcm2_to_kgm2(10.8),
    viscous_friction_nms=_RE25_VISCOUS,
    coulomb_friction_nm=_RE25_COULOMB,
    speed_regularisation_rad_s=2.0,
    catalogue=CatalogueOperatingPoint(
        part_number="maxon RE 25, order number 118752",
        nominal_voltage_v=24.0,
        no_load_speed_rad_s=_RE25_NO_LOAD_SPEED,
        no_load_current_a=_RE25_NO_LOAD_CURRENT,
        stall_torque_nm=243.0e-3,
        stall_current_a=10.4,
        max_continuous_current_a=1.16,
        max_efficiency=0.85,
    ),
)
"""maxon RE 25, 25 mm, graphite brushes, 20 W, order number 118752, 24 V winding.

Catalogue values: terminal resistance 2.32 ohm, terminal inductance 0.238 mH, torque
constant 23.4 mNm/A, speed constant 408 rpm/V, rotor inertia 10.8 g cm^2, no load speed
9560 rpm, no load current 36.9 mA, stall torque 243 mNm, stall current 10.4 A, maximum
continuous current 1.16 A, maximum efficiency 85 percent.
"""

CATALOGUE_SPEED_CONSTANT_RPM_PER_V: Final[float] = 408.0
"""Catalogue speed constant of :data:`MAXON_RE25_118752` in rpm/V.

Published independently of the torque constant, so comparing the two is an external
check that the SI conversion is right rather than a restatement of it.
"""


def electromagnetic_torque(motor: MotorParameters, current_a: float) -> float:
    """Return the electromagnetic torque produced by an armature current, in Nm."""
    return motor.torque_constant_nm_per_a * current_a


def friction_torque(motor: MotorParameters, speed_rad_s: float) -> float:
    """Return the internal friction torque opposing rotation, in Nm.

    The result always carries the sign of ``speed_rad_s``, so the friction power
    ``friction_torque * speed`` is non negative at every speed. That property is what
    makes the energy balance close with a single dissipated term.
    """
    coulomb = motor.coulomb_friction_nm * smooth_sign(
        speed_rad_s, motor.speed_regularisation_rad_s
    )
    return motor.viscous_friction_nms * speed_rad_s + coulomb


def current_derivative(
    motor: MotorParameters,
    applied_voltage_v: float,
    current_a: float,
    speed_rad_s: float,
) -> float:
    """Return ``di/dt`` for the armature circuit, in A/s."""
    back_emf = motor.back_emf_constant_v_s_per_rad * speed_rad_s
    return (applied_voltage_v - motor.resistance_ohm * current_a - back_emf) / motor.inductance_h


def holding_voltage(
    motor: MotorParameters,
    current_a: float,
    speed_rad_s: float,
) -> float:
    """Return the terminal voltage at which ``di/dt`` is exactly zero, in V.

    A hardware current limiter works by reducing the applied voltage to this value once
    the measured current reaches the trip level. Returning it explicitly lets the
    limiter be exact in the continuous time model rather than approximate.
    """
    return motor.resistance_ohm * current_a + motor.back_emf_constant_v_s_per_rad * speed_rad_s


def predicted_no_load_speed(motor: MotorParameters, supply_voltage_v: float) -> float:
    """Return the steady state unloaded speed predicted by the model, in rad/s.

    Solves the coupled steady state of the electrical and mechanical equations with no
    external load, including both friction terms. Coulomb friction is taken at its full
    magnitude because the no load speed is far above the regularisation scale.
    """
    k = motor.torque_constant_nm_per_a
    resistance = motor.resistance_ohm
    denominator = k + resistance * motor.viscous_friction_nms / k
    numerator = supply_voltage_v - resistance * motor.coulomb_friction_nm / k
    return numerator / denominator


def predicted_no_load_current(motor: MotorParameters, supply_voltage_v: float) -> float:
    """Return the steady state unloaded armature current predicted by the model, in A."""
    speed = predicted_no_load_speed(motor, supply_voltage_v)
    return friction_torque(motor, speed) / motor.torque_constant_nm_per_a


def predicted_stall_torque(motor: MotorParameters, supply_voltage_v: float) -> float:
    """Return the shaft torque at zero speed predicted by the model, in Nm.

    At zero speed the back electromotive force vanishes, so the armature current is
    ``v / R``. Coulomb friction is subtracted at its full magnitude, which is the torque
    a locked shaft has to be held against once it has just started to move.
    """
    current = supply_voltage_v / motor.resistance_ohm
    return electromagnetic_torque(motor, current) - motor.coulomb_friction_nm
