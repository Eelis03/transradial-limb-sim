"""Planetary gearbox model with reduction ratio, reflected inertia and loss.

The gearbox is described by a reduction ratio ``N``, a mass moment of inertia already
referred to the motor shaft, and a maximum efficiency ``eta``. Gear tooth friction is
modelled as a load dependent Coulomb loss torque opposing rotation,

    ``tau_loss = (1 / eta - 1) * |tau_output| / N * sign(omega_motor)``

The coefficient follows from the catalogue definition of efficiency and not the other way
round. In steady forward motion the motor torque is ``tau_output / N + tau_loss``, which
rearranges to ``tau_output = eta * N * tau_motor``, so the model reproduces the catalogue
efficiency exactly during motion. Writing the coefficient as ``1 - eta`` instead, which is
the common slip, understates the loss by a factor of ``eta`` and for a three stage
gearhead at 59 percent that is a 41 percent error in the loss term.

The sign of the loss follows the motor speed, so the dissipated power is non negative at
every instant, which is what allows the whole simulation to satisfy an exact energy
balance.

Below a small speed the model switches to stiction, following Karnopp: the friction torque
takes whatever value inside its capacity cancels the net driving torque, so a stalled
drive holds instead of creeping. Without it a regularised sign function returns zero
friction at zero speed, the drive keeps turning, and the tendon tension climbs towards the
frictionless limit, which for this transmission is nearly three times the correct value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from transradial_sim.model.units import gcm2_to_kgm2, rpm_to_rad_s, smooth_sign

__all__ = [
    "MAXON_GP26B_84",
    "GearboxParameters",
    "loss_torque_on_motor",
    "reflected_load_torque",
]


@dataclass(frozen=True, slots=True)
class GearboxParameters:
    """Reduction gearbox parameters, in SI units."""

    ratio: float
    """Reduction ratio ``N``, output speed is motor speed divided by ``N``."""

    efficiency: float
    """Maximum efficiency as a fraction, from the manufacturer catalogue."""

    inertia_kgm2: float
    """Mass moment of inertia referred to the motor shaft, in kg m^2."""

    max_continuous_output_torque_nm: float
    """Catalogue maximum continuous output torque in Nm."""

    max_intermittent_output_torque_nm: float
    """Catalogue intermittently permissible output torque in Nm."""

    max_input_speed_rad_s: float
    """Catalogue recommended maximum input speed in rad/s."""

    part_number: str
    """Manufacturer order number."""

    speed_regularisation_rad_s: float = 0.02
    """Speed scale of the sliding branch, in rad/s.

    Small enough that the sliding friction is at 99 percent of its capacity by the time
    the speed leaves the stick band, and large enough that the induced pole,
    ``tau_capacity / (omega_eps * J_rotor)``, stays well inside the stability region of the
    integrator at the default step.
    """

    stick_speed_rad_s: float = 0.5
    """Motor speed below which the gear friction sticks rather than slides, in rad/s.

    At the reference reduction this is 48 micrometres per second of tendon travel, which is one
    twentieth of one percent of the free running cord speed. The band is set wide enough that
    the sliding friction has already saturated at its edge, so the drive sticks at the torque
    the catalogue efficiency implies rather than somewhere above it. A stuck drive moves less
    than one newton's worth of tendon stretch per second.
    """

    def __post_init__(self) -> None:
        if self.ratio <= 0.0:
            raise ValueError("ratio must be strictly positive")
        if not 0.0 < self.efficiency <= 1.0:
            raise ValueError("efficiency must lie in (0, 1]")


MAXON_GP26B_84: Final[GearboxParameters] = GearboxParameters(
    ratio=84.0,
    efficiency=0.59,
    inertia_kgm2=gcm2_to_kgm2(0.4),
    max_continuous_output_torque_nm=1.3,
    max_intermittent_output_torque_nm=1.9,
    max_input_speed_rad_s=rpm_to_rad_s(8000.0),
    part_number="maxon GP 26 B, order number 144039",
)
"""maxon planetary gearhead GP 26 B, 26 mm, three stages, 84:1.

Catalogue values: reduction 84:1, three stages, maximum efficiency 59 percent, maximum
continuous output torque 1.3 Nm, intermittently permissible output torque 1.9 Nm, mass
inertia 0.4 g cm^2 referred to the motor shaft, recommended maximum input speed 8000 rpm.
"""


def reflected_load_torque(gearbox: GearboxParameters, output_torque_nm: float) -> float:
    """Return the output load torque referred to the motor shaft, in Nm.

    This is the ideal kinematic reflection only. The friction loss is a separate term so
    that it can be accounted for as dissipated power; see :func:`loss_torque_on_motor`.
    """
    return output_torque_nm / gearbox.ratio


def loss_torque_on_motor(
    gearbox: GearboxParameters,
    output_torque_nm: float,
    motor_speed_rad_s: float,
    driving_torque_nm: float = 0.0,
) -> float:
    """Return the gear friction torque acting on the motor shaft, in Nm.

    The capacity is ``(1 / eta - 1)`` times the transmitted torque referred to the motor
    shaft, which is the coefficient implied by the catalogue definition of efficiency.

    Above the stick speed the friction slides and its sign follows the motor speed. Inside
    the stick band it takes whatever value within its capacity cancels the net driving
    torque, which is what holds a stalled drive still. The stick branch is used only when
    it is dissipative, that is when the driving torque and the speed agree in sign, so the
    friction power is non negative in every branch and the energy balance stays exact.

    Args:
        gearbox: Gearbox parameters.
        output_torque_nm: Torque transmitted at the gearbox output, in Nm.
        motor_speed_rad_s: Motor shaft speed, in rad/s.
        driving_torque_nm: Net torque on the rotor before gear friction, in Nm.

    Returns:
        The friction torque acting on the motor shaft, in Nm.
    """
    capacity = (1.0 / gearbox.efficiency - 1.0) * abs(output_torque_nm) / gearbox.ratio
    if (
        abs(motor_speed_rad_s) <= gearbox.stick_speed_rad_s
        and driving_torque_nm * motor_speed_rad_s >= 0.0
    ):
        return max(-capacity, min(capacity, driving_torque_nm))
    return capacity * smooth_sign(motor_speed_rad_s, gearbox.speed_regularisation_rad_s)


def output_speed(gearbox: GearboxParameters, motor_speed_rad_s: float) -> float:
    """Return the gearbox output speed in rad/s."""
    return motor_speed_rad_s / gearbox.ratio


def current_limit_from_gearbox(
    gearbox: GearboxParameters,
    torque_constant_nm_per_a: float,
) -> float:
    """Return the motor current at which the gearbox reaches its continuous rating, in A.

    The gearbox, not the motor, often sets the usable current in a compact prosthetic
    drive. Computing the limit from the catalogue rating makes that visible rather than
    leaving it as a chosen constant.
    """
    motor_torque = gearbox.max_continuous_output_torque_nm / (gearbox.ratio * gearbox.efficiency)
    return motor_torque / torque_constant_nm_per_a
