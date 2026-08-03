"""Tendon transmission: series elasticity, one way force transmission, capstan friction.

Three properties define a tendon as a transmission element and all three are modelled
here.

One way transmission. A tendon pulls and cannot push. The tension is therefore clamped
at zero, and the finger becomes mechanically disconnected from the motor whenever the
motor pays out cord faster than the return springs take it up. Any model that omits the
clamp will silently produce a bidirectional rod.

Series elasticity. The cord, its terminations and the routing all stretch under load.
The whole compliance is lumped into one spring of stiffness ``k`` acting on the
difference between the cord paid out by the drive pulley and the cord consumed by the
finger joints. A small parallel damper represents the viscoelastic loss of a polymer
cord and keeps the contact transients bounded.

Capstan friction. Where the cord runs over a guide it presses on the guide, and the
resulting friction changes the tension along the path. Integrating the friction around
an element of wrap angle gives the capstan, or Euler-Eytelwein, relation

    ``T_high = T_low * exp(mu * theta)``

with ``theta`` the total wrap angle in radians. The exponent is the reason a tendon
driven hand loses a large fraction of its actuation force in the routing, and the reason
the loss cannot be reduced by increasing the tension.

Lost motion. The gearhead in front of the drive pulley has angular play, so a length of
cord equal to that play times the pulley radius has to be wound in before the series
element begins to stretch at all. It is entered here as ``lost_motion_m`` rather than as a
gearbox state, and the reason it can be is given in :func:`evaluate_tendon`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from transradial_sim.model.units import smooth_sign

__all__ = [
    "PROSTHETIC_ROUTING",
    "PROSTHETIC_TENDON",
    "RoutingSegment",
    "TendonParameters",
    "TendonState",
    "capstan_ratio",
    "evaluate_tendon",
    "total_wrap_angle",
]


@dataclass(frozen=True, slots=True)
class RoutingSegment:
    """One guide the tendon wraps around on its way from the drive pulley to the finger."""

    name: str
    """Human readable location of the guide."""

    wrap_angle_rad: float
    """Angle subtended by the contact arc, in rad."""

    def __post_init__(self) -> None:
        if self.wrap_angle_rad < 0.0:
            raise ValueError("wrap_angle_rad must be non negative")


PROSTHETIC_ROUTING: Final[tuple[RoutingSegment, ...]] = (
    RoutingSegment(name="forearm exit guide", wrap_angle_rad=1.20),
    RoutingSegment(name="wrist pass through", wrap_angle_rad=0.90),
    RoutingSegment(name="metacarpophalangeal idler", wrap_angle_rad=0.80),
    RoutingSegment(name="proximal interphalangeal idler", wrap_angle_rad=0.60),
)
"""Routing of one finger tendon from a forearm mounted drive to the fingertip.

Total wrap angle 3.50 rad, which is 201 degrees. The path corresponds to a drive unit in
the forearm shell, a guide at the wrist, and one idler at each of the two proximal
joints.
"""


@dataclass(frozen=True, slots=True)
class TendonParameters:
    """Parameters of the tendon transmission, in SI units."""

    drive_radius_m: float
    """Radius of the drive pulley on the gearbox output, in m."""

    stiffness_n_per_m: float
    """Lumped series stiffness of cord, terminations and routing, in N/m."""

    damping_ns_per_m: float
    """Parallel viscous damping of the series element, in N s/m."""

    friction_coefficient: float
    """Coulomb friction coefficient between the cord and the guides."""

    routing: tuple[RoutingSegment, ...]
    """Guides the cord wraps around, in order from drive to finger."""

    velocity_regularisation_m_per_s: float
    """Speed scale of the sliding branch, in m/s.

    Capstan friction is Coulomb friction and its magnitude does not fall off with speed, so
    this scale is set well below the slowest cord motion of interest. Below
    :attr:`stick_velocity_m_per_s` the model switches to the impending sliding direction
    instead, so the loss does not vanish when the drive stops.
    """

    stick_velocity_m_per_s: float = 5.0e-5
    """Cord speed below which the routing friction sticks rather than slides, in m/s.

    Inside the band the direction of the friction is taken from the direction the drive is
    pushing rather than from the measured speed, which is what keeps the full capstan loss
    in place when a grasp has settled and the cord has stopped moving.
    """

    def __post_init__(self) -> None:
        if self.drive_radius_m <= 0.0:
            raise ValueError("drive_radius_m must be strictly positive")
        if self.stiffness_n_per_m <= 0.0:
            raise ValueError("stiffness_n_per_m must be strictly positive")
        if self.friction_coefficient < 0.0:
            raise ValueError("friction_coefficient must be non negative")

    @property
    def wrap_angle_rad(self) -> float:
        """Total wrap angle over the whole routing path, in rad."""
        return total_wrap_angle(self.routing)


def total_wrap_angle(routing: tuple[RoutingSegment, ...]) -> float:
    """Return the sum of the wrap angles of every guide, in rad."""
    return math.fsum(segment.wrap_angle_rad for segment in routing)


def capstan_ratio(friction_coefficient: float, wrap_angle_rad: float) -> float:
    """Return ``exp(-mu * theta)``, the tension ratio across the routing when pulling.

    The cord leaves the routing with this fraction of the tension applied at the drive
    end. The reciprocal applies when the finger drives the cord back through the routing.

    Args:
        friction_coefficient: Coulomb friction coefficient between cord and guide.
        wrap_angle_rad: Total wrap angle in rad.

    Returns:
        A value in ``(0, 1]``, equal to one when either argument is zero.
    """
    return math.exp(-friction_coefficient * wrap_angle_rad)


@dataclass(frozen=True, slots=True)
class TendonState:
    """Instantaneous tendon quantities derived from the plant state."""

    extension_m: float
    """Relative displacement of the two ends of the transmission, in m.

    Negative means the cord is slack. This is the drive displacement minus the finger
    displacement, before the gearhead lost motion is taken out of it.
    """

    elastic_extension_m: float
    """Stretch actually carried by the series spring, in m.

    :attr:`extension_m` less the lost motion, clamped at zero. The two differ only by the
    width of the backlash dead band, and they are reported separately so that the play can
    be read off a trace rather than inferred from it.
    """

    extension_rate_m_per_s: float
    """Rate of change of the stretch, in m/s."""

    drive_velocity_m_per_s: float
    """Speed at which the drive pulley takes cord in, in m/s."""

    finger_tension_n: float
    """Tension seen by the finger, in N. Never negative."""

    drive_tension_n: float
    """Tension the drive pulley has to apply, in N. Never negative."""

    friction_power_w: float
    """Power dissipated by capstan friction, in W. Never negative."""

    damper_power_w: float
    """Power dissipated by the series damper, in W. Never negative."""

    stored_energy_j: float
    """Elastic energy held in the series element, in J."""


def evaluate_tendon(
    tendon: TendonParameters,
    drive_displacement_m: float,
    drive_velocity_m_per_s: float,
    finger_displacement_m: float,
    finger_velocity_m_per_s: float,
    impending_direction: float = 0.0,
    lost_motion_m: float = 0.0,
) -> TendonState:
    """Evaluate the tendon transmission for one instant.

    The series element is placed downstream of the routing, so the tension it carries is
    the tension the finger sees, and the drive end has to supply that tension multiplied
    by the capstan factor in whichever direction the cord is moving.

    ``lost_motion_m`` is the cord travel absorbed by the angular play of the gearhead
    ahead of the drive pulley. It enters as a dead band on the extension rather than as a
    hysteresis state, and for this transmission the two are the same thing. The play opens
    only when the sign of the load torque on the pulley reverses, the load here is the
    tendon tension, and a tendon pulls and cannot push, so the torque never reverses while
    the cord is taut and the teeth stay on one flank throughout a closure. The play is
    therefore taken up once, at the start of each closure, and is given back only when the
    cord goes slack, which is exactly what a dead band on the extension does. Modelling it
    as a stateful hysteresis would add a discontinuous state and would produce the same
    trajectory.

    Args:
        tendon: Transmission parameters.
        drive_displacement_m: Cord length paid in by the drive pulley, in m.
        drive_velocity_m_per_s: Rate of change of ``drive_displacement_m``, in m/s.
        finger_displacement_m: Cord length consumed by the finger joints, in m.
        finger_velocity_m_per_s: Rate of change of ``finger_displacement_m``, in m/s.
        impending_direction: Sign of the direction the drive is pushing the cord. Used
            only inside the stick band, where the measured speed carries no information
            about which way the cord is about to slide.
        lost_motion_m: Width of the backlash dead band referred to the cord, in m. Zero
            for an ideal gearhead.

    Returns:
        The tendon state, with both tensions clamped at zero and both loss powers
        guaranteed non negative.
    """
    extension = drive_displacement_m - finger_displacement_m
    extension_rate = drive_velocity_m_per_s - finger_velocity_m_per_s
    elastic = extension - lost_motion_m

    if elastic > 0.0:
        spring_force = tendon.stiffness_n_per_m * elastic
        stored = 0.5 * spring_force * elastic
        damper_force = tendon.damping_ns_per_m * extension_rate
    else:
        # Slack cord, or play not yet taken up. Neither the spring nor the damper is
        # engaged, and no energy is stored or dissipated inside the dead band.
        elastic = 0.0
        spring_force = 0.0
        stored = 0.0
        damper_force = 0.0

    # A tendon pulls and cannot push, so the tension is clamped at zero. The damper power
    # is defined as the residual, which keeps the identity
    #   finger_tension * extension_rate = d(stored)/dt + damper_power
    # exact in both the engaged and the clamped branch.
    finger_tension = max(0.0, spring_force + damper_force)
    damper_power = (finger_tension - spring_force) * extension_rate

    if abs(drive_velocity_m_per_s) >= tendon.stick_velocity_m_per_s:
        direction = smooth_sign(
            drive_velocity_m_per_s, tendon.velocity_regularisation_m_per_s
        )
    else:
        direction = impending_direction
        if direction * drive_velocity_m_per_s < 0.0:
            # Keep the friction dissipative even when the cord momentarily backs up
            # inside the stick band.
            direction = -direction
    exponent = tendon.friction_coefficient * tendon.wrap_angle_rad * direction
    drive_tension = finger_tension * math.exp(exponent)
    friction_power = (drive_tension - finger_tension) * drive_velocity_m_per_s

    return TendonState(
        extension_m=extension,
        elastic_extension_m=elastic,
        extension_rate_m_per_s=extension_rate,
        drive_velocity_m_per_s=drive_velocity_m_per_s,
        finger_tension_n=finger_tension,
        drive_tension_n=drive_tension,
        friction_power_w=friction_power,
        damper_power_w=damper_power,
        stored_energy_j=stored,
    )


PROSTHETIC_TENDON: Final[TendonParameters] = TendonParameters(
    drive_radius_m=8.0e-3,
    stiffness_n_per_m=2.0e4,
    damping_ns_per_m=1.6,
    friction_coefficient=0.147,
    routing=PROSTHETIC_ROUTING,
    velocity_regularisation_m_per_s=2.0e-6,
)
"""Reference tendon: a stainless steel rope running through metal guides in the forearm.

The friction coefficient of 0.147 is not a handbook figure. It is the value Agrawal, Peine
and Yao measured directly on a 0.52 mm uncoated stainless steel 7 by 19 rope running in a
1.2 mm inner diameter stainless conduit over 12 mm pulleys, at pretensions from 0.7 N to
7.3 N. The reference tendon here is specified as the same construction so that the
measurement applies to the geometry being modelled rather than to a similar one. A model
fit to the same data back calculates 0.156.

Nahvi, Hollerbach, Xu and Hunter report 0.13 for a tendon over the routing pulleys of a
dexterous hand, which brackets the same range, but it was measured on 12.4 mm pulleys with
a 3 mm bore and is therefore an effective coefficient dominated by the bearing rather than
by sliding of the cord.

No peer reviewed source reports a directly measured coefficient for an ultra high molecular
weight polyethylene cord on steel or aluminium, so a polymer tendon is not modelled here by
a friction coefficient. Friedl, Chalon, Reinecke and Grebenstein instead report losses per
routing element, about 1.1 percent for steel and about 2.5 to 4.75 percent for Dyneema,
reaching 30 percent in total across seven pulleys.

With the reference wrap angle of 3.50 rad the coefficient gives a pulling transmission ratio
of 0.598, that is a 40 percent loss in the routing alone. Two independent measurements on
assembled hands agree: Li and colleagues measured 60 percent force transmission efficiency
on a tendon driven anthropomorphic finger at 10 N, and Grosu and colleagues report 64 to 76
percent for a cable driven actuator.

The stiffness is the lumped value for the rope together with its terminations, which are the
dominant compliance in a short run. The damping follows from a structural loss factor of
0.05 evaluated at 100 Hz, ``c = 0.05 * k / omega``, which is the usual way the hysteretic
loss of a stranded rope and its crimps is expressed.
"""
