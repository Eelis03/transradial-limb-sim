"""Standard plant configurations and the scenarios reported in the README.

Every number that defines the reference prosthesis is set once here, so a sensitivity
study can be written as a transformation of this configuration rather than as a second
copy of it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from transradial_sim.algorithm.controllers import CurrentController, ScheduledCurrentController
from transradial_sim.model.contact import (
    CircularObject,
    ContactGeometry,
    ContactModel,
    FlatSurface,
)
from transradial_sim.model.finger import INDEX_FINGER
from transradial_sim.model.gearbox import MAXON_GP26B_84, current_limit_from_gearbox
from transradial_sim.model.motor import MAXON_RE25_118752
from transradial_sim.model.system import PowerSupply, SystemParameters
from transradial_sim.model.tendon import PROSTHETIC_TENDON
from transradial_sim.pipeline.simulate import ScenarioConfig

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
    "build_plant",
    "closing_scenario",
    "current_controller",
    "grasp_controller",
    "grasp_scenario",
    "reference_current_limit_a",
    "stall_scenario",
]

PROSTHESIS_SUPPLY: Final[PowerSupply] = PowerSupply(
    open_circuit_voltage_v=22.2,
    internal_resistance_ohm=0.15,
    driver_resistance_ohm=0.08,
    driver_quiescent_power_w=0.25,
    max_duty=0.98,
)
"""Six cell lithium ion pack and an H bridge sized for the motor.

22.2 V is the nominal open circuit voltage of six lithium ion cells in series, which is
the usual pack for a 24 V winding. The internal resistance is that of a small capacity
pack of cylindrical cells, the bridge conduction resistance is two low voltage MOSFETs in
the current path, and the quiescent power covers gate drive and logic. These three numbers
are representative rather than taken from a datasheet, and they are named separately in
the loss table so their share can be judged.
"""

SOFT_CONTACT: Final[ContactModel] = ContactModel(
    stiffness_n_per_m_pow=4.0e4,
    exponent=1.5,
    damping_s_per_m=8.0,
    samples_per_phalanx=3,
)
"""Contact against a compliant object, a soft plastic or a firm foam.

At 3 mm of indentation this gives about 6.6 N per contact point, which is the range in
which a prosthetic hand grasps a deformable object. The exponent is the Hertzian value.
"""

COMPLIANT_CONTACT: Final[ContactModel] = ContactModel(
    stiffness_n_per_m_pow=5.0e3,
    exponent=1.5,
    damping_s_per_m=8.0,
    samples_per_phalanx=3,
)
"""Contact against a soft object such as a foam block or a sponge.

Eight times more compliant than :data:`SOFT_CONTACT`. Reaching 5 N takes about 10 mm of
indentation, which puts the force gain low enough that closed loop force control has
usable stability margins. It is the object the force control example uses.
"""

STIFF_CONTACT: Final[ContactModel] = ContactModel(
    stiffness_n_per_m_pow=2.0e6,
    exponent=1.5,
    damping_s_per_m=2.0,
    samples_per_phalanx=3,
)
"""Contact against a rigid object, used for the stall force measurement.

Fifty times stiffer than :data:`SOFT_CONTACT`, which puts the indentation at a few tens of
micrometres and makes the grasp force a property of the transmission rather than of the
object.
"""

FLAT_PLATE: Final[FlatSurface] = FlatSurface(point_m=(0.0, 0.028), normal=(0.0, -1.0))
"""A flat face 28 mm above the finger base, the shape a book or a shelf presents."""

LARGE_CYLINDER: Final[CircularObject] = CircularObject(centre_m=(0.020, 0.038), radius_m=0.030)
"""A 60 mm diameter cylinder, the size of a drinks can."""

SMALL_CYLINDER: Final[CircularObject] = CircularObject(centre_m=(0.005, 0.028), radius_m=0.020)
"""A 40 mm diameter cylinder, the size of a jar lid."""

TEST_OBJECTS: Final[tuple[tuple[str, ContactGeometry], ...]] = (
    ("flat plate", FLAT_PLATE),
    ("60 mm cylinder", LARGE_CYLINDER),
    ("40 mm cylinder", SMALL_CYLINDER),
)
"""The object set used for the adaptive grasp measurement.

The three shapes are chosen so that a finger which conforms will settle into three clearly
different postures, and one which does not will settle into the same one three times.
"""


def reference_current_limit_a() -> float:
    """Return the drive current limit of the reference prosthesis, in A.

    The limit is not chosen. It is the current at which the gearbox reaches its catalogue
    continuous output torque, which for this pairing is below the motor's own continuous
    current rating. Reporting it this way makes visible which component sets the limit.
    """
    return current_limit_from_gearbox(MAXON_GP26B_84, MAXON_RE25_118752.torque_constant_nm_per_a)


def build_plant(
    obstacle: ContactGeometry | None = None,
    contact: ContactModel | None = None,
    tendon_stiffness_n_per_m: float | None = None,
    tendon_friction_coefficient: float | None = None,
    current_limit_a: float | None = None,
) -> SystemParameters:
    """Return the reference plant, with named parameters optionally overridden.

    Args:
        obstacle: Object to grasp, or ``None`` for free motion.
        contact: Contact law, :data:`SOFT_CONTACT` by default.
        tendon_stiffness_n_per_m: Overrides the tendon series stiffness.
        tendon_friction_coefficient: Overrides the capstan friction coefficient.
        current_limit_a: Overrides the drive current limit.

    Returns:
        The assembled plant parameters.
    """
    tendon = PROSTHETIC_TENDON
    if tendon_stiffness_n_per_m is not None:
        tendon = replace(tendon, stiffness_n_per_m=tendon_stiffness_n_per_m)
    if tendon_friction_coefficient is not None:
        tendon = replace(tendon, friction_coefficient=tendon_friction_coefficient)
    return SystemParameters(
        supply=PROSTHESIS_SUPPLY,
        motor=MAXON_RE25_118752,
        gearbox=MAXON_GP26B_84,
        tendon=tendon,
        finger=INDEX_FINGER,
        contact=contact if contact is not None else SOFT_CONTACT,
        current_limit_a=(
            current_limit_a if current_limit_a is not None else reference_current_limit_a()
        ),
        obstacle=obstacle,
    )


def current_controller(
    params: SystemParameters,
    reference_a: float | None = None,
    sample_period_s: float = 1.0e-4,
    speed_limit_rad_s: float | None = None,
) -> CurrentController:
    """Return the inner current regulator configured for a plant."""
    controller = CurrentController(
        motor=params.motor,
        supply_voltage_v=params.supply.open_circuit_voltage_v,
        current_limit_a=params.current_limit_a,
        sample_period_s=sample_period_s,
        speed_limit_rad_s=speed_limit_rad_s,
    )
    controller.set_reference(
        reference_a if reference_a is not None else params.current_limit_a
    )
    return controller


APPROACH_CURRENT_A: Final[float] = 0.35
"""Current used while the finger closes, before it touches the object, in A.

About a third of the limit. Enough to close in well under a second against the return
springs, low enough that the finger arrives at the object with little kinetic energy.
"""

SQUEEZE_START_S: Final[float] = 1.00
"""Time at which the drive switches from approach current to grasp current, in s."""

APPROACH_SPEED_LIMIT_RAD_S: Final[float] = 400.0
"""Motor speed the drive holds while closing, in rad/s.

Forty four percent of the free running speed, which is 38 mm/s of tendon travel and closes
the finger in 0.80 s. It is a compromise between closing time and the energy the finger
carries into the object. At the free running speed the rotor holds 0.47 J, and a tendon
that absorbs it stretches to a tension the motor could never have produced and cannot
undo, because a tendon does not push. At this limit the rotor carries 0.09 J and the
residual stretch is worth about 28 N.

For scale, Belter and colleagues tabulate fingertip speeds rather than closing times for
commercial hands. The times those speeds imply, derived here rather than quoted by them,
are about 0.33 s for the Ottobock SensorHand and about 0.37 s for the Michelangelo, which
is the range this drive reaches when the speed limiter is not engaged.
"""


def grasp_controller(
    params: SystemParameters,
    approach_a: float = APPROACH_CURRENT_A,
    squeeze_start_s: float = SQUEEZE_START_S,
    sample_period_s: float = 1.0e-4,
) -> ScheduledCurrentController:
    """Return the two phase approach and squeeze controller used for grasp scenarios."""
    return ScheduledCurrentController(
        inner=current_controller(
            params,
            approach_a,
            sample_period_s,
            speed_limit_rad_s=APPROACH_SPEED_LIMIT_RAD_S,
        ),
        schedule=((0.0, approach_a), (squeeze_start_s, params.current_limit_a)),
    )


def closing_scenario(duration_s: float = 0.80, step_s: float = 2.0e-5) -> ScenarioConfig:
    """Free closing of the finger with no object present."""
    return ScenarioConfig(
        name="free closing",
        params=build_plant(),
        duration_s=duration_s,
        step_s=step_s,
        sample_stride=10,
    )


def grasp_scenario(
    name: str,
    obstacle: ContactGeometry,
    duration_s: float = 1.70,
    step_s: float = 5.0e-5,
) -> ScenarioConfig:
    """Closing onto one object with the compliant contact law."""
    return ScenarioConfig(
        name=name,
        params=build_plant(obstacle=obstacle),
        duration_s=duration_s,
        step_s=step_s,
        sample_stride=10,
    )


def stall_scenario(duration_s: float = 1.70, step_s: float = 5.0e-5) -> ScenarioConfig:
    """Grasp of a rigid 60 mm cylinder, used for the stall force measurement.

    A rigid object is used rather than a compliant one so that the drive genuinely stalls
    and the measured force is set by the transmission rather than by how far the object
    deforms.
    """
    return ScenarioConfig(
        name="rigid grasp",
        params=build_plant(obstacle=LARGE_CYLINDER, contact=STIFF_CONTACT),
        duration_s=duration_s,
        step_s=step_s,
        sample_stride=10,
    )
