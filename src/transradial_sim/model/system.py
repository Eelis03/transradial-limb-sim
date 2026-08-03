"""The complete plant: battery, driver, motor, gearbox, tendon, finger and object.

The state vector carries the physical states first and a set of energy accumulators
after them. Integrating the accumulators with the same scheme as the physical states is
what allows the energy balance to be checked at the end of a run: every power term in the
model is written so that it is individually signed correctly, and the identity

    ``P_battery = sum(P_loss) + d(E_stored)/dt + P_object``

holds exactly at every instant, not only on average. A residual in the integrated balance
is therefore a measure of integration error alone, which makes it a usable convergence
diagnostic as well as a correctness check.

Layout of the state vector:

======  =========================================
Index   Quantity
======  =========================================
0       armature current, A
1       motor shaft angle, rad
2       motor shaft angular velocity, rad/s
3..n+2  joint angles, rad
n+3..   joint angular velocities, rad/s
then    twelve energy accumulators, J
======  =========================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from transradial_sim.model.contact import (
    ContactGeometry,
    ContactModel,
    ContactResult,
    evaluate_contact,
    no_contact,
)
from transradial_sim.model.finger import (
    FingerGeometry,
    Vector2,
    bias_torques,
    joint_limit_torques,
    joint_origins,
    kinetic_energy,
    limit_spring_torques,
    mass_matrix,
    point_on_phalanx,
    solve_accelerations,
    spring_energy,
    spring_torques,
    tendon_displacement,
)
from transradial_sim.model.gearbox import (
    GearboxParameters,
    loss_torque_on_motor,
    lost_motion_m,
    reflected_load_torque,
)
from transradial_sim.model.motor import MotorParameters, electromagnetic_torque, friction_torque
from transradial_sim.model.tendon import TendonParameters, TendonState, evaluate_tendon

__all__ = [
    "ACCUMULATOR_NAMES",
    "IDX_CURRENT",
    "IDX_JOINTS",
    "IDX_MOTOR_ANGLE",
    "IDX_MOTOR_SPEED",
    "INTERNAL_LOSS_NAMES",
    "PlantEvaluation",
    "PowerSupply",
    "StoredEnergy",
    "SystemParameters",
    "energy_residual",
    "evaluate_plant",
    "initial_state",
    "object_energy_residual",
    "state_derivative",
    "stored_energy",
]

IDX_CURRENT: Final[int] = 0
IDX_MOTOR_ANGLE: Final[int] = 1
IDX_MOTOR_SPEED: Final[int] = 2
IDX_JOINTS: Final[int] = 3

INTERNAL_LOSS_NAMES: Final[tuple[str, ...]] = (
    "battery_internal_loss_j",
    "driver_loss_j",
    "motor_copper_loss_j",
    "motor_friction_loss_j",
    "gearbox_loss_j",
    "capstan_loss_j",
    "tendon_damper_loss_j",
    "joint_damping_loss_j",
    "joint_limit_loss_j",
)
"""Energy dissipated inside the prosthesis, one entry per physical mechanism."""

ACCUMULATOR_NAMES: Final[tuple[str, ...]] = (
    "battery_energy_j",
    *INTERNAL_LOSS_NAMES,
    "object_work_j",
    "object_dissipation_j",
)
"""Names of the twelve integrated energy channels, in state vector order.

The balance the simulation satisfies is

    ``battery_energy = sum(internal losses) + object_work + change in stored energy``

with the stored energy taken over the elements inside the prosthesis. ``object_work`` is
the useful output, the energy handed to the grasped object, and it splits again into the
elastic energy the object holds and ``object_dissipation``, the part the object turns into
heat. ``object_dissipation`` is therefore a breakdown of ``object_work`` and must not be
added to the internal losses; doing so double counts it, and double counts the object's
stored energy along with it.
"""


@dataclass(frozen=True, slots=True)
class PowerSupply:
    """Battery and motor driver, the two stages ahead of the motor terminals."""

    open_circuit_voltage_v: float
    """Battery open circuit voltage in V."""

    internal_resistance_ohm: float
    """Battery internal resistance in ohm."""

    driver_resistance_ohm: float
    """Total conduction resistance of the bridge in the motor current path, in ohm."""

    driver_quiescent_power_w: float
    """Power drawn by the driver logic and gate drive regardless of load, in W."""

    max_duty: float
    """Largest usable magnitude of the bridge duty ratio."""


@dataclass(frozen=True, slots=True)
class SystemParameters:
    """Every parameter needed to evaluate the plant."""

    supply: PowerSupply
    motor: MotorParameters
    gearbox: GearboxParameters
    tendon: TendonParameters
    finger: FingerGeometry
    contact: ContactModel
    current_limit_a: float
    """Trip level of the drive current limiter, in A."""

    obstacle: ContactGeometry | None = None
    """Object being grasped, or ``None`` for free motion."""

    gravity_m_per_s2: Vector2 = (0.0, 0.0)
    """Gravity vector in the finger plane, in m/s^2. Zero by default."""

    @property
    def state_size(self) -> int:
        """Length of the state vector."""
        return IDX_JOINTS + 2 * self.finger.joint_count + len(ACCUMULATOR_NAMES)

    @property
    def accumulator_offset(self) -> int:
        """Index of the first energy accumulator in the state vector."""
        return IDX_JOINTS + 2 * self.finger.joint_count

    @property
    def rotor_inertia_kgm2(self) -> float:
        """Rotor plus gearbox inertia referred to the motor shaft, in kg m^2."""
        return self.motor.rotor_inertia_kgm2 + self.gearbox.inertia_kgm2

    @property
    def lost_motion_m(self) -> float:
        """Cord travel the gearbox play absorbs before the tendon loads, in m.

        The drive pulley is mounted on the gearbox output, so the angular play of the
        gearhead appears at the cord as a dead band of this width.
        """
        return lost_motion_m(self.gearbox, self.tendon.drive_radius_m)


def initial_state(
    params: SystemParameters, angles_rad: tuple[float, ...] | None = None
) -> list[float]:
    """Return a state vector with the finger extended, at rest, and no stored energy."""
    joints = angles_rad if angles_rad is not None else (0.0,) * params.finger.joint_count
    if len(joints) != params.finger.joint_count:
        raise ValueError("angles_rad must have one entry per joint")
    state = [0.0] * params.state_size
    for index, angle in enumerate(joints):
        state[IDX_JOINTS + index] = angle
    # The drive pulley starts at the position that leaves the cord exactly taut. With a
    # gearhead that has play the tendon is still disengaged there, because the play is
    # free whenever the cord carries no tension, so the run begins by taking it up.
    state[IDX_MOTOR_ANGLE] = (
        tendon_displacement(params.finger, joints)
        * params.gearbox.ratio
        / params.tendon.drive_radius_m
    )
    return state


@dataclass(frozen=True, slots=True)
class StoredEnergy:
    """Energy currently held in the system, by storage element."""

    inductor_j: float
    rotor_kinetic_j: float
    finger_kinetic_j: float
    tendon_elastic_j: float
    return_spring_j: float
    joint_limit_j: float
    contact_elastic_j: float
    gravitational_j: float

    @property
    def internal_total_j(self) -> float:
        """Energy stored inside the prosthesis, in J.

        Excludes :attr:`contact_elastic_j`, which is held by the grasped object and is
        already accounted for inside ``object_work_j``.
        """
        return (
            self.inductor_j
            + self.rotor_kinetic_j
            + self.finger_kinetic_j
            + self.tendon_elastic_j
            + self.return_spring_j
            + self.joint_limit_j
            + self.gravitational_j
        )

    @property
    def total_j(self) -> float:
        """Sum of every storage term including the object, in J."""
        return self.internal_total_j + self.contact_elastic_j


@dataclass(frozen=True, slots=True)
class PlantEvaluation:
    """Derivative of the state together with the diagnostics used for reporting."""

    derivative: list[float]
    applied_voltage_v: float
    duty: float
    battery_current_a: float
    battery_power_w: float
    motor_torque_nm: float
    output_torque_nm: float
    tendon: TendonState
    contact: ContactResult
    joint_accelerations_rad_s2: list[float]


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def stored_energy(params: SystemParameters, state: list[float]) -> StoredEnergy:
    """Return the energy held in every storage element for a given state."""
    joint_count = params.finger.joint_count
    angles = tuple(state[IDX_JOINTS : IDX_JOINTS + joint_count])
    rates = tuple(state[IDX_JOINTS + joint_count : IDX_JOINTS + 2 * joint_count])
    origins, directions, _ = joint_origins(params.finger, angles)

    matrix = mass_matrix(params.finger, origins, directions)
    _, limit_stored = joint_limit_torques(params.finger, angles, rates)

    drive_displacement = (
        state[IDX_MOTOR_ANGLE] * params.tendon.drive_radius_m / params.gearbox.ratio
    )
    finger_displacement = tendon_displacement(params.finger, angles)
    elastic = drive_displacement - finger_displacement - params.lost_motion_m
    tendon_elastic = (
        0.5 * params.tendon.stiffness_n_per_m * elastic * elastic if elastic > 0.0 else 0.0
    )

    contact_elastic = 0.0
    if params.obstacle is not None:
        contact_result = evaluate_contact(
            params.finger,
            params.contact,
            params.obstacle,
            origins,
            directions,
            (0.0,) * joint_count,
        )
        contact_elastic = contact_result.stored_energy_j

    gravitational = 0.0
    gravity = params.gravity_m_per_s2
    if gravity != (0.0, 0.0):
        for index, phalanx in enumerate(params.finger.phalanges):
            com = point_on_phalanx(origins, directions, index, phalanx.com_distance_m)
            gravitational -= phalanx.mass_kg * (gravity[0] * com[0] + gravity[1] * com[1])

    current = state[IDX_CURRENT]
    speed = state[IDX_MOTOR_SPEED]
    return StoredEnergy(
        inductor_j=0.5 * params.motor.inductance_h * current * current,
        rotor_kinetic_j=0.5 * params.rotor_inertia_kgm2 * speed * speed,
        finger_kinetic_j=kinetic_energy(matrix, rates),
        tendon_elastic_j=tendon_elastic,
        return_spring_j=spring_energy(params.finger, angles),
        joint_limit_j=limit_stored,
        contact_elastic_j=contact_elastic,
        gravitational_j=gravitational,
    )


def evaluate_plant(
    params: SystemParameters,
    state: list[float],
    duty_command: float,
) -> PlantEvaluation:
    """Evaluate the plant derivative and diagnostics for one state and one command.

    Args:
        params: Plant parameters.
        state: State vector of length ``params.state_size``.
        duty_command: Requested bridge duty ratio in ``[-1, 1]``.

    Returns:
        The evaluation, whose ``derivative`` field is the time derivative of ``state``.
    """
    finger = params.finger
    motor = params.motor
    gearbox = params.gearbox
    tendon = params.tendon
    supply = params.supply
    joint_count = finger.joint_count

    current = state[IDX_CURRENT]
    motor_speed = state[IDX_MOTOR_SPEED]
    angles = tuple(state[IDX_JOINTS : IDX_JOINTS + joint_count])
    rates = tuple(state[IDX_JOINTS + joint_count : IDX_JOINTS + 2 * joint_count])

    # --- Drive electronics, with a hardware style current limiter -------------------
    duty = _clamp(duty_command, supply.max_duty)
    series_resistance = motor.resistance_ohm + supply.driver_resistance_ohm
    back_emf = motor.back_emf_constant_v_s_per_rad * motor_speed
    open_circuit = supply.open_circuit_voltage_v
    limit = params.current_limit_a
    if current >= limit:
        hold_duty = (limit * series_resistance + back_emf) / open_circuit
        duty = min(duty, hold_duty)
    elif current <= -limit:
        hold_duty = (-limit * series_resistance + back_emf) / open_circuit
        duty = max(duty, hold_duty)

    battery_current = duty * current
    applied_voltage = (
        duty * open_circuit
        - duty * battery_current * supply.internal_resistance_ohm
        - current * supply.driver_resistance_ohm
    )
    battery_loss = battery_current * battery_current * supply.internal_resistance_ohm
    driver_loss = (
        current * current * supply.driver_resistance_ohm + supply.driver_quiescent_power_w
    )
    battery_power = open_circuit * battery_current + supply.driver_quiescent_power_w

    net_voltage = applied_voltage - motor.resistance_ohm * current - back_emf
    current_rate = net_voltage / motor.inductance_h
    copper_loss = current * current * motor.resistance_ohm

    # --- Kinematics shared by the dynamics and the contact model ---------------------
    origins, directions, _ = joint_origins(finger, angles)

    # --- Tendon ----------------------------------------------------------------------
    radius = tendon.drive_radius_m
    ratio = gearbox.ratio
    drive_displacement = state[IDX_MOTOR_ANGLE] * radius / ratio
    drive_velocity = motor_speed * radius / ratio
    finger_displacement = tendon_displacement(finger, angles)
    finger_velocity = 0.0
    for index, phalanx in enumerate(finger.phalanges):
        finger_velocity += phalanx.moment_arm_m * rates[index]
    motor_torque = electromagnetic_torque(motor, current)
    motor_friction = friction_torque(motor, motor_speed)
    # Direction the drive is pushing the cord, which is the direction the cord is about to
    # slide once it has stopped. Evaluated before the tendon so that the routing friction
    # has a direction to use inside its stick band.
    shaft_torque = motor_torque - motor_friction
    impending = 0.0
    if shaft_torque > 0.0:
        impending = 1.0
    elif shaft_torque < 0.0:
        impending = -1.0
    tendon_state = evaluate_tendon(
        tendon,
        drive_displacement,
        drive_velocity,
        finger_displacement,
        finger_velocity,
        impending,
        params.lost_motion_m,
    )

    # --- Motor and gearbox -----------------------------------------------------------
    output_torque = radius * tendon_state.drive_tension_n
    load_torque = reflected_load_torque(gearbox, output_torque)
    driving_torque = shaft_torque - load_torque
    gear_loss_torque = loss_torque_on_motor(
        gearbox, output_torque, motor_speed, driving_torque
    )
    speed_rate = (driving_torque - gear_loss_torque) / params.rotor_inertia_kgm2

    motor_friction_loss = motor_friction * motor_speed
    gearbox_loss = gear_loss_torque * motor_speed

    # --- Finger ----------------------------------------------------------------------
    tension = tendon_state.finger_tension_n
    applied = [phalanx.moment_arm_m * tension for phalanx in finger.phalanges]

    springs = spring_torques(finger, angles)
    limits, _ = joint_limit_torques(finger, angles, rates)
    limit_springs = limit_spring_torques(finger, angles)

    joint_damping_loss = 0.0
    joint_limit_loss = 0.0
    for index, phalanx in enumerate(finger.phalanges):
        rate = rates[index]
        damping_torque = -phalanx.damping_nms_per_rad * rate
        applied[index] += springs[index] + damping_torque + limits[index]
        joint_damping_loss += phalanx.damping_nms_per_rad * rate * rate
        joint_limit_loss += (limit_springs[index] - limits[index]) * rate

    if params.obstacle is not None:
        contact_result = evaluate_contact(
            finger, params.contact, params.obstacle, origins, directions, rates
        )
        for index in range(joint_count):
            applied[index] += contact_result.joint_torques_nm[index]
    else:
        contact_result = no_contact(joint_count)

    matrix = mass_matrix(finger, origins, directions)
    bias = bias_torques(finger, angles, rates, params.gravity_m_per_s2)
    right_hand_side = [applied[index] - bias[index] for index in range(joint_count)]
    accelerations = solve_accelerations(matrix, right_hand_side)

    # --- Assemble the derivative ------------------------------------------------------
    derivative = [0.0] * params.state_size
    derivative[IDX_CURRENT] = current_rate
    derivative[IDX_MOTOR_ANGLE] = motor_speed
    derivative[IDX_MOTOR_SPEED] = speed_rate
    for index in range(joint_count):
        derivative[IDX_JOINTS + index] = rates[index]
        derivative[IDX_JOINTS + joint_count + index] = accelerations[index]

    offset = params.accumulator_offset
    derivative[offset + 0] = battery_power
    derivative[offset + 1] = battery_loss
    derivative[offset + 2] = driver_loss
    derivative[offset + 3] = copper_loss
    derivative[offset + 4] = motor_friction_loss
    derivative[offset + 5] = gearbox_loss
    derivative[offset + 6] = tendon_state.friction_power_w
    derivative[offset + 7] = tendon_state.damper_power_w
    derivative[offset + 8] = joint_damping_loss
    derivative[offset + 9] = joint_limit_loss
    derivative[offset + 10] = contact_result.power_into_object_w
    derivative[offset + 11] = contact_result.dissipated_power_w

    return PlantEvaluation(
        derivative=derivative,
        applied_voltage_v=applied_voltage,
        duty=duty,
        battery_current_a=battery_current,
        battery_power_w=battery_power,
        motor_torque_nm=motor_torque,
        output_torque_nm=output_torque,
        tendon=tendon_state,
        contact=contact_result,
        joint_accelerations_rad_s2=accelerations,
    )


def state_derivative(
    params: SystemParameters, state: list[float], duty_command: float
) -> list[float]:
    """Return only the time derivative of ``state``, for use inside an integrator."""
    return evaluate_plant(params, state, duty_command).derivative


def energy_residual(params: SystemParameters, initial: list[float], final: list[float]) -> float:
    """Return the closure error of the system energy balance between two states, in J.

    The residual is ``E_battery - sum(internal losses) - work on the object - change in
    internal stored energy``. Every term on the right is written so that it is signed
    correctly at every instant, so in exact arithmetic the residual is zero for any
    trajectory and its size measures the integration error alone.
    """
    offset = params.accumulator_offset
    supplied = final[offset] - initial[offset]
    channels = 1 + len(INTERNAL_LOSS_NAMES) + 1
    consumed = math.fsum(final[offset + k] - initial[offset + k] for k in range(1, channels))
    change = (
        stored_energy(params, final).internal_total_j
        - stored_energy(params, initial).internal_total_j
    )
    return supplied - consumed - change


def object_energy_residual(
    params: SystemParameters, initial: list[float], final: list[float]
) -> float:
    """Return the closure error of the object energy split, in J.

    The work delivered to the object must equal the elastic energy the object now holds
    plus the energy it has dissipated. Checking this separately from the system balance
    isolates the contact law from the rest of the model.
    """
    offset = params.accumulator_offset
    work_index = offset + 1 + len(INTERNAL_LOSS_NAMES)
    work = final[work_index] - initial[work_index]
    dissipated = final[work_index + 1] - initial[work_index + 1]
    stored = (
        stored_energy(params, final).contact_elastic_j
        - stored_energy(params, initial).contact_elastic_j
    )
    return work - dissipated - stored
