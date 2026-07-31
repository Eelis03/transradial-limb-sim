"""Simulation driver: advance the plant under a controller and record a trace.

The loop is deliberately explicit about the two rates it involves. The controller runs on
its own sample period with a zero order hold on the duty ratio, and the plant is advanced
by an integer number of integration steps inside each control period. Nothing else in the
project chooses a time step, so the convergence study has one knob to turn.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from transradial_sim.algorithm.controllers import PlantMeasurement
from transradial_sim.algorithm.integrators import RungeKutta4
from transradial_sim.model.contact import evaluate_contact
from transradial_sim.model.finger import joint_origins
from transradial_sim.model.system import (
    ACCUMULATOR_NAMES,
    IDX_CURRENT,
    IDX_JOINTS,
    IDX_MOTOR_ANGLE,
    IDX_MOTOR_SPEED,
    SystemParameters,
    evaluate_plant,
    initial_state,
    stored_energy,
)
from transradial_sim.pipeline.trace import SimulationTrace

__all__ = ["ScenarioConfig", "SimulationController", "fingertip_force", "run_scenario"]


class SimulationController(Protocol):
    """The subset of the controller interface the driver uses."""

    @property
    def sample_period_s(self) -> float:
        """Interval between control updates, in s."""

    def reset(self) -> None:
        """Return the controller to its initial internal state."""

    def update(self, measurement: PlantMeasurement) -> float:
        """Return the duty ratio to hold until the next update."""


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """Everything that defines one run apart from the controller."""

    name: str
    params: SystemParameters
    duration_s: float
    step_s: float = 2.0e-5
    sample_stride: int = 5
    """Number of control periods between recorded samples."""

    initial_angles_rad: tuple[float, ...] | None = None
    """Starting joint angles, in rad. Full extension by default."""

    initial_motor_speed_rad_s: float = 0.0
    """Starting motor shaft speed, in rad/s.

    Nonzero values are used by the conservation tests, which need the plant to start with
    energy already in it and no source connected.
    """

    def with_step(self, step_s: float) -> ScenarioConfig:
        """Return a copy of this configuration with a different integration step."""
        return replace(self, step_s=step_s)

    def with_params(self, params: SystemParameters) -> ScenarioConfig:
        """Return a copy of this configuration with different plant parameters."""
        return replace(self, params=params)


def fingertip_force(params: SystemParameters, state: list[float]) -> float:
    """Return the fingertip contact force for a state, in N.

    This is the quantity a fingertip force sensor would report, and it is evaluated
    separately from the plant derivative so that the controller sees a measurement rather
    than reaching into the plant.
    """
    if params.obstacle is None:
        return 0.0
    joint_count = params.finger.joint_count
    angles = tuple(state[IDX_JOINTS : IDX_JOINTS + joint_count])
    rates = tuple(state[IDX_JOINTS + joint_count : IDX_JOINTS + 2 * joint_count])
    origins, directions, _ = joint_origins(params.finger, angles)
    result = evaluate_contact(
        params.finger, params.contact, params.obstacle, origins, directions, rates
    )
    return result.fingertip_force_n


def run_scenario(
    config: ScenarioConfig,
    controller: SimulationController,
    integrator: RungeKutta4 | None = None,
) -> SimulationTrace:
    """Run one scenario and return its trace.

    Args:
        config: Scenario definition.
        controller: Controller, reset before the run begins.
        integrator: Fixed step scheme, fourth order Runge-Kutta by default.

    Returns:
        The recorded trace.

    Raises:
        ValueError: If the control period is not a positive multiple of the step.
    """
    scheme = integrator if integrator is not None else RungeKutta4()
    params = config.params
    control_period = controller.sample_period_s
    steps_per_control = round(control_period / config.step_s)
    if steps_per_control < 1:
        raise ValueError("control period must be at least one integration step")
    if abs(steps_per_control * config.step_s - control_period) > 1.0e-12 * control_period:
        raise ValueError("control period must be an integer multiple of the integration step")

    control_count = round(config.duration_s / control_period)
    controller.reset()
    state = initial_state(params, config.initial_angles_rad)
    state[IDX_MOTOR_SPEED] = config.initial_motor_speed_rad_s
    joint_count = params.finger.joint_count

    times: list[float] = []
    rows: list[list[float]] = []
    joint_angles: list[list[float]] = []
    joint_rates: list[list[float]] = []
    tips: list[list[float]] = []
    accumulators: list[list[float]] = []

    offset = params.accumulator_offset
    channels = len(ACCUMULATOR_NAMES)
    first_stored = stored_energy(params, state)
    time_s = 0.0
    held_duty = [0.0]

    def derivative(vector: list[float], moment: float) -> list[float]:
        del moment
        return evaluate_plant(params, vector, held_duty[0]).derivative

    for index in range(control_count + 1):
        measurement = PlantMeasurement(
            time_s=time_s,
            current_a=state[IDX_CURRENT],
            motor_angle_rad=state[IDX_MOTOR_ANGLE],
            motor_speed_rad_s=state[IDX_MOTOR_SPEED],
            grasp_force_n=fingertip_force(params, state),
        )
        held_duty[0] = controller.update(measurement)

        if index % config.sample_stride == 0 or index == control_count:
            evaluation = evaluate_plant(params, state, held_duty[0])
            angles = tuple(state[IDX_JOINTS : IDX_JOINTS + joint_count])
            origins, _, _ = joint_origins(params.finger, angles)
            tip = origins[-1]
            times.append(time_s)
            rows.append(
                [
                    state[IDX_CURRENT],
                    state[IDX_MOTOR_ANGLE],
                    state[IDX_MOTOR_SPEED],
                    evaluation.duty,
                    evaluation.applied_voltage_v,
                    evaluation.battery_current_a,
                    evaluation.battery_power_w,
                    evaluation.motor_torque_nm,
                    evaluation.output_torque_nm,
                    evaluation.tendon.drive_tension_n,
                    evaluation.tendon.finger_tension_n,
                    evaluation.tendon.extension_m,
                    evaluation.contact.total_normal_force_n,
                    evaluation.contact.fingertip_force_n,
                    float(evaluation.contact.contact_count),
                ]
            )
            joint_angles.append(list(angles))
            joint_rates.append(list(state[IDX_JOINTS + joint_count : IDX_JOINTS + 2 * joint_count]))
            tips.append([tip[0], tip[1]])
            accumulators.append(list(state[offset : offset + channels]))

        if index == control_count:
            break

        for _ in range(steps_per_control):
            state = scheme.step(derivative, state, time_s, config.step_s)
            time_s += config.step_s

    table = np.asarray(rows, dtype=np.float64)
    return SimulationTrace(
        name=config.name,
        params=params,
        step_s=config.step_s,
        control_period_s=control_period,
        time_s=np.asarray(times, dtype=np.float64),
        current_a=table[:, 0],
        motor_angle_rad=table[:, 1],
        motor_speed_rad_s=table[:, 2],
        joint_angles_rad=np.asarray(joint_angles, dtype=np.float64),
        joint_rates_rad_s=np.asarray(joint_rates, dtype=np.float64),
        duty=table[:, 3],
        applied_voltage_v=table[:, 4],
        battery_current_a=table[:, 5],
        battery_power_w=table[:, 6],
        motor_torque_nm=table[:, 7],
        output_torque_nm=table[:, 8],
        drive_tension_n=table[:, 9],
        finger_tension_n=table[:, 10],
        tendon_extension_m=table[:, 11],
        contact_force_n=table[:, 12],
        fingertip_force_n=table[:, 13],
        contact_count=table[:, 14],
        tip_position_m=np.asarray(tips, dtype=np.float64),
        accumulators_j=np.asarray(accumulators, dtype=np.float64),
        initial_stored=first_stored,
        final_stored=stored_energy(params, state),
        final_state=tuple(state),
    )
