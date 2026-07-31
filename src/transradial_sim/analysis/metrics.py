"""Performance metrics extracted from a trace, each with an explicit definition.

Every metric here states the definition it uses. A closing time is meaningless without
saying what fraction of travel counts as closed, and a grasp force is meaningless without
saying which bodies it is summed over, so both are named in the function that computes
them and reported alongside the value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from transradial_sim.model.contact import evaluate_contact
from transradial_sim.model.finger import joint_origins
from transradial_sim.model.motor import MotorParameters, predicted_no_load_speed
from transradial_sim.model.system import IDX_JOINTS, SystemParameters
from transradial_sim.model.units import rad_s_to_rpm
from transradial_sim.pipeline.trace import SimulationTrace

__all__ = [
    "GraspSummary",
    "PerformanceSummary",
    "closing_time_s",
    "free_running_speeds",
    "grasp_summary",
    "performance_summary",
    "settling_index",
]


def closing_time_s(trace: SimulationTrace, fraction: float = 0.95) -> float:
    """Return the time to reach ``fraction`` of the total flexion the run achieves.

    Defined against the run's own final posture rather than against the joint limits, so
    the metric is meaningful whether or not the finger reaches its stops. The convention
    of 95 percent matches the way settling times are quoted for a step response.

    Raises:
        ValueError: If ``fraction`` is outside ``(0, 1]``.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    flexion = trace.total_flexion_rad
    target = fraction * float(flexion[-1])
    reached = np.flatnonzero(flexion >= target)
    if reached.size == 0:
        return float(trace.time_s[-1])
    return float(trace.time_s[int(reached[0])])


def settling_index(trace: SimulationTrace, speed_fraction: float = 0.01) -> int:
    """Return the last sample index, used as the quasi static operating point.

    The drive is taken to have settled once its speed has fallen below
    ``speed_fraction`` of the peak reached during the run. The returned index is the first
    such sample after the peak, or the last sample if the drive never settles.
    """
    speed = np.abs(trace.motor_speed_rad_s)
    peak_index = int(np.argmax(speed))
    threshold = speed_fraction * float(speed[peak_index])
    tail = speed[peak_index:]
    below = np.flatnonzero(tail <= threshold)
    if below.size == 0:
        return trace.sample_count - 1
    return peak_index + int(below[0])


def free_running_speeds(
    motor: MotorParameters, supply_voltage_v: float
) -> tuple[float, float]:
    """Return the predicted no load speed in rad/s and in rpm at a given supply voltage."""
    speed = predicted_no_load_speed(motor, supply_voltage_v)
    return speed, rad_s_to_rpm(speed)


@dataclass(frozen=True, slots=True)
class GraspSummary:
    """The settled state of one grasp."""

    name: str
    joint_angles_deg: tuple[float, ...]
    phalanx_forces_n: tuple[float, ...]
    total_force_n: float
    fingertip_force_n: float
    contact_points: int
    drive_tension_n: float
    finger_tension_n: float
    measured_capstan_ratio: float
    battery_energy_j: float

    @property
    def force_spread(self) -> float:
        """Largest difference between any two phalanx forces, in N.

        A finger that cannot conform loads one phalanx and leaves the others idle, so a
        large spread on a round object and a small one on a flat object is the signature
        being measured.
        """
        if not self.phalanx_forces_n:
            return 0.0
        return max(self.phalanx_forces_n) - min(self.phalanx_forces_n)


def grasp_summary(trace: SimulationTrace, index: int | None = None) -> GraspSummary:
    """Return the settled grasp state of a run.

    Args:
        trace: Recorded run.
        index: Sample to report. The last sample by default.

    Returns:
        The summary, with contact forces recomputed at zero joint rate so that the
        reported force is the static grasp force and not a transient impact force.
    """
    position = trace.sample_count - 1 if index is None else index
    params = trace.params
    joint_count = params.finger.joint_count
    angles = tuple(float(a) for a in trace.joint_angles_rad[position])
    origins, directions, _ = joint_origins(params.finger, angles)
    if params.obstacle is None:
        forces: tuple[float, ...] = (0.0,) * joint_count
        total = 0.0
        points = 0
    else:
        result = evaluate_contact(
            params.finger,
            params.contact,
            params.obstacle,
            origins,
            directions,
            (0.0,) * joint_count,
        )
        forces = result.phalanx_forces_n
        total = result.total_normal_force_n
        points = result.contact_count
    drive = float(trace.drive_tension_n[position])
    finger = float(trace.finger_tension_n[position])
    return GraspSummary(
        name=trace.name,
        joint_angles_deg=tuple(float(np.degrees(a)) for a in angles),
        phalanx_forces_n=forces,
        total_force_n=total,
        fingertip_force_n=forces[-1] if forces else 0.0,
        contact_points=points,
        drive_tension_n=drive,
        finger_tension_n=finger,
        measured_capstan_ratio=finger / drive if drive > 0.0 else 1.0,
        battery_energy_j=float(trace.accumulator("battery_energy_j")[position]),
    )


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    """The headline performance numbers of one configuration."""

    closing_time_s: float
    peak_motor_speed_rad_s: float
    peak_motor_speed_rpm: float
    peak_tendon_speed_m_per_s: float
    no_load_speed_rpm: float
    stall_drive_tension_n: float
    stall_finger_tension_n: float
    stall_grasp_force_n: float
    stall_fingertip_force_n: float


def performance_summary(
    closing: SimulationTrace,
    stall: SimulationTrace,
) -> PerformanceSummary:
    """Combine a free closing run and a rigid grasp run into the headline metrics.

    Args:
        closing: Free closing run, which sets the speed metrics.
        stall: Rigid object grasp run, which sets the force metrics.

    Returns:
        The summary.
    """
    params: SystemParameters = closing.params
    peak_speed = float(np.max(np.abs(closing.motor_speed_rad_s)))
    tendon_speed = peak_speed * params.tendon.drive_radius_m / params.gearbox.ratio
    _, no_load_rpm = free_running_speeds(
        params.motor, params.supply.open_circuit_voltage_v
    )
    settled = grasp_summary(stall)
    return PerformanceSummary(
        closing_time_s=closing_time_s(closing),
        peak_motor_speed_rad_s=peak_speed,
        peak_motor_speed_rpm=rad_s_to_rpm(peak_speed),
        peak_tendon_speed_m_per_s=tendon_speed,
        no_load_speed_rpm=no_load_rpm,
        stall_drive_tension_n=settled.drive_tension_n,
        stall_finger_tension_n=settled.finger_tension_n,
        stall_grasp_force_n=settled.total_force_n,
        stall_fingertip_force_n=settled.fingertip_force_n,
    )


def joint_angles_at(trace: SimulationTrace, index: int) -> tuple[float, ...]:
    """Return the joint angles at one sample, in rad."""
    return tuple(float(a) for a in trace.joint_angles_rad[index])


def final_joint_angles(state: list[float], joint_count: int) -> tuple[float, ...]:
    """Return the joint angles held in a raw state vector, in rad."""
    return tuple(state[IDX_JOINTS : IDX_JOINTS + joint_count])
