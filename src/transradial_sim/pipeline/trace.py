"""The structured record a simulation run produces.

Everything downstream of the pipeline, the metrics, the efficiency chain, the sensitivity
study and the figures, reads a :class:`SimulationTrace` and nothing else. Keeping the
trace as plain arrays with named fields means the analysis layer never needs to know how
the run was configured or which controller produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from transradial_sim.model.system import ACCUMULATOR_NAMES, StoredEnergy, SystemParameters

__all__ = ["SimulationTrace"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SimulationTrace:
    """Sampled history of one simulation run."""

    name: str
    """Identifier of the scenario that produced this trace."""

    params: SystemParameters
    """Plant parameters the run used."""

    step_s: float
    """Integration step, in s."""

    control_period_s: float
    """Controller sample period, in s."""

    time_s: FloatArray
    """Sample times, in s. Shape ``(n,)``."""

    current_a: FloatArray
    """Armature current, in A. Shape ``(n,)``."""

    motor_angle_rad: FloatArray
    """Motor shaft angle, in rad. Shape ``(n,)``."""

    motor_speed_rad_s: FloatArray
    """Motor shaft speed, in rad/s. Shape ``(n,)``."""

    joint_angles_rad: FloatArray
    """Joint angles, in rad. Shape ``(n, joints)``."""

    joint_rates_rad_s: FloatArray
    """Joint rates, in rad/s. Shape ``(n, joints)``."""

    duty: FloatArray
    """Bridge duty ratio actually applied after limiting. Shape ``(n,)``."""

    applied_voltage_v: FloatArray
    """Voltage at the motor terminals, in V. Shape ``(n,)``."""

    battery_current_a: FloatArray
    """Current drawn from the battery, in A. Shape ``(n,)``."""

    battery_power_w: FloatArray
    """Power drawn from the battery including driver quiescent draw, in W. Shape ``(n,)``."""

    motor_torque_nm: FloatArray
    """Electromagnetic torque at the motor shaft, in Nm. Shape ``(n,)``."""

    output_torque_nm: FloatArray
    """Torque at the gearbox output, in Nm. Shape ``(n,)``."""

    drive_tension_n: FloatArray
    """Tendon tension at the drive pulley, in N. Shape ``(n,)``."""

    finger_tension_n: FloatArray
    """Tendon tension at the finger, in N. Shape ``(n,)``."""

    tendon_extension_m: FloatArray
    """Stretch of the tendon series element, in m. Shape ``(n,)``."""

    contact_force_n: FloatArray
    """Sum of the contact normal force magnitudes, in N. Shape ``(n,)``."""

    fingertip_force_n: FloatArray
    """Normal force at the distal tip sample point, in N. Shape ``(n,)``."""

    contact_count: FloatArray
    """Number of sample points in contact. Shape ``(n,)``."""

    tip_position_m: FloatArray
    """Fingertip position in the finger plane, in m. Shape ``(n, 2)``."""

    accumulators_j: FloatArray
    """Integrated energy channels, in J. Shape ``(n, 12)``, columns as
    :data:`~transradial_sim.model.system.ACCUMULATOR_NAMES`."""

    initial_stored: StoredEnergy
    """Stored energy at the first sample."""

    final_stored: StoredEnergy
    """Stored energy at the last sample."""

    final_state: tuple[float, ...]
    """Raw state vector at the end of the run."""

    def accumulator(self, name: str) -> FloatArray:
        """Return one integrated energy channel by name.

        Raises:
            KeyError: If ``name`` is not one of
                :data:`~transradial_sim.model.system.ACCUMULATOR_NAMES`.
        """
        if name not in ACCUMULATOR_NAMES:
            raise KeyError(f"unknown accumulator: {name}")
        column = ACCUMULATOR_NAMES.index(name)
        return np.asarray(self.accumulators_j[:, column], dtype=np.float64)

    def final_accumulator(self, name: str) -> float:
        """Return the final value of one integrated energy channel, in J."""
        return float(self.accumulator(name)[-1])

    @property
    def sample_count(self) -> int:
        """Number of samples in the trace."""
        return int(self.time_s.shape[0])

    @property
    def duration_s(self) -> float:
        """Time spanned by the trace, in s."""
        return float(self.time_s[-1] - self.time_s[0])

    @property
    def total_flexion_rad(self) -> FloatArray:
        """Sum of the joint angles at each sample, in rad."""
        return np.asarray(self.joint_angles_rad.sum(axis=1), dtype=np.float64)
