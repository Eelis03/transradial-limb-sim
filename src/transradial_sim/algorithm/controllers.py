"""Controllers, all running at a fixed rate with a zero order hold on the duty ratio.

The cascade is the one a real prosthetic drive uses. An inner current regulator makes the
motor behave as a torque source over its own bandwidth, and an outer position or force
regulator commands a current. The inner loop is tuned by pole placement against the
armature, ``Kp = L * omega_c`` and ``Ki = R * omega_c``, so the zero of the regulator
cancels the electrical pole and the closed current loop becomes first order with corner
frequency ``omega_c``. That single number is the actuator bandwidth every outer loop then
has to live inside, which is the point of putting the motor model in the loop at all.

Two features present in real drives and important for the results are modelled: the
current reference is slew limited, which is what protects a small gearbox from shock
loading, and every integrator has clamping anti windup, without which the outer loop winds
up during the several hundred milliseconds the finger spends closing against nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from transradial_sim.model.motor import MotorParameters

__all__ = [
    "ConstantDuty",
    "CurrentController",
    "ForceController",
    "PlantMeasurement",
    "PositionController",
    "ScheduledCurrentController",
    "current_loop_gains",
]


@dataclass(frozen=True, slots=True)
class PlantMeasurement:
    """The quantities a drive can actually measure."""

    time_s: float
    current_a: float
    motor_angle_rad: float
    motor_speed_rad_s: float
    grasp_force_n: float


def current_loop_gains(motor: MotorParameters, bandwidth_hz: float) -> tuple[float, float]:
    """Return the proportional and integral gains of the current regulator.

    Pole placement on the armature: choosing ``Kp = L * omega_c`` and ``Ki = R * omega_c``
    puts the regulator zero exactly on the electrical pole ``-R / L``, leaving a first
    order closed loop with corner frequency ``omega_c``. The gains therefore follow from
    the motor parameters and one design choice, the bandwidth, rather than from tuning.

    Args:
        motor: Motor parameters.
        bandwidth_hz: Target closed loop current bandwidth in Hz.

    Returns:
        ``(proportional_v_per_a, integral_v_per_a_s)``.
    """
    if bandwidth_hz <= 0.0:
        raise ValueError("bandwidth_hz must be strictly positive")
    omega = 2.0 * math.pi * bandwidth_hz
    return motor.inductance_h * omega, motor.resistance_ohm * omega


@dataclass(slots=True)
class ConstantDuty:
    """Applies a fixed duty ratio, used for open loop and adversarial tests."""

    duty: float
    sample_period_s: float = 1.0e-4

    def reset(self) -> None:
        """No internal state to reset."""

    def update(self, measurement: PlantMeasurement) -> float:
        """Return the fixed duty ratio."""
        del measurement
        return self.duty


@dataclass(slots=True)
class CurrentController:
    """Inner current regulator with a slew limited, hard clamped reference."""

    motor: MotorParameters
    supply_voltage_v: float
    current_limit_a: float
    bandwidth_hz: float = 500.0
    slew_limit_a_per_s: float = 40.0
    sample_period_s: float = 1.0e-4
    reference_a: float = 0.0
    speed_limit_rad_s: float | None = None
    """Motor speed the drive will not exceed, in rad/s, or ``None`` for no limit.

    A prosthetic hand does not close at the free running speed of its motor. If it did,
    the finger would arrive at the object carrying enough kinetic energy to stretch the
    tendon well past any tension the motor can hold, and because a tendon cannot push, that
    stretch never comes back out: the grasp force would be set by the impact rather than by
    the commanded current. The limiter rolls the current reference off over the top fifth
    of the speed range, which is how a drive with a speed loop behaves.
    """
    _integral_v: float = field(default=0.0, init=False)
    _reference_a: float = field(default=0.0, init=False)

    def reset(self) -> None:
        """Clear the regulator integrator and the slew limited reference."""
        self._integral_v = 0.0
        self._reference_a = 0.0

    def set_reference(self, current_a: float) -> None:
        """Set the commanded current, before clamping and slew limiting."""
        self.reference_a = current_a

    @property
    def applied_reference_a(self) -> float:
        """The reference actually in force after clamping and slew limiting, in A."""
        return self._reference_a

    def update(self, measurement: PlantMeasurement) -> float:
        """Return the duty ratio for the next sample period."""
        target = max(-self.current_limit_a, min(self.current_limit_a, self.reference_a))
        if self.speed_limit_rad_s is not None:
            speed = measurement.motor_speed_rad_s
            if target * speed > 0.0:
                headroom = self.speed_limit_rad_s - abs(speed)
                allowance = headroom / (0.2 * self.speed_limit_rad_s)
                target *= max(0.0, min(1.0, allowance))
        maximum_change = self.slew_limit_a_per_s * self.sample_period_s
        delta = max(-maximum_change, min(maximum_change, target - self._reference_a))
        self._reference_a += delta

        proportional, integral = current_loop_gains(self.motor, self.bandwidth_hz)
        error = self._reference_a - measurement.current_a
        feedforward = self.motor.back_emf_constant_v_s_per_rad * measurement.motor_speed_rad_s
        candidate = feedforward + proportional * error + self._integral_v
        duty = candidate / self.supply_voltage_v
        if -1.0 <= duty <= 1.0:
            # Clamping anti windup: the integrator only accumulates while the bridge has
            # authority left, so it cannot build up a command the hardware cannot deliver.
            self._integral_v += integral * error * self.sample_period_s
            return duty
        return 1.0 if duty > 1.0 else -1.0


@dataclass(slots=True)
class ScheduledCurrentController:
    """Applies a piecewise constant current reference on a schedule.

    A prosthetic hand does not close at full torque. It closes at a moderate current so
    that the fingers reach the object gently, and only then raises the current to squeeze.
    Modelling that two phase profile matters for the reported grasp force: closing at full
    current makes the finger arrive with enough kinetic energy that the impact stretches
    the tendon well past its quasi static equilibrium, and the tendon cannot push the
    stretch back out again.
    """

    inner: CurrentController
    schedule: tuple[tuple[float, float], ...]
    """Pairs of ``(start time in s, reference current in A)``, in increasing time order."""

    @property
    def sample_period_s(self) -> float:
        """Interval between control updates, in s."""
        return self.inner.sample_period_s

    def reset(self) -> None:
        """Clear the inner loop."""
        self.inner.reset()

    def update(self, measurement: PlantMeasurement) -> float:
        """Return the duty ratio for the next sample period."""
        reference = 0.0
        for start_s, current_a in self.schedule:
            if measurement.time_s >= start_s:
                reference = current_a
        self.inner.set_reference(reference)
        return self.inner.update(measurement)


@dataclass(slots=True)
class PositionController:
    """Outer proportional and integral loop on motor angle, cascaded onto the current loop.

    The motor angle is the natural controlled variable for a tendon drive because it is the
    only position the drive can measure. The joint angles follow it only through the tendon,
    which is exactly why the tendon compliance limits the achievable position bandwidth.
    """

    inner: CurrentController
    proportional_a_per_rad: float
    integral_a_per_rad_s: float
    derivative_a_s_per_rad: float = 0.0
    target_rad: float = 0.0
    _integral_a: float = field(default=0.0, init=False)

    @property
    def sample_period_s(self) -> float:
        """Interval between control updates, in s."""
        return self.inner.sample_period_s

    def reset(self) -> None:
        """Clear this loop and the inner loop."""
        self._integral_a = 0.0
        self.inner.reset()

    def update(self, measurement: PlantMeasurement) -> float:
        """Return the duty ratio for the next sample period."""
        error = self.target_rad - measurement.motor_angle_rad
        command = (
            self.proportional_a_per_rad * error
            + self._integral_a
            - self.derivative_a_s_per_rad * measurement.motor_speed_rad_s
        )
        limit = self.inner.current_limit_a
        if abs(command) < limit:
            self._integral_a += self.integral_a_per_rad_s * error * self.sample_period_s
        self.inner.set_reference(command)
        return self.inner.update(measurement)


@dataclass(slots=True)
class ForceController:
    """Outer proportional and integral loop on measured fingertip force.

    Force control against a compliant object is the case where actuator dynamics matter
    most: the loop gain that the tendon and the object allow is small, and the achievable
    bandwidth is bounded above by the current loop that sits underneath it.

    The integrator runs only while the fingertip is in contact. Out of contact there is
    no force to correct and no amount of current will create one until the finger
    arrives, so integrating during the approach only builds a command that has to be
    unwound afterwards. Conditional integration of this kind is standard in force
    controlled manipulation and it is what makes the approach transient bounded.
    """

    inner: CurrentController
    proportional_a_per_n: float
    integral_a_per_n_s: float
    target_n: float = 0.0
    feedforward_a: float = 0.0
    integral_limit_a: float = 0.40
    """Authority given to the integral term, in A.

    Bounding it separately from the drive current limit is what stops the approach phase,
    during which there is nothing to push against, from building a command that then has to
    be unwound after contact.
    """

    _integral_a: float = field(default=0.0, init=False)

    @property
    def sample_period_s(self) -> float:
        """Interval between control updates, in s."""
        return self.inner.sample_period_s

    def reset(self) -> None:
        """Clear this loop and the inner loop."""
        self._integral_a = 0.0
        self.inner.reset()

    def update(self, measurement: PlantMeasurement) -> float:
        """Return the duty ratio for the next sample period."""
        error = self.target_n - measurement.grasp_force_n
        command = self.feedforward_a + self.proportional_a_per_n * error + self._integral_a
        limit = self.integral_limit_a
        engaged = measurement.grasp_force_n > 0.0 or error > 0.0
        if engaged and abs(command) < self.inner.current_limit_a:
            self._integral_a += self.integral_a_per_n_s * error * self.sample_period_s
            self._integral_a = max(-limit, min(limit, self._integral_a))
        self.inner.set_reference(command)
        return self.inner.update(measurement)
