"""Structural interfaces for the controllers and the integrator.

Both are expressed as :class:`typing.Protocol` so that the pipeline depends on the shape
of a controller and of an integrator rather than on a particular class. Adding a new
controller requires no change anywhere else, and a test can substitute an adversarial
command source without inheriting from anything.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

__all__ = ["Controller", "Derivative", "Integrator", "Measurement"]

Derivative = Callable[[list[float], float], list[float]]
"""A right hand side: ``(state, time) -> d(state)/dt``."""


class Measurement(Protocol):
    """What a controller is allowed to see.

    Restricting the controller to a small measurement record rather than the whole state
    keeps the boundary between plant and controller honest: a real drive measures current,
    motor angle and speed from its own sensors, and a grasp force from a fingertip sensor.
    """

    @property
    def time_s(self) -> float:
        """Current time, in s."""

    @property
    def current_a(self) -> float:
        """Measured armature current, in A."""

    @property
    def motor_angle_rad(self) -> float:
        """Measured motor shaft angle, in rad."""

    @property
    def motor_speed_rad_s(self) -> float:
        """Measured motor shaft speed, in rad/s."""

    @property
    def grasp_force_n(self) -> float:
        """Measured contact force at the fingertip, in N."""


class Controller(Protocol):
    """A discrete time controller producing a bridge duty ratio."""

    @property
    def sample_period_s(self) -> float:
        """Interval between control updates, in s."""

    def reset(self) -> None:
        """Return the controller to its initial internal state."""

    def update(self, measurement: Measurement) -> float:
        """Return the duty ratio to hold until the next update."""


class Integrator(Protocol):
    """A fixed step scheme advancing a state vector."""

    @property
    def order(self) -> int:
        """Order of accuracy of the global error."""

    @property
    def stages(self) -> int:
        """Number of right hand side evaluations per step."""

    def step(
        self, derivative: Derivative, state: list[float], time_s: float, step_s: float
    ) -> list[float]:
        """Advance ``state`` by one step of ``step_s`` and return the new state."""
