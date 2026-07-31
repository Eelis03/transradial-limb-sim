"""Fixed step explicit integrators.

A fixed step scheme is used rather than an adaptive one because the plant is driven by a
zero order hold controller running at a fixed rate. With an adaptive solver the control
instants would have to be imposed as event boundaries, and the reported control bandwidth
would depend on the solver tolerance rather than on the drive. A fixed step also makes
the convergence study a direct statement about the reported results: halving the step is
the only change made.

The classical fourth order Runge-Kutta scheme is the default. It is fourth order accurate
on smooth right hand sides, and its stability region reaches ``|lambda h| = 2.78`` on the
negative real axis, which is what sets the step relative to the fastest pole of the plant,
the armature electrical time constant ``L / R``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from transradial_sim.algorithm.protocols import Derivative

__all__ = ["ForwardEuler", "RungeKutta4", "integrate_fixed_step", "richardson_order"]


@dataclass(frozen=True, slots=True)
class RungeKutta4:
    """Classical four stage, fourth order Runge-Kutta scheme."""

    @property
    def order(self) -> int:
        """Order of accuracy of the global error."""
        return 4

    @property
    def stages(self) -> int:
        """Number of right hand side evaluations per step."""
        return 4

    def step(
        self, derivative: Derivative, state: list[float], time_s: float, step_s: float
    ) -> list[float]:
        """Advance ``state`` by one step and return the new state."""
        half = 0.5 * step_s
        k1 = derivative(state, time_s)
        stage = [state[i] + half * k1[i] for i in range(len(state))]
        k2 = derivative(stage, time_s + half)
        stage = [state[i] + half * k2[i] for i in range(len(state))]
        k3 = derivative(stage, time_s + half)
        stage = [state[i] + step_s * k3[i] for i in range(len(state))]
        k4 = derivative(stage, time_s + step_s)
        sixth = step_s / 6.0
        return [
            state[i] + sixth * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            for i in range(len(state))
        ]


@dataclass(frozen=True, slots=True)
class ForwardEuler:
    """First order explicit scheme, kept as the reference point of the order study."""

    @property
    def order(self) -> int:
        """Order of accuracy of the global error."""
        return 1

    @property
    def stages(self) -> int:
        """Number of right hand side evaluations per step."""
        return 1

    def step(
        self, derivative: Derivative, state: list[float], time_s: float, step_s: float
    ) -> list[float]:
        """Advance ``state`` by one step and return the new state."""
        rate = derivative(state, time_s)
        return [state[i] + step_s * rate[i] for i in range(len(state))]


def integrate_fixed_step(
    integrator: RungeKutta4 | ForwardEuler,
    derivative: Derivative,
    state: list[float],
    start_s: float,
    steps: int,
    step_s: float,
) -> list[float]:
    """Advance ``state`` by ``steps`` steps and return the final state."""
    current = state
    time_s = start_s
    for _ in range(steps):
        current = integrator.step(derivative, current, time_s, step_s)
        time_s += step_s
    return current


def richardson_order(coarse: float, medium: float, fine: float) -> float:
    """Estimate the observed order of accuracy from three step halvings.

    With errors proportional to ``h^p`` the ratio of successive differences is ``2^p``, so
    the observed order is ``log2((coarse - medium) / (medium - fine))``. Returning the
    observed order rather than a raw difference lets a convergence test state what it is
    measuring.

    Raises:
        ValueError: If the differences vanish, which leaves the order undefined.
    """
    numerator = coarse - medium
    denominator = medium - fine
    if denominator == 0.0 or numerator == 0.0:
        raise ValueError("cannot estimate an order from identical results")
    return math.log2(abs(numerator / denominator))
