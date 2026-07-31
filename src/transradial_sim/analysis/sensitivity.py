"""Sensitivity of the reported performance to tendon elasticity and capstan friction.

The two parameters swept here are the ones a designer can actually change, and the ones
that are least well known before a prototype exists. Sweeping them turns two uncertain
inputs into a stated range on every reported output.
"""

from __future__ import annotations

from dataclasses import dataclass

from transradial_sim.analysis.metrics import closing_time_s, grasp_summary
from transradial_sim.pipeline.scenario import (
    LARGE_CYLINDER,
    STIFF_CONTACT,
    build_plant,
    current_controller,
    grasp_controller,
)
from transradial_sim.pipeline.simulate import ScenarioConfig, run_scenario

__all__ = [
    "FRICTION_VALUES",
    "STIFFNESS_VALUES",
    "SensitivityPoint",
    "relative_spread",
    "sweep_capstan_friction",
    "sweep_tendon_stiffness",
]

STIFFNESS_VALUES: tuple[float, ...] = (5.0e3, 1.0e4, 2.0e4, 5.0e4, 2.0e5)
"""Tendon series stiffness values swept, in N/m.

The reference is 2e4 N/m. The range spans a slack polymer cord with compliant
terminations at the low end and a short steel cable at the high end.
"""

FRICTION_VALUES: tuple[float, ...] = (0.0, 0.05, 0.10, 0.147, 0.20)
"""Capstan friction coefficients swept.

Zero is the frictionless reference that isolates the effect, 0.147 is the measured value
used for the reference configuration, and 0.20 represents a contaminated or worn routing.
"""


@dataclass(frozen=True, slots=True)
class SensitivityPoint:
    """One configuration in a sweep and the metrics it produced."""

    parameter: str
    value: float
    closing_time_s: float
    grasp_force_n: float
    fingertip_force_n: float
    finger_tension_n: float
    transmission_ratio: float
    """Measured tendon tension at the finger divided by the tension at the drive."""

    battery_energy_j: float
    capstan_loss_j: float


def _evaluate(
    parameter: str,
    value: float,
    stiffness_n_per_m: float | None,
    friction: float | None,
    duration_s: float,
    step_s: float,
    squeeze_start_s: float,
) -> SensitivityPoint:
    closing_plant = build_plant(
        tendon_stiffness_n_per_m=stiffness_n_per_m,
        tendon_friction_coefficient=friction,
    )
    closing = run_scenario(
        ScenarioConfig(
            name=f"{parameter}={value:g} closing",
            params=closing_plant,
            duration_s=0.60,
            step_s=step_s,
            sample_stride=20,
        ),
        current_controller(closing_plant),
    )

    grasp_plant = build_plant(
        obstacle=LARGE_CYLINDER,
        contact=STIFF_CONTACT,
        tendon_stiffness_n_per_m=stiffness_n_per_m,
        tendon_friction_coefficient=friction,
    )
    grasp = run_scenario(
        ScenarioConfig(
            name=f"{parameter}={value:g} grasp",
            params=grasp_plant,
            duration_s=squeeze_start_s + duration_s,
            step_s=step_s,
            sample_stride=20,
        ),
        grasp_controller(grasp_plant, squeeze_start_s=squeeze_start_s),
    )
    settled = grasp_summary(grasp)
    return SensitivityPoint(
        parameter=parameter,
        value=value,
        closing_time_s=closing_time_s(closing),
        grasp_force_n=settled.total_force_n,
        fingertip_force_n=settled.fingertip_force_n,
        finger_tension_n=settled.finger_tension_n,
        transmission_ratio=settled.measured_capstan_ratio,
        battery_energy_j=settled.battery_energy_j,
        capstan_loss_j=grasp.final_accumulator("capstan_loss_j"),
    )


def sweep_tendon_stiffness(
    values: tuple[float, ...] = STIFFNESS_VALUES,
    duration_s: float = 0.70,
    step_s: float = 5.0e-5,
    squeeze_start_s: float = 1.00,
) -> tuple[SensitivityPoint, ...]:
    """Sweep the tendon series stiffness and return one point per value."""
    return tuple(
        _evaluate(
            "tendon_stiffness_n_per_m",
            value,
            value,
            None,
            duration_s,
            step_s,
            squeeze_start_s,
        )
        for value in values
    )


def sweep_capstan_friction(
    values: tuple[float, ...] = FRICTION_VALUES,
    duration_s: float = 0.70,
    step_s: float = 5.0e-5,
    squeeze_start_s: float = 1.00,
) -> tuple[SensitivityPoint, ...]:
    """Sweep the capstan friction coefficient and return one point per value."""
    return tuple(
        _evaluate(
            "friction_coefficient",
            value,
            None,
            value,
            duration_s,
            step_s,
            squeeze_start_s,
        )
        for value in values
    )


def relative_spread(points: tuple[SensitivityPoint, ...], attribute: str) -> float:
    """Return the spread of one metric across a sweep, relative to its mean.

    Raises:
        ValueError: If the sweep is empty or the mean of the metric is zero.
    """
    if not points:
        raise ValueError("sweep contains no points")
    values = [float(getattr(point, attribute)) for point in points]
    mean = sum(values) / len(values)
    if mean == 0.0:
        raise ValueError(f"mean of {attribute} is zero, relative spread is undefined")
    return (max(values) - min(values)) / abs(mean)
