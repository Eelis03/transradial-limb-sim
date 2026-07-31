"""Sensitivity of closing time and grasp force to tendon elasticity and capstan friction.

Both parameters are poorly known before a prototype exists, so every reported figure is
only as firm as the range they are allowed to take. The study replaces a single number
with a range and names which of the two parameters each metric is sensitive to.
"""

from __future__ import annotations

from _common import parse_options, save

from transradial_sim.analysis.figures import sensitivity_figure
from transradial_sim.analysis.sensitivity import (
    FRICTION_VALUES,
    STIFFNESS_VALUES,
    SensitivityPoint,
    relative_spread,
    sweep_capstan_friction,
    sweep_tendon_stiffness,
)


def _print_table(title: str, unit: str, points: tuple[SensitivityPoint, ...]) -> None:
    print(title)
    print(
        f"{unit:>12}{'closing s':>12}{'grasp N':>10}{'fingertip N':>13}"
        f"{'finger T N':>12}{'ratio':>8}{'capstan J':>11}"
    )
    for point in points:
        print(
            f"{point.value:>12g}{point.closing_time_s:>12.3f}{point.grasp_force_n:>10.2f}"
            f"{point.fingertip_force_n:>13.2f}{point.finger_tension_n:>12.2f}"
            f"{point.transmission_ratio:>8.3f}{point.capstan_loss_j:>11.4f}"
        )
    for attribute in ("closing_time_s", "grasp_force_n", "finger_tension_n"):
        spread = relative_spread(points, attribute)
        print(f"  relative spread of {attribute:<18}{100.0 * spread:6.1f}%")
    print()


def main() -> None:
    """Run both sweeps and print the tables."""
    options = parse_options(__doc__ or "sensitivity study")
    if options.quick:
        stiffness_values = (5.0e3, 2.0e5)
        friction_values = (0.0, 0.147)
        duration = 0.30
        step = 5.0e-5
        squeeze = 0.70
    else:
        stiffness_values = STIFFNESS_VALUES
        friction_values = FRICTION_VALUES
        duration = 0.70
        step = 5.0e-5
        squeeze = 1.00

    stiffness = sweep_tendon_stiffness(
        stiffness_values, duration_s=duration, step_s=step, squeeze_start_s=squeeze
    )
    friction = sweep_capstan_friction(
        friction_values, duration_s=duration, step_s=step, squeeze_start_s=squeeze
    )

    _print_table("tendon series stiffness sweep", "N/m", stiffness)
    _print_table("capstan friction coefficient sweep", "mu", friction)

    frictionless = friction[0]
    reference = next(point for point in friction if point.value == 0.147)
    loss = 100.0 * (1.0 - reference.finger_tension_n / frictionless.finger_tension_n)
    print(f"capstan friction alone removes {loss:.1f} percent of the tendon tension that")
    print("reaches the finger, relative to a frictionless routing of the same geometry.")

    save(sensitivity_figure(stiffness, friction), "sensitivity.png", options)


if __name__ == "__main__":
    main()
