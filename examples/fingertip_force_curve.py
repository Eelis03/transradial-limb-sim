"""Fingertip force against motor current, measured and compared with the transmission.

The transmission predicts a straight line through the origin offset by the motor's own
Coulomb friction. Measuring it is the check that the tendon, the routing and the finger
geometry were assembled consistently.
"""

from __future__ import annotations

from _common import parse_options

from transradial_sim.analysis.energetics import force_chain, frictionless_tension_n
from transradial_sim.analysis.metrics import grasp_summary
from transradial_sim.pipeline.scenario import (
    LARGE_CYLINDER,
    STIFF_CONTACT,
    build_plant,
    grasp_controller,
)
from transradial_sim.pipeline.simulate import ScenarioConfig, run_scenario

CURRENT_FRACTIONS = (0.4, 0.6, 0.8, 1.0)
"""Fractions of the drive current limit at which the force is measured."""


def main() -> None:
    """Sweep the grasp current and print measured against predicted force."""
    options = parse_options(__doc__ or "fingertip force curve")
    duration = 1.00 if options.quick else 1.70
    step = 5.0e-5
    squeeze = 0.70 if options.quick else 1.00
    fractions = (0.5, 1.0) if options.quick else CURRENT_FRACTIONS

    reference = build_plant(obstacle=LARGE_CYLINDER, contact=STIFF_CONTACT)
    limit = reference.current_limit_a

    print(
        f"{'current A':>11}{'sliding N':>12}{'measured N':>13}{'lossless N':>13}"
        f"{'grasp N':>10}{'tip N':>8}{'in band':>9}"
    )
    rows = []
    for fraction in fractions:
        current = fraction * limit
        params = build_plant(
            obstacle=LARGE_CYLINDER, contact=STIFF_CONTACT, current_limit_a=current
        )
        config = ScenarioConfig(
            name=f"grasp at {current:.3f} A",
            params=params,
            duration_s=duration,
            step_s=step,
            sample_stride=10,
        )
        trace = run_scenario(
            config,
            grasp_controller(params, approach_a=min(0.35, current), squeeze_start_s=squeeze),
        )
        settled = grasp_summary(trace)
        chain = force_chain(params, current)
        rows.append((current, settled.total_force_n))
        upper = frictionless_tension_n(params, current)
        inside = chain.finger_tension_n <= settled.finger_tension_n <= upper
        print(
            f"{current:>11.4f}{chain.finger_tension_n:>12.2f}"
            f"{settled.finger_tension_n:>13.2f}{upper:>13.2f}"
            f"{settled.total_force_n:>10.2f}{settled.fingertip_force_n:>8.2f}"
            f"{'yes' if inside else 'no':>9}"
        )

    print()
    first_current, first_force = rows[0]
    last_current, last_force = rows[-1]
    slope = (last_force - first_force) / (last_current - first_current)
    print(f"measured grasp force per ampere        {slope:.2f} N/A")
    predicted = (
        reference.motor.torque_constant_nm_per_a
        * reference.gearbox.efficiency
        * reference.gearbox.ratio
        / reference.tendon.drive_radius_m
    )
    print(f"tendon tension per ampere at the drive {predicted:.2f} N/A from the transmission")
    print("The sliding column is the quasi static chain with the gearbox and capstan")
    print("losses fully applied, and the lossless column is the same chain with both")
    print("removed. The measured tension sits above the sliding value by the residual")
    print("stretch the approach impact left in the cord, which a tendon cannot push back")
    print("out and a stuck gearbox will not back drive. The grasp force is a further")
    print("geometric fraction of the tension, set by the posture the finger settles into.")


if __name__ == "__main__":
    main()
