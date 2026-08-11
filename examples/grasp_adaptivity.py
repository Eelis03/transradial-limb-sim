"""Measure how the underactuated finger conforms to differently shaped objects.

One motor drives three joints through one tendon. If the mechanism is genuinely adaptive,
the finger must settle into a different posture and a different contact force distribution
for each object, without anything in the controller knowing which object is present. The
controller here is identical in all three runs.
"""

from __future__ import annotations

import math

from _common import parse_options, save

from transradial_sim.analysis.figures import grasp_posture_figure
from transradial_sim.analysis.metrics import grasp_summary
from transradial_sim.pipeline.scenario import TEST_OBJECTS, grasp_controller, grasp_scenario
from transradial_sim.pipeline.simulate import run_scenario


def main() -> None:
    """Close onto each test object and print the settled posture and forces."""
    options = parse_options(__doc__ or "grasp adaptivity")
    duration = 1.00 if options.quick else 1.70
    step = 5.0e-5
    squeeze = 0.70 if options.quick else 1.00

    traces = []
    summaries = []
    for name, obstacle in TEST_OBJECTS:
        config = grasp_scenario(name, obstacle, duration_s=duration, step_s=step)
        trace = run_scenario(config, grasp_controller(config.params, squeeze_start_s=squeeze))
        traces.append(trace)
        summaries.append(grasp_summary(trace))

    print(
        f"{'object':<16}{'proximal':>10}{'middle':>9}{'distal':>9}"
        f"{'total N':>10}{'spread N':>10}{'points':>8}"
    )
    for summary in summaries:
        angles = ", ".join(f"{value:.1f}" for value in summary.joint_angles_deg)
        forces = summary.phalanx_forces_n
        print(
            f"{summary.name:<16}"
            f"{forces[0]:>10.2f}{forces[1]:>9.2f}{forces[2]:>9.2f}"
            f"{summary.total_force_n:>10.2f}{summary.force_spread:>10.2f}"
            f"{summary.contact_points:>8d}   angles deg {angles}"
        )

    print()
    print("posture difference between object pairs, root mean square joint angle, deg")
    for first in range(len(summaries)):
        for second in range(first + 1, len(summaries)):
            left = summaries[first]
            right = summaries[second]
            squared = sum(
                (a - b) ** 2
                for a, b in zip(left.joint_angles_deg, right.joint_angles_deg, strict=True)
            )
            distance = math.sqrt(squared / len(left.joint_angles_deg))
            print(f"  {left.name} against {right.name}: {distance:.1f}")

    print()
    print("tendon tension at the finger is nearly the same in every grasp, because one")
    print("actuator sets it. What differs is how the finger distributes that tension.")
    for summary in summaries:
        print(f"  {summary.name:<16}{summary.finger_tension_n:8.2f} N")

    save(grasp_posture_figure(traces), "grasp_postures.png", options)


if __name__ == "__main__":
    main()
