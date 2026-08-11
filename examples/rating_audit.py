"""Check the reference runs against the published ratings of the parts they drive.

Runs the free closing scenario and the rigid grasp and prints every catalogue rating the
two touch, next to the peak each run reached and the time it spent above it.
"""

from __future__ import annotations

from _common import parse_options

from transradial_sim.analysis.ratings import RatingAudit, rating_audit
from transradial_sim.model.units import rad_s_to_rpm
from transradial_sim.pipeline.scenario import (
    closing_scenario,
    current_controller,
    grasp_controller,
    stall_scenario,
)
from transradial_sim.pipeline.simulate import run_scenario


def report(audit: RatingAudit) -> None:
    """Print one audit as a table, one row per published rating."""
    print(f"{audit.name}, {audit.duration_s:.2f} s")
    print(f"  {'rating':<36}{'published':>14}{'peak':>14}{'margin':>9}{'time above':>13}")
    for check in audit.checks:
        published = f"{check.rating:.4g} {check.unit}"
        peak = f"{check.peak:.4g} {check.unit}"
        print(
            f"  {check.name:<36}{published:>14}{peak:>14}"
            f"{check.margin:>9.3f}{check.time_above_s:>11.3f} s"
        )
    if audit.within_ratings:
        print(f"  inside every rating, closest is {audit.worst.name} at {audit.worst.margin:.3f}")
        return
    for check in audit.exceedances:
        print(
            f"  exceeded: {check.name} at {check.margin:.3f} times its rating, "
            f"for {100.0 * check.share:.1f} percent of the run"
        )


def main() -> None:
    """Audit the free closing run and the rigid grasp against the catalogue."""
    options = parse_options(__doc__ or "rating audit")
    closing = closing_scenario(
        duration_s=0.20 if options.quick else 0.80,
        step_s=5.0e-5 if options.quick else 2.0e-5,
    )
    grasp = stall_scenario(duration_s=1.00 if options.quick else 1.70, step_s=5.0e-5)
    squeeze = 0.70 if options.quick else 1.00

    closing_audit = rating_audit(run_scenario(closing, current_controller(closing.params)))
    grasp_audit = rating_audit(
        run_scenario(grasp, grasp_controller(grasp.params, squeeze_start_s=squeeze))
    )
    report(closing_audit)
    print()
    report(grasp_audit)

    print()
    print("the two gearhead ratings in the units the catalogue quotes them in")
    for label, audit in (("free closing", closing_audit), ("rigid grasp", grasp_audit)):
        speed = audit.check("gearbox input speed")
        torque = audit.check("gearbox continuous output torque")
        print(
            f"  {label:<14}input speed {rad_s_to_rpm(speed.peak):>6.0f} rpm of "
            f"{rad_s_to_rpm(speed.rating):.0f} rpm, output torque {torque.peak:.4f} Nm of "
            f"{torque.rating:.2f} Nm continuous"
        )


if __name__ == "__main__":
    main()
