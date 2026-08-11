"""Closed loop position and force control with the motor model in the loop.

The point of the example is the bandwidth. Both loops sit on top of a current regulator
whose corner frequency is set by the armature, and on top of a tendon whose series
compliance and capstan friction stand between the motor and the fingertip. Neither loop
can be made fast by choosing larger gains, and the numbers below show what each one
actually achieves.
"""

from __future__ import annotations

import math

from _common import parse_options

from transradial_sim.algorithm.controllers import ForceController, PositionController
from transradial_sim.pipeline.scenario import (
    COMPLIANT_CONTACT,
    FLAT_PLATE,
    build_plant,
    current_controller,
)
from transradial_sim.pipeline.simulate import ScenarioConfig, run_scenario
from transradial_sim.pipeline.trace import SimulationTrace

TARGETS_N = (4.0, 8.0, 12.0)
"""Fingertip force set points, in N."""

POSITION_TARGET_RAD = 250.0
"""Motor shaft angle set point, in rad. About two thirds of a full closure."""


def first_time_at(trace: SimulationTrace, values: list[float], threshold: float) -> float:
    """Return the first sample time at which ``values`` reaches ``threshold``, in s."""
    for index, value in enumerate(values):
        if value >= threshold:
            return float(trace.time_s[index])
    return float(trace.time_s[-1])


def main() -> None:
    """Run a position step and a set of force steps, and report the tracking."""
    options = parse_options(__doc__ or "closed loop control")
    duration = 1.20 if options.quick else 3.00
    step = 5.0e-5
    targets = (8.0,) if options.quick else TARGETS_N

    params = build_plant()
    inner = current_controller(params, 0.0)
    corner_hz = 1.0 / (2.0 * math.pi * params.motor.electrical_time_constant_s)
    print(f"armature corner frequency        {corner_hz:.0f} Hz")
    print(f"current loop bandwidth           {inner.bandwidth_hz:.0f} Hz")
    print(f"current reference slew limit     {inner.slew_limit_a_per_s:.0f} A/s")
    print(f"controller sample rate           {1.0 / inner.sample_period_s:.0f} Hz")

    print()
    print("position control of the motor shaft, no object present")
    controller = PositionController(
        inner=current_controller(params, 0.0),
        proportional_a_per_rad=0.010,
        integral_a_per_rad_s=0.050,
        derivative_a_s_per_rad=0.0030,
        target_rad=POSITION_TARGET_RAD,
    )
    config = ScenarioConfig(
        name="position step",
        params=params,
        duration_s=duration,
        step_s=step,
        sample_stride=10,
    )
    trace = run_scenario(config, controller)
    angles = [float(value) for value in trace.motor_angle_rad]
    peak = max(angles)
    settled = angles[-1]
    rise = first_time_at(trace, angles, 0.9 * POSITION_TARGET_RAD)
    print(f"  target                         {POSITION_TARGET_RAD:.1f} rad")
    print(f"  settled                        {settled:.2f} rad")
    print(
        f"  steady state error             {settled - POSITION_TARGET_RAD:+.2f} rad "
        f"({100.0 * (settled / POSITION_TARGET_RAD - 1.0):+.2f} percent)"
    )
    print(
        f"  overshoot                      {100.0 * (peak / POSITION_TARGET_RAD - 1.0):.1f} percent"
    )
    print(f"  time to 90 percent of target   {rise:.3f} s")

    print()
    print("fingertip force control against a compliant flat object")
    print(f"{'target N':>10}{'settled N':>12}{'error N':>10}{'peak N':>10}{'to 90 pct s':>13}")
    force_params = build_plant(obstacle=FLAT_PLATE, contact=COMPLIANT_CONTACT)
    for target in targets:
        force_controller = ForceController(
            inner=current_controller(force_params, 0.0),
            proportional_a_per_n=0.0010,
            integral_a_per_n_s=0.050,
            target_n=target,
            integral_limit_a=force_params.current_limit_a,
        )
        force_config = ScenarioConfig(
            name=f"force step to {target:.0f} N",
            params=force_params,
            duration_s=duration,
            step_s=step,
            sample_stride=10,
        )
        force_trace = run_scenario(force_config, force_controller)
        forces = [float(value) for value in force_trace.fingertip_force_n]
        reached = first_time_at(force_trace, forces, 0.9 * target)
        print(
            f"{target:>10.1f}{forces[-1]:>12.2f}{forces[-1] - target:>10.2f}"
            f"{max(forces):>10.2f}{reached:>13.3f}"
        )

    print()
    print("The force loop is deliberately slow. Its gain is bounded above by the series")
    print("compliance of the tendon and by the current loop underneath it, and the finger")
    print("must travel to the object before any force exists at all, so the transient")
    print("contains an approach that no amount of feedback can remove.")


if __name__ == "__main__":
    main()
