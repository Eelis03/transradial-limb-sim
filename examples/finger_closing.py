"""Close the finger against nothing and report the speed metrics.

Produces the closing time, the peak motor and tendon speeds, and a figure showing joint
angles, armature current and tendon tension against time.
"""

from __future__ import annotations

import math

from _common import parse_options, save

from transradial_sim.analysis.figures import closing_figure
from transradial_sim.analysis.metrics import closing_time_s, free_running_speeds
from transradial_sim.model.units import rad_s_to_rpm
from transradial_sim.pipeline.scenario import closing_scenario, current_controller
from transradial_sim.pipeline.simulate import run_scenario
from transradial_sim.pipeline.trace import SimulationTrace


def peak_tip_speed(trace: SimulationTrace) -> float:
    """Return the largest fingertip speed in the trace, in m/s, by finite difference."""
    positions = trace.tip_position_m
    times = trace.time_s
    fastest = 0.0
    for index in range(1, len(times)):
        interval = float(times[index] - times[index - 1])
        dx = float(positions[index][0] - positions[index - 1][0])
        dy = float(positions[index][1] - positions[index - 1][1])
        fastest = max(fastest, (dx * dx + dy * dy) ** 0.5 / interval)
    return fastest


def main() -> None:
    """Run the free closing scenario and print the speed metrics."""
    options = parse_options(__doc__ or "finger closing")
    config = closing_scenario(
        duration_s=0.20 if options.quick else 0.80,
        step_s=5.0e-5 if options.quick else 2.0e-5,
    )
    params = config.params
    trace = run_scenario(config, current_controller(params))

    peak_speed = float(max(abs(value) for value in trace.motor_speed_rad_s))
    tendon_speed = peak_speed * params.tendon.drive_radius_m / params.gearbox.ratio
    _, no_load_rpm = free_running_speeds(params.motor, params.supply.open_circuit_voltage_v)
    final = trace.joint_angles_rad[-1]

    print(
        f"supply {params.supply.open_circuit_voltage_v:.1f} V, current limit "
        f"{params.current_limit_a:.3f} A"
    )
    print(f"closing time to 95 percent of travel   {closing_time_s(trace):.3f} s")
    print(f"peak motor speed                       {rad_s_to_rpm(peak_speed):.0f} rpm")
    print(f"motor no load speed at this supply     {no_load_rpm:.0f} rpm")
    print(
        f"gearbox recommended input speed limit  "
        f"{rad_s_to_rpm(params.gearbox.max_input_speed_rad_s):.0f} rpm"
    )
    print(f"peak tendon speed                      {1.0e3 * tendon_speed:.1f} mm/s")
    print(f"peak fingertip speed                   {1.0e3 * peak_tip_speed(trace):.0f} mm/s")
    print(
        "final joint angles, deg                "
        + ", ".join(f"{math.degrees(float(a)):.1f}" for a in final)
    )
    print(
        f"battery energy for one closure         "
        f"{trace.final_accumulator('battery_energy_j'):.3f} J"
    )

    save(closing_figure(trace), "closing.png", options)


if __name__ == "__main__":
    main()
