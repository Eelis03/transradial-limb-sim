"""The efficiency chain from battery to fingertip, both calculated and measured.

Prints the quasi static force chain stage by stage, then the measured energy budget of a
complete grasp of a rigid object, and finally the difference between the two.
"""

from __future__ import annotations

from _common import parse_options, save

from transradial_sim.analysis.energetics import energy_budget, force_chain
from transradial_sim.analysis.figures import loss_figure
from transradial_sim.analysis.metrics import grasp_summary
from transradial_sim.pipeline.scenario import grasp_controller, stall_scenario
from transradial_sim.pipeline.simulate import run_scenario


def main() -> None:
    """Print the force chain and the measured energy budget."""
    options = parse_options(__doc__ or "efficiency chain")
    config = stall_scenario(
        duration_s=1.00 if options.quick else 1.70,
        step_s=5.0e-5,
    )
    squeeze = 0.70 if options.quick else 1.00
    params = config.params
    chain = force_chain(params)

    print("quasi static force chain at the drive current limit")
    print(f"{'stage':<34}{'output':>16}{'stage efficiency':>18}")
    for stage in chain.stages:
        value = f"{stage.value:.4g} {stage.unit}"
        print(f"{stage.name:<34}{value:>16}{stage.efficiency:>18.4f}")
    print(f"{'product, motor shaft to finger':<34}{'':>16}{chain.transmission_efficiency:>18.4f}")

    trace = run_scenario(config, grasp_controller(params, squeeze_start_s=squeeze))
    settled = grasp_summary(trace)
    budget = energy_budget(trace)

    print()
    print("measured at the settled grasp")
    print(
        f"  tendon tension at the drive   {settled.drive_tension_n:8.2f} N "
        f"(chain {chain.drive_tension_n:.2f} N)"
    )
    print(
        f"  tendon tension at the finger  {settled.finger_tension_n:8.2f} N "
        f"(chain {chain.finger_tension_n:.2f} N)"
    )
    print(
        f"  measured capstan ratio        {settled.measured_capstan_ratio:8.4f} "
        f"(chain {chain.stages[-1].efficiency:.4f})"
    )
    print(f"  total grasp force             {settled.total_force_n:8.2f} N")
    print(f"  fingertip force               {settled.fingertip_force_n:8.2f} N")

    print()
    print(f"energy budget for one grasp, battery energy {budget.battery_energy_j:.4f} J")
    print(f"{'destination':<28}{'energy, J':>12}{'share':>10}")
    for row in budget.losses:
        print(f"{row.name.replace('_', ' '):<28}{row.energy_j:>12.4f}{100.0 * row.share:>9.2f}%")
    print(
        f"{'work on the object':<28}{budget.object_work_j:>12.4f}"
        f"{100.0 * budget.object_share:>9.2f}%"
    )
    print(
        f"{'still stored at the end':<28}{budget.stored_change_j:>12.4f}"
        f"{100.0 * budget.stored_share:>9.2f}%"
    )
    print(
        f"{'balance residual':<28}{budget.residual_j:>12.3e}"
        f"{100.0 * budget.relative_residual:>9.4f}%"
    )

    save(loss_figure(budget), "energy_budget.png", options)


if __name__ == "__main__":
    main()
