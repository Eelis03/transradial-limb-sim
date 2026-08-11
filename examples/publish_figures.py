"""Regenerate the three figures tracked in ``docs/figures``.

This is the one command that produces every tracked image, so the pictures in the README
and the numbers in the README come from the same run of the same code. It writes into
``docs/figures`` rather than into ``outputs``, because these three are published rather
than scratch output.

Matplotlib does not produce byte identical files across platforms or releases, so the
tracked images are snapshots that are refreshed deliberately and are not compared byte for
byte by the continuous integration workflow.
"""

from __future__ import annotations

from pathlib import Path

from _common import parse_options

from transradial_sim.analysis.energetics import energy_budget, force_chain
from transradial_sim.analysis.figures import (
    PUBLISHED_DPI,
    published_energy_breakdown_figure,
    published_grasp_posture_figure,
    published_tendon_tension_figure,
)
from transradial_sim.pipeline.scenario import (
    TEST_OBJECTS,
    grasp_controller,
    grasp_scenario,
    stall_scenario,
)
from transradial_sim.pipeline.simulate import run_scenario

FIGURE_DIRECTORY = Path(__file__).resolve().parent.parent / "docs" / "figures"


def main() -> None:
    """Build the tracked figures and report where each one went."""
    options = parse_options(__doc__ or "publish figures")
    duration = 0.30 if options.quick else 1.70
    squeeze = 0.10 if options.quick else 1.00
    step = 1.0e-4 if options.quick else 5.0e-5

    stall = stall_scenario(duration_s=duration, step_s=step)
    stall_trace = run_scenario(stall, grasp_controller(stall.params, squeeze_start_s=squeeze))
    budget = energy_budget(stall_trace)
    chain = force_chain(stall.params)

    grasps = []
    for name, obstacle in TEST_OBJECTS:
        config = grasp_scenario(name, obstacle, duration_s=duration, step_s=step)
        grasps.append(
            run_scenario(config, grasp_controller(config.params, squeeze_start_s=squeeze))
        )

    figures = (
        ("energy-breakdown.png", published_energy_breakdown_figure(budget)),
        ("grasp-postures.png", published_grasp_posture_figure(grasps)),
        ("tendon-tension.png", published_tendon_tension_figure(stall_trace, chain)),
    )

    if options.no_figures:
        for name, _ in figures:
            print(f"would write {FIGURE_DIRECTORY / name}")
        return

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, figure in figures:
        path = FIGURE_DIRECTORY / name
        figure.savefig(path, dpi=PUBLISHED_DPI, facecolor=figure.get_facecolor())
        size = path.stat().st_size
        total += size
        print(f"wrote {path} ({size} bytes)")
    print(f"total {total} bytes")


if __name__ == "__main__":
    main()
