"""Figure builders. The only place in the project that imports matplotlib.

Each function returns a figure rather than saving or showing it, so the caller decides
where output goes and the analysis layer stays free of input and output.

The three builders whose names begin with ``published_`` are the ones whose output is
tracked in ``docs/figures``. They share one palette and one axis treatment so that the
three read as a set, and they are sized and captioned for the width a README renders at
rather than for a screen.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from transradial_sim.analysis.energetics import EnergyBudget, ForceChain
from transradial_sim.analysis.sensitivity import SensitivityPoint
from transradial_sim.model.finger import joint_origins
from transradial_sim.pipeline.trace import SimulationTrace

__all__ = [
    "LOSS_LABELS",
    "PUBLISHED_DPI",
    "closing_figure",
    "grasp_posture_figure",
    "loss_figure",
    "published_energy_breakdown_figure",
    "published_grasp_posture_figure",
    "published_tendon_tension_figure",
    "sensitivity_figure",
]

PUBLISHED_DPI: Final[int] = 100
"""Dots per inch used for the tracked figures.

Chosen with the figure sizes below so that each image lands near 750 pixels wide, which
is what a README column renders at, and so that the three together stay well inside the
250 kilobyte budget the portfolio checker enforces. Raising it buys no legibility in a
README and costs bytes roughly as the square.
"""

_SURFACE: Final[str] = "#fcfcfb"
_INK: Final[str] = "#0b0b0b"
_MUTED: Final[str] = "#52514e"
_GRID: Final[str] = "#dedcd6"
_PRIMARY: Final[str] = "#2a78d6"
_ACCENT: Final[str] = "#eb6834"

LOSS_LABELS: Final[dict[str, str]] = {
    "motor_copper_loss": "motor copper, i squared R",
    "driver_loss": "bridge conduction and quiescent",
    "gearbox_loss": "gearbox tooth friction",
    "capstan_loss": "capstan friction in the routing",
    "motor_friction_loss": "motor brush and bearing friction",
    "joint_damping_loss": "joint damping",
    "battery_internal_loss": "battery internal resistance",
    "tendon_damper_loss": "tendon viscoelastic damping",
    "joint_limit_loss": "joint end stops",
}
"""Readable names for the loss channels, keyed by the name the budget reports."""


def _style_axis(axis: Axes, *, grid_axis: Literal["both", "x", "y"] | None = "x") -> None:
    """Apply the shared axis treatment: recessive grid, no box, muted tick labels."""
    axis.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(_GRID)
    axis.tick_params(colors=_MUTED, labelsize=8, length=3)
    if grid_axis is not None:
        axis.grid(True, axis=grid_axis, color=_GRID, linewidth=0.7)
        axis.set_axisbelow(True)


def _joint_angles_deg(angles_rad: Sequence[float]) -> list[float]:
    """Return joint angles in degrees, for the panel captions."""
    return [float(np.degrees(angle)) for angle in angles_rad]


def _common_window_mm(traces: Sequence[SimulationTrace]) -> tuple[float, float]:
    """Return one square window in mm that contains every posture and every object.

    A shared window is what makes the panels comparable. Squaring it keeps the aspect
    ratio true in both directions, so a round object is drawn round.
    """
    lows: list[float] = []
    highs: list[float] = []
    for trace in traces:
        angles = tuple(float(a) for a in trace.joint_angles_rad[-1])
        origins, _, _ = joint_origins(trace.params.finger, angles)
        for point in origins:
            lows.extend((1000.0 * point[0], 1000.0 * point[1]))
            highs.extend((1000.0 * point[0], 1000.0 * point[1]))
        obstacle = trace.params.obstacle
        centre = getattr(obstacle, "centre_m", None)
        radius = getattr(obstacle, "radius_m", None)
        if centre is not None and radius is not None:
            lows.extend((1000.0 * (centre[0] - radius), 1000.0 * (centre[1] - radius)))
            highs.extend((1000.0 * (centre[0] + radius), 1000.0 * (centre[1] + radius)))
    low = min(lows) if lows else 0.0
    high = max(highs) if highs else 1.0
    margin = 0.08 * (high - low)
    return low - margin, high + margin


def closing_figure(trace: SimulationTrace) -> Figure:
    """Plot joint angles, armature current and tendon tension against time."""
    figure, axes = plt.subplots(3, 1, figsize=(7.0, 8.0), sharex=True)
    names = [phalanx.name for phalanx in trace.params.finger.phalanges]
    for index, name in enumerate(names):
        axes[0].plot(trace.time_s, np.degrees(trace.joint_angles_rad[:, index]), label=name)
    axes[0].set_ylabel("joint angle, deg")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(trace.time_s, trace.current_a, color="tab:red")
    axes[1].axhline(trace.params.current_limit_a, color="k", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("armature current, A")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(trace.time_s, trace.drive_tension_n, label="at the drive")
    axes[2].plot(trace.time_s, trace.finger_tension_n, label="at the finger")
    axes[2].set_ylabel("tendon tension, N")
    axes[2].set_xlabel("time, s")
    axes[2].legend(loc="lower right")
    axes[2].grid(True, alpha=0.3)

    figure.suptitle(f"Scenario: {trace.name}")
    figure.tight_layout()
    return figure


def grasp_posture_figure(traces: Sequence[SimulationTrace]) -> Figure:
    """Draw the settled finger posture for each of several grasps."""
    figure, axes = plt.subplots(1, len(traces), figsize=(4.0 * len(traces), 4.2), squeeze=False)
    for column, trace in enumerate(traces):
        axis = axes[0][column]
        angles = tuple(float(a) for a in trace.joint_angles_rad[-1])
        origins, _, _ = joint_origins(trace.params.finger, angles)
        xs = [1000.0 * point[0] for point in origins]
        ys = [1000.0 * point[1] for point in origins]
        axis.plot(xs, ys, "-o", color="tab:blue")
        obstacle = trace.params.obstacle
        centre = getattr(obstacle, "centre_m", None)
        radius = getattr(obstacle, "radius_m", None)
        if centre is not None and radius is not None:
            angle = np.linspace(0.0, 2.0 * np.pi, 120)
            axis.plot(
                1000.0 * (centre[0] + radius * np.cos(angle)),
                1000.0 * (centre[1] + radius * np.sin(angle)),
                color="tab:orange",
            )
        surface = getattr(obstacle, "point_m", None)
        if surface is not None:
            axis.axhline(1000.0 * surface[1], color="tab:orange")
        axis.set_aspect("equal")
        axis.set_title(trace.name)
        axis.set_xlabel("x, mm")
        if column == 0:
            axis.set_ylabel("y, mm")
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    return figure


def loss_figure(budget: EnergyBudget) -> Figure:
    """Draw the measured energy budget as a horizontal bar chart."""
    labels = [row.name.replace("_", " ") for row in budget.losses]
    values = [row.energy_j for row in budget.losses]
    labels.append("work on object")
    values.append(budget.object_work_j)
    labels.append("still stored")
    values.append(budget.stored_change_j)

    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    positions = np.arange(len(labels))
    axis.barh(positions, values, color="tab:blue")
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    axis.set_xlabel("energy, J")
    axis.set_title(f"Battery energy {budget.battery_energy_j:.3f} J")
    axis.grid(True, axis="x", alpha=0.3)
    figure.tight_layout()
    return figure


def sensitivity_figure(
    stiffness: Sequence[SensitivityPoint],
    friction: Sequence[SensitivityPoint],
) -> Figure:
    """Plot closing time and grasp force against each swept parameter."""
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.5))

    axes[0][0].semilogx([p.value for p in stiffness], [p.closing_time_s for p in stiffness], "-o")
    axes[0][0].set_xlabel("tendon stiffness, N/m")
    axes[0][0].set_ylabel("closing time, s")

    axes[0][1].semilogx([p.value for p in stiffness], [p.grasp_force_n for p in stiffness], "-o")
    axes[0][1].set_xlabel("tendon stiffness, N/m")
    axes[0][1].set_ylabel("grasp force, N")

    axes[1][0].plot([p.value for p in friction], [p.closing_time_s for p in friction], "-o")
    axes[1][0].set_xlabel("capstan friction coefficient")
    axes[1][0].set_ylabel("closing time, s")

    axes[1][1].plot([p.value for p in friction], [p.grasp_force_n for p in friction], "-o")
    axes[1][1].set_xlabel("capstan friction coefficient")
    axes[1][1].set_ylabel("grasp force, N")

    for row in axes:
        for axis in row:
            axis.grid(True, alpha=0.3)
    figure.tight_layout()
    return figure


def published_energy_breakdown_figure(budget: EnergyBudget) -> Figure:
    """Draw where one grasp spends the battery energy, largest destination first.

    The point of the figure is the bar that is not there. The work handed to the object
    is a fraction of a percent of what the battery supplies, and a linear axis shared with
    the copper loss is the only honest way to show that. Every row therefore carries its
    own value as a direct label, so the rows with no visible bar still report a number.
    """
    entries = [
        (LOSS_LABELS.get(row.name, row.name.replace("_", " ")), row.energy_j, row.share, False)
        for row in budget.losses
    ]
    entries.append(
        ("still stored in springs and cord", budget.stored_change_j, budget.stored_share, False)
    )
    entries.append(
        ("work delivered to the object", budget.object_work_j, budget.object_share, True)
    )
    entries.sort(key=lambda item: item[1], reverse=True)

    figure, axis = plt.subplots(figsize=(7.4, 4.0), dpi=PUBLISHED_DPI)
    figure.patch.set_facecolor(_SURFACE)
    _style_axis(axis)

    positions = np.arange(len(entries))
    values = [item[1] for item in entries]
    colours = [_ACCENT if item[3] else _PRIMARY for item in entries]
    axis.barh(positions, values, height=0.62, color=colours)

    span = max(values) if values else 1.0
    for position, (_, value, share, highlighted) in enumerate(entries):
        if highlighted:
            # The bar is too short to see, which is the finding. A marker at its end keeps
            # the row locatable without drawing a bar wider than the number it stands for.
            axis.plot([value], [position], marker="o", markersize=5.5, color=_ACCENT)
        axis.text(
            value + 0.020 * span,
            position,
            f"{value:.4f} J    {100.0 * share:.2f} percent",
            va="center",
            fontsize=8,
            color=_INK if highlighted else _MUTED,
            fontweight="bold" if highlighted else "normal",
        )

    axis.set_yticks(positions)
    axis.set_yticklabels([item[0] for item in entries], fontsize=8.5, color=_INK)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.40 * span)
    axis.set_xlabel("energy drawn from the battery, J", fontsize=9, color=_MUTED)
    axis.set_title(
        f"One grasp costs {budget.battery_energy_j:.4f} J. "
        f"{100.0 * budget.object_share:.2f} percent of it reaches the object.",
        fontsize=10.5,
        color=_INK,
        loc="left",
        pad=10,
    )
    figure.tight_layout()
    return figure


def published_grasp_posture_figure(traces: Sequence[SimulationTrace]) -> Figure:
    """Draw the settled posture of the same finger on each object, side by side.

    One motor, one tendon, one controller, three shapes. The postures are the measurement:
    a finger with a fixed ratio between its joints would draw the same outline three times.
    """
    figure, axes = plt.subplots(
        1,
        len(traces),
        figsize=(7.4, 3.0),
        dpi=PUBLISHED_DPI,
        squeeze=False,
    )
    figure.patch.set_facecolor(_SURFACE)

    # One square window for every panel, so the three outlines are directly comparable and
    # the 40 mm cylinder is drawn smaller than the 60 mm one rather than merely labelled so.
    limits = _common_window_mm(traces)

    for column, trace in enumerate(traces):
        axis = axes[0][column]
        _style_axis(axis, grid_axis=None)
        angles = tuple(float(a) for a in trace.joint_angles_rad[-1])
        origins, _, _ = joint_origins(trace.params.finger, angles)
        xs = [1000.0 * point[0] for point in origins]
        ys = [1000.0 * point[1] for point in origins]

        obstacle = trace.params.obstacle
        centre = getattr(obstacle, "centre_m", None)
        radius = getattr(obstacle, "radius_m", None)
        if centre is not None and radius is not None:
            sweep = np.linspace(0.0, 2.0 * np.pi, 180)
            circle_x = 1000.0 * (centre[0] + radius * np.cos(sweep))
            circle_y = 1000.0 * (centre[1] + radius * np.sin(sweep))
            axis.fill(circle_x, circle_y, color=_ACCENT, alpha=0.16, linewidth=0.0)
            axis.plot(circle_x, circle_y, color=_ACCENT, linewidth=1.6)
        surface = getattr(obstacle, "point_m", None)
        if surface is not None:
            # A half space has no outline to fill, so the plate is drawn as its face only.
            height = 1000.0 * surface[1]
            axis.plot(limits, (height, height), color=_ACCENT, linewidth=1.6)

        axis.plot(xs, ys, "-", color=_PRIMARY, linewidth=2.6, solid_capstyle="round")
        axis.plot(xs[:-1], ys[:-1], "o", color=_PRIMARY, markersize=4.0)
        axis.plot(xs[-1:], ys[-1:], "o", color=_PRIMARY, markersize=8.5)
        axis.set_xlim(*limits)
        axis.set_ylim(*limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(trace.name, fontsize=9.5, color=_INK)
        axis.set_xlabel(
            ", ".join(f"{value:.0f}" for value in _joint_angles_deg(angles)) + " deg",
            fontsize=8.5,
            color=_MUTED,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for side in ("left", "bottom"):
            axis.spines[side].set_visible(False)

    figure.suptitle(
        "One actuator, three objects, three postures. The large dot is the fingertip.",
        fontsize=10.5,
        color=_INK,
        x=0.015,
        ha="left",
    )
    figure.tight_layout()
    return figure


def published_tendon_tension_figure(trace: SimulationTrace, chain: ForceChain) -> Figure:
    """Draw the tension at each end of the routing through one grasp.

    The gap between the two curves is the capstan loss, measured rather than assumed, and
    the gap between the settled finger tension and the dashed quasi static line is the
    stretch the impact put into the cord that the drive cannot take back out.
    """
    figure, axis = plt.subplots(figsize=(7.4, 3.3), dpi=PUBLISHED_DPI)
    figure.patch.set_facecolor(_SURFACE)
    _style_axis(axis, grid_axis="y")

    axis.plot(
        trace.time_s,
        trace.drive_tension_n,
        color=_PRIMARY,
        linewidth=2.0,
        label="at the drive pulley",
    )
    axis.plot(
        trace.time_s,
        trace.finger_tension_n,
        color=_ACCENT,
        linewidth=2.0,
        label="at the finger, after the routing",
    )
    for value, text in (
        (chain.drive_tension_n, "chain, at the drive"),
        (chain.finger_tension_n, "chain, at the finger"),
    ):
        axis.axhline(value, color=_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
        axis.text(
            float(trace.time_s[-1]),
            value,
            f"  {text}, {value:.1f} N",
            va="center",
            ha="left",
            fontsize=7.5,
            color=_MUTED,
        )

    axis.set_xlim(float(trace.time_s[0]), float(trace.time_s[-1]))
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel("time, s", fontsize=9, color=_MUTED)
    axis.set_ylabel("tendon tension, N", fontsize=9, color=_MUTED)
    axis.set_title(
        "The routing takes a fixed fraction. The impact adds a stretch the drive cannot undo.",
        fontsize=10.5,
        color=_INK,
        loc="left",
        pad=10,
    )
    legend = axis.legend(loc="upper left", fontsize=8.5, frameon=False)
    for label in legend.get_texts():
        label.set_color(_INK)
    figure.tight_layout(rect=(0.0, 0.0, 0.845, 1.0))
    return figure
