"""Figure builders. The only place in the project that imports matplotlib.

Each function returns a figure rather than saving or showing it, so the caller decides
where output goes and the analysis layer stays free of input and output.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from transradial_sim.analysis.energetics import EnergyBudget
from transradial_sim.analysis.sensitivity import SensitivityPoint
from transradial_sim.model.finger import joint_origins
from transradial_sim.pipeline.trace import SimulationTrace

__all__ = [
    "closing_figure",
    "grasp_posture_figure",
    "loss_figure",
    "sensitivity_figure",
]


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
