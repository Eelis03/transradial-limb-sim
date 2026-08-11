"""Tests of the figure builders, including the three whose output is tracked.

A figure is the one artefact of this project a reader looks at before reading anything,
and it is also the easiest place for a number to drift away from the table beside it.
These tests therefore assert the contents of each figure against the data it was built
from, rather than only that a figure was produced.

The runs here are a twentieth of a second long. Every assertion below is about how a
figure is assembled from a trace, not about what the trace contains, so the cheapest trace
that has the right shape is the correct one to use.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import cache

import matplotlib.pyplot as plt
import pytest

from transradial_sim.analysis.energetics import energy_budget, force_chain
from transradial_sim.analysis.figures import (
    LOSS_LABELS,
    PUBLISHED_DPI,
    closing_figure,
    grasp_posture_figure,
    loss_figure,
    published_energy_breakdown_figure,
    published_grasp_posture_figure,
    published_tendon_tension_figure,
    sensitivity_figure,
)
from transradial_sim.analysis.sensitivity import SensitivityPoint
from transradial_sim.model.contact import ContactGeometry
from transradial_sim.model.finger import joint_origins
from transradial_sim.pipeline.scenario import (
    FLAT_PLATE,
    LARGE_CYLINDER,
    STIFF_CONTACT,
    TEST_OBJECTS,
    build_plant,
    current_controller,
)
from transradial_sim.pipeline.simulate import ScenarioConfig, run_scenario
from transradial_sim.pipeline.trace import SimulationTrace

SHORT_DURATION_S = 0.05
SHORT_STEP_S = 5.0e-5


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    """Close every figure a test opened, so the pyplot registry does not grow."""
    yield
    plt.close("all")


@cache
def _short_trace(name: str, obstacle: ContactGeometry | None) -> SimulationTrace:
    params = build_plant(obstacle=obstacle, contact=STIFF_CONTACT)
    config = ScenarioConfig(
        name=name,
        params=params,
        duration_s=SHORT_DURATION_S,
        step_s=SHORT_STEP_S,
        sample_stride=1,
    )
    return run_scenario(config, current_controller(params))


def _sweep(parameter: str, values: tuple[float, ...]) -> tuple[SensitivityPoint, ...]:
    """Build sweep points directly, since the figures under test only read their fields."""
    return tuple(
        SensitivityPoint(
            parameter=parameter,
            value=value,
            closing_time_s=0.36 + 0.001 * index,
            grasp_force_n=40.0 - 2.0 * index,
            fingertip_force_n=24.0 - 1.0 * index,
            finger_tension_n=124.0 - 5.0 * index,
            transmission_ratio=0.60 - 0.01 * index,
            battery_energy_j=3.9 + 0.1 * index,
            capstan_loss_j=0.28 - 0.01 * index,
        )
        for index, value in enumerate(values)
    )


def test_the_published_dpi_is_the_one_the_figures_are_saved_at() -> None:
    """The tracked figures are sized for a README column, not for a screen."""
    assert PUBLISHED_DPI == 100
    figure = published_energy_breakdown_figure(
        energy_budget(_short_trace("budget", LARGE_CYLINDER))
    )
    assert figure.dpi == PUBLISHED_DPI
    width_in, height_in = figure.get_size_inches()
    assert 700 <= width_in * PUBLISHED_DPI <= 800
    assert height_in < width_in


def test_energy_breakdown_draws_one_bar_per_destination_in_descending_order() -> None:
    """Every channel appears exactly once, largest first, with its own value as a label.

    Sorting is part of the argument the figure makes, so it is asserted rather than
    assumed, and the bar widths are compared against the budget so that a relabelled row
    cannot silently show the wrong quantity.
    """
    budget = energy_budget(_short_trace("budget", LARGE_CYLINDER))
    figure = published_energy_breakdown_figure(budget)
    axis = figure.axes[0]
    widths = [patch.get_width() for patch in axis.containers[0].patches]

    assert len(widths) == len(budget.losses) + 2
    assert widths == sorted(widths, reverse=True)
    expected = sorted(
        [row.energy_j for row in budget.losses] + [budget.object_work_j, budget.stored_change_j],
        reverse=True,
    )
    assert widths == pytest.approx(expected, rel=1.0e-12)

    labels = [text.get_text() for text in axis.get_yticklabels()]
    assert "work delivered to the object" in labels
    assert LOSS_LABELS["motor_copper_loss"] in labels
    assert len(labels) == len(widths)
    assert f"{budget.battery_energy_j:.4f} J" in axis.get_title(loc="left")


def test_energy_breakdown_labels_the_row_whose_bar_cannot_be_seen() -> None:
    """The object row carries a marker and a label because its bar has no width.

    The finding the figure exists to show is that the work on the object is invisible next
    to the copper loss. A reader still has to be able to find that row and read its value,
    so the marker sits at the end of the bar and the label states the number.
    """
    budget = energy_budget(_short_trace("budget", LARGE_CYLINDER))
    figure = published_energy_breakdown_figure(budget)
    axis = figure.axes[0]

    markers = [line for line in axis.lines if line.get_marker() == "o"]
    assert len(markers) == 1
    assert float(markers[0].get_xdata()[0]) == pytest.approx(budget.object_work_j, rel=1.0e-12)

    texts = [text.get_text() for text in axis.texts]
    assert f"{budget.object_work_j:.4f} J" in " ".join(texts)
    assert len(texts) == len(budget.losses) + 2


def test_published_postures_share_one_window_so_the_panels_compare() -> None:
    """Three panels, one scale, one polyline per finger, drawn to the same limits.

    Without shared limits the 40 mm cylinder would be drawn the same size as the 60 mm one
    and the figure would misreport the measurement it exists to show.
    """
    traces = [_short_trace(name, obstacle) for name, obstacle in TEST_OBJECTS]
    figure = published_grasp_posture_figure(traces)
    assert len(figure.axes) == len(traces)

    first = figure.axes[0]
    for axis, trace in zip(figure.axes, traces, strict=True):
        assert axis.get_xlim() == first.get_xlim()
        assert axis.get_ylim() == first.get_ylim()
        assert axis.get_aspect() == 1.0
        assert axis.get_title() == trace.name

        angles = tuple(float(a) for a in trace.joint_angles_rad[-1])
        origins, _, _ = joint_origins(trace.params.finger, angles)
        drawn = [line for line in axis.lines if len(line.get_xdata()) == len(origins)]
        assert len(drawn) == 1
        assert list(drawn[0].get_xdata()) == pytest.approx([1000.0 * point[0] for point in origins])
        assert list(drawn[0].get_ydata()) == pytest.approx([1000.0 * point[1] for point in origins])


def test_tendon_tension_figure_marks_both_quasi_static_references() -> None:
    """Two measured curves and two dashed chain values, at the tensions the chain gives.

    The gap between the measured settled tension and the dashed line is the result the
    figure is placed next to in the README, so the dashed lines have to be the chain
    values and not round numbers near them.
    """
    trace = _short_trace("tension", LARGE_CYLINDER)
    chain = force_chain(trace.params)
    figure = published_tendon_tension_figure(trace, chain)
    axis = figure.axes[0]

    solid = [line for line in axis.lines if line.get_linestyle() == "-"]
    assert len(solid) == 2
    assert [line.get_label() for line in solid] == [
        "at the drive pulley",
        "at the finger, after the routing",
    ]
    assert list(solid[0].get_ydata()) == pytest.approx(list(trace.drive_tension_n))
    assert list(solid[1].get_ydata()) == pytest.approx(list(trace.finger_tension_n))

    dashed = [line for line in axis.lines if line.get_linestyle() != "-"]
    levels = sorted(float(line.get_ydata()[0]) for line in dashed)
    assert levels == pytest.approx(
        sorted((chain.drive_tension_n, chain.finger_tension_n)), rel=1.0e-12
    )
    assert axis.get_ylim()[0] == 0.0


def test_scratch_figures_still_plot_what_they_are_given() -> None:
    """The untracked builders used by the example scripts report the trace they are given.

    These three write into ``outputs`` rather than into ``docs/figures``, but they are the
    figures a reader gets when running an example, so they carry the same obligation to
    show the data they were handed.
    """
    trace = _short_trace("tension", LARGE_CYLINDER)
    closing = closing_figure(trace)
    assert len(closing.axes) == 3
    assert list(closing.axes[1].lines[0].get_ydata()) == pytest.approx(list(trace.current_a))

    budget = energy_budget(trace)
    losses = loss_figure(budget)
    widths = [patch.get_width() for patch in losses.axes[0].containers[0].patches]
    assert widths == pytest.approx(
        [row.energy_j for row in budget.losses] + [budget.object_work_j, budget.stored_change_j],
        rel=1.0e-12,
    )

    # A round object and a flat one, so both obstacle branches of the drawing are used.
    postures = grasp_posture_figure([trace, _short_trace("flat plate", FLAT_PLATE)])
    assert len(postures.axes) == 2
    assert postures.axes[0].get_ylabel() == "y, mm"
    assert postures.axes[1].get_ylabel() == ""


def test_sensitivity_figure_plots_each_sweep_against_its_own_parameter() -> None:
    """Four panels, the swept values on the x axis of each, no axis shared between sweeps.

    The two sweeps have different units, so putting them on one pair of axes would be a
    dual scale. The figure keeps them apart and the test asserts which values landed where.
    """
    stiffness = _sweep("tendon_stiffness_n_per_m", (5.0e3, 2.0e4, 2.0e5))
    friction = _sweep("friction_coefficient", (0.0, 0.147, 0.20))
    figure = sensitivity_figure(stiffness, friction)
    assert len(figure.axes) == 4

    for index, sweep in ((0, stiffness), (1, stiffness), (2, friction), (3, friction)):
        axis = figure.axes[index]
        assert list(axis.lines[0].get_xdata()) == pytest.approx([p.value for p in sweep])
    assert figure.axes[0].get_xscale() == "log"
    assert figure.axes[2].get_xscale() == "linear"
