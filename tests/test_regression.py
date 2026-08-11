"""Regression tier: a recorded reference run pinned with numeric tolerances.

What is pinned matters more than how tightly. Only quantities that are reproducible on
another machine are recorded here: closed form chain values, converged steady states, and
integrated energy balances. Raw late run state of a stiff contact simulation is not pinned,
because a difference in the order a linear algebra kernel reduces a sum grows over many
iterations into a visibly different answer, and a baseline recorded on one machine then
fails on another running the same code.

Each pinned value carries the tolerance appropriate to how it was produced, stated in the
docstring of the test that uses it.
"""

from __future__ import annotations

from functools import cache

import pytest

from transradial_sim.analysis.energetics import energy_budget, force_chain
from transradial_sim.analysis.metrics import closing_time_s, grasp_summary
from transradial_sim.pipeline.scenario import (
    LARGE_CYLINDER,
    STIFF_CONTACT,
    build_plant,
    current_controller,
    grasp_controller,
)
from transradial_sim.pipeline.simulate import ScenarioConfig, run_scenario
from transradial_sim.pipeline.trace import SimulationTrace

REFERENCE_STEP_S = 5.0e-5
REFERENCE_SQUEEZE_START_S = 1.00
REFERENCE_GRASP_DURATION_S = 1.70
REFERENCE_CLOSING_DURATION_S = 0.80
SAMPLE_STRIDE = 10

CHAIN_TOLERANCE = 1.0e-7
"""Closed form values involve no iteration, so only round off separates two machines.

The tolerance is set by the seven significant figures the literals below are written to,
not by the arithmetic, which is exact to fifteen.
"""

STEADY_STATE_TOLERANCE = 2.0e-3
"""Converged steady states are reproducible to the accuracy of the settling itself.

The recorded run was checked at three steps, 1e-4, 5e-5 and 2.5e-5. The settled tensions
and forces agreed to one part in a thousand and the posture to three decimal places in
degrees, so what is pinned is the fixed point of the model rather than the state of an
unconverged solve. Two tenths of a percent is twice that spread, which leaves room for a
difference in the order a summation is reduced on another machine without leaving room
for a change of behaviour.
"""


@cache
def _reference_grasp() -> SimulationTrace:
    params = build_plant(obstacle=LARGE_CYLINDER, contact=STIFF_CONTACT)
    config = ScenarioConfig(
        name="reference grasp",
        params=params,
        duration_s=REFERENCE_GRASP_DURATION_S,
        step_s=REFERENCE_STEP_S,
        sample_stride=SAMPLE_STRIDE,
    )
    return run_scenario(config, grasp_controller(params, squeeze_start_s=REFERENCE_SQUEEZE_START_S))


def test_reference_force_chain_values() -> None:
    """The quasi static chain is a closed form calculation and is pinned exactly.

    Nothing here is iterative, so the recorded values are reproducible to round off on any
    machine. A change to a catalogue parameter, a wrap angle or a pulley radius will move
    one of these and the test names which.
    """
    params = build_plant()
    chain = force_chain(params)
    assert chain.current_a == pytest.approx(1.1209757, rel=CHAIN_TOLERANCE)
    assert chain.drive_tension_n == pytest.approx(160.62780, rel=1.0e-6)
    assert chain.finger_tension_n == pytest.approx(96.02321, rel=1.0e-6)
    assert chain.transmission_efficiency == pytest.approx(0.348638, rel=1.0e-5)
    assert chain.stages[-1].efficiency == pytest.approx(0.597799, rel=1.0e-6)


def test_reference_grasp_steady_state() -> None:
    """The settled grasp of a rigid cylinder is pinned as a converged steady state.

    The drive has stalled and the contact forces have stopped changing, so these are the
    fixed point of the model rather than a snapshot of a transient. Tolerance is the
    steady state tolerance defined at the top of this module.

    These literals were recorded before the gearhead backlash was modelled and were not
    changed when it was. The design notes predicted that: the play is taken up once and
    stays taken up, so it moves the motor angle at which the grasp settles and not the
    force the grasp settles at. Leaving the numbers alone is the evidence for it.
    """
    trace = _reference_grasp()
    settled = grasp_summary(trace)
    assert settled.drive_tension_n == pytest.approx(207.62, rel=STEADY_STATE_TOLERANCE)
    assert settled.finger_tension_n == pytest.approx(124.12, rel=STEADY_STATE_TOLERANCE)
    assert settled.measured_capstan_ratio == pytest.approx(0.597799, rel=STEADY_STATE_TOLERANCE)
    assert settled.total_force_n == pytest.approx(38.289, rel=STEADY_STATE_TOLERANCE)
    assert settled.fingertip_force_n == pytest.approx(23.091, rel=STEADY_STATE_TOLERANCE)
    assert settled.contact_points == 2


def test_reference_grasp_posture() -> None:
    """The settled posture is pinned in degrees, to a tenth of a degree.

    A tenth of a degree corresponds to a fingertip displacement of about ninety micrometres,
    which is below the indentation of the rigid contact and therefore below the resolution
    at which the posture is defined.
    """
    settled = grasp_summary(_reference_grasp())
    expected = (18.68, 32.76, 80.04)
    for measured, reference in zip(settled.joint_angles_deg, expected, strict=True):
        assert measured == pytest.approx(reference, abs=0.1)


def test_reference_energy_budget() -> None:
    """The integrated energy budget of the reference grasp is pinned.

    Energy integrals are smooth functionals of the whole trajectory rather than samples of
    it, which makes them the most portable quantity the simulation produces. Tolerance is
    half a percent on each channel, and the balance residual is required to stay three
    orders of magnitude below the input, against a fourth order truncation bound at this
    step of five percent.
    """
    budget = energy_budget(_reference_grasp())
    assert budget.battery_energy_j == pytest.approx(3.9570, rel=5.0e-3)
    assert budget.loss("motor_copper_loss").energy_j == pytest.approx(2.1689, rel=5.0e-3)
    assert budget.loss("gearbox_loss").energy_j == pytest.approx(0.4858, rel=5.0e-3)
    assert budget.loss("capstan_loss").energy_j == pytest.approx(0.2820, rel=5.0e-3)
    assert budget.loss("driver_loss").energy_j == pytest.approx(0.4998, rel=5.0e-3)
    assert budget.object_work_j == pytest.approx(0.00722, rel=2.0e-2)
    assert budget.relative_residual < 1.0e-3


def test_reference_closing_time() -> None:
    """The free closing time is pinned to one sample interval.

    The metric is a threshold crossing on a sampled trace, so it is quantised by the sample
    interval. The tolerance is exactly that interval, which is the stride times the control
    period, and not a smaller number that would sit on its own quantisation boundary.

    The pinned value moved from 0.358 s to 0.360 s when the gearhead backlash was modelled,
    which is the two milliseconds the drive spends winding in the 0.223 mm of play before
    the cord begins to pull. That delay is what the design notes said backlash would add,
    and it is the only pinned number the change moved.
    """
    params = build_plant()
    config = ScenarioConfig(
        name="reference closing",
        params=params,
        duration_s=REFERENCE_CLOSING_DURATION_S,
        step_s=REFERENCE_STEP_S,
        sample_stride=SAMPLE_STRIDE,
    )
    trace = run_scenario(config, current_controller(params))
    sample_interval_s = SAMPLE_STRIDE * trace.control_period_s
    assert closing_time_s(trace) == pytest.approx(0.360, abs=sample_interval_s)
    assert float(trace.motor_speed_rad_s.max()) == pytest.approx(923.32, rel=1.0e-3)


def test_reference_verdicts_are_stable() -> None:
    """Discrete conclusions the README states are pinned as counts and orderings."""
    settled = grasp_summary(_reference_grasp())
    assert settled.phalanx_forces_n[2] > settled.phalanx_forces_n[0]
    assert settled.phalanx_forces_n[0] > settled.phalanx_forces_n[1]
    assert settled.finger_tension_n < settled.drive_tension_n
    params = build_plant()
    assert params.current_limit_a < params.motor.catalogue.max_continuous_current_a
    assert params.finger.joint_count == 3
    assert len(params.tendon.routing) == 4
