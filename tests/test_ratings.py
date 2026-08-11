"""Tests of the catalogue rating audit: what a run demands against what is published.

The ratings were already carried by the parameter objects and nothing read them. These
tests assert that each is now measured against a run, that the measurement is a duration
and not only a peak, and that the two exceedances the reference configuration produces are
exactly the two a quasi static sizing calculation cannot see.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from transradial_sim.analysis.ratings import rating_audit
from transradial_sim.model.system import ACCUMULATOR_NAMES, initial_state, stored_energy
from transradial_sim.model.units import rad_s_to_rpm
from transradial_sim.pipeline.scenario import (
    LARGE_CYLINDER,
    STIFF_CONTACT,
    build_plant,
    closing_scenario,
    current_controller,
    grasp_controller,
)
from transradial_sim.pipeline.simulate import ScenarioConfig, run_scenario
from transradial_sim.pipeline.trace import SimulationTrace

GRASP_DURATION_S = 1.30
SQUEEZE_START_S = 1.00
"""The squeeze starts once the finger has closed and the run ends 0.30 s later.

Shorter than the 1.70 s the README reports, because a torque rating here is crossed on the
impact and held from there rather than approached slowly: the peak output torque of this
run is within a quarter of a percent of the longer one.
"""

CLOSING_DURATION_S = 0.20
"""Free closing runs long enough to contain the speed peak, which arrives at 0.096 s."""


@cache
def _closing() -> SimulationTrace:
    config = closing_scenario(duration_s=CLOSING_DURATION_S, step_s=5.0e-5)
    return run_scenario(config, current_controller(config.params))


@cache
def _grasp() -> SimulationTrace:
    params = build_plant(obstacle=LARGE_CYLINDER, contact=STIFF_CONTACT)
    config = ScenarioConfig(
        name="rigid grasp",
        params=params,
        duration_s=GRASP_DURATION_S,
        step_s=5.0e-5,
        sample_stride=10,
    )
    return run_scenario(config, grasp_controller(params, squeeze_start_s=SQUEEZE_START_S))


def _written_trace(
    times: tuple[float, ...],
    motor_speed_rad_s: tuple[float, ...],
    current_a: tuple[float, ...],
) -> SimulationTrace:
    """Return a trace whose speed and current histories are written down by hand.

    Placing a threshold crossing is arithmetic on the two samples that bracket it, so it is
    checked against a signal chosen for its crossings rather than against a simulation,
    where the crossing time would itself have to be measured before it could be asserted.
    """
    params = build_plant()
    count = len(times)
    joints = params.finger.joint_count
    zeros = np.zeros(count, dtype=np.float64)
    stored = stored_energy(params, initial_state(params))
    return SimulationTrace(
        name="written by hand",
        params=params,
        step_s=1.0e-4,
        control_period_s=1.0e-4,
        time_s=np.asarray(times, dtype=np.float64),
        current_a=np.asarray(current_a, dtype=np.float64),
        motor_angle_rad=zeros,
        motor_speed_rad_s=np.asarray(motor_speed_rad_s, dtype=np.float64),
        joint_angles_rad=np.zeros((count, joints), dtype=np.float64),
        joint_rates_rad_s=np.zeros((count, joints), dtype=np.float64),
        duty=zeros,
        applied_voltage_v=zeros,
        battery_current_a=zeros,
        battery_power_w=zeros,
        motor_torque_nm=zeros,
        output_torque_nm=zeros,
        drive_tension_n=zeros,
        finger_tension_n=zeros,
        tendon_extension_m=zeros,
        contact_force_n=zeros,
        fingertip_force_n=zeros,
        contact_count=zeros,
        tip_position_m=np.zeros((count, 2), dtype=np.float64),
        accumulators_j=np.zeros((count, len(ACCUMULATOR_NAMES)), dtype=np.float64),
        initial_stored=stored,
        final_stored=stored,
        final_state=tuple(initial_state(params)),
    )


def test_every_rating_the_components_publish_is_checked_against_the_run() -> None:
    """Each check names its part and carries the catalogue number, not a copy of it.

    The four ratings are the two the reference configuration is sized against, the motor
    continuous current and the gearbox continuous output torque, and the two nothing in the
    project read at all before this, the intermittently permissible torque and the
    recommended input speed.
    """
    audit = rating_audit(_grasp())
    params = _grasp().params
    assert tuple(check.name for check in audit.checks) == (
        "motor continuous current",
        "gearbox continuous output torque",
        "gearbox intermittent output torque",
        "gearbox input speed",
    )
    assert audit.duration_s == pytest.approx(GRASP_DURATION_S, abs=_grasp().control_period_s)

    expected = (
        params.motor.catalogue.max_continuous_current_a,
        params.gearbox.max_continuous_output_torque_nm,
        params.gearbox.max_intermittent_output_torque_nm,
        params.gearbox.max_input_speed_rad_s,
    )
    for check, rating in zip(audit.checks, expected, strict=True):
        assert check.rating == rating, check.name
        assert check.peak >= 0.0, check.name
        assert 0.0 <= check.share <= 1.0, check.name
        assert check.margin == pytest.approx(check.peak / check.rating, rel=1.0e-12)
    assert audit.check("gearbox input speed").component == params.gearbox.part_number
    assert audit.check("motor continuous current").component == (params.motor.catalogue.part_number)
    assert tuple(check.unit for check in audit.checks) == ("A", "Nm", "Nm", "rad/s")


def test_free_closing_runs_the_gearhead_above_its_recommended_input_speed() -> None:
    """The one rating the free closing run breaks is the gearbox input speed.

    Free closing is speed limited rather than force limited, so the drive spends it near
    the motor's back electromotive force limit, which is above the 8000 rpm the gearhead
    catalogue recommends. The motor is happy there and the gearhead is not, and nothing in
    the sizing calculation looks at speed at all.
    """
    audit = rating_audit(_closing())
    speed = audit.check("gearbox input speed")
    assert speed.exceeded
    assert rad_s_to_rpm(speed.rating) == pytest.approx(8000.0, rel=1.0e-12)
    assert rad_s_to_rpm(speed.peak) > 8000.0
    assert speed.margin > 1.0
    # The peak arrives at 0.096 s and the drive stays there, so the run is above the
    # rating for most of its length rather than for one sample of it.
    assert speed.time_above_s > 0.5 * audit.duration_s
    assert audit.exceedances == (speed,)
    assert not audit.within_ratings
    assert not audit.check("motor continuous current").exceeded


def test_the_settled_grasp_sits_between_the_two_gearbox_torque_ratings() -> None:
    """The grasp asks the gearhead for more than the torque that set the current limit.

    The current limit is the current at which the gearhead reaches its continuous output
    torque, so the quasi static chain lands exactly on that rating by construction. The run
    does not: the finger arrives carrying rotor kinetic energy, the impact stretches the
    cord, and neither the cord nor the stuck gearbox can push the stretch back out, so the
    settled output torque stays above the continuous rating. It stays below the
    intermittently permissible one, which is what makes the reference grasp a legitimate
    duty rather than an overload.
    """
    audit = rating_audit(_grasp())
    continuous = audit.check("gearbox continuous output torque")
    intermittent = audit.check("gearbox intermittent output torque")
    assert continuous.exceeded
    assert not intermittent.exceeded
    assert continuous.peak == intermittent.peak
    assert continuous.rating < intermittent.rating
    assert continuous.time_above_s > 0.0
    assert intermittent.time_above_s == 0.0
    assert audit.exceedances == (continuous,)

    # The motor is inside its own rating throughout, which is the statement the README
    # makes when it says the gearbox and not the motor sets the usable current.
    current = audit.check("motor continuous current")
    assert not current.exceeded
    assert current.peak < current.rating
    assert audit.worst is continuous


def test_the_time_above_a_rating_is_interpolated_and_not_counted_in_samples() -> None:
    """A crossing is placed between the two samples that bracket it.

    The signal rises from zero to three times the rating over one second and falls back
    over the next, so it is above the rating for two thirds of each of those two seconds
    and for none of the third. Counting whole sample intervals would return one of the
    whole seconds instead, which for a rating crossed briefly is the whole of the answer.
    """
    rating = build_plant().gearbox.max_input_speed_rad_s
    trace = _written_trace(
        times=(0.0, 1.0, 2.0, 3.0),
        motor_speed_rad_s=(0.0, 3.0 * rating, 0.0, 0.0),
        current_a=(0.0, 0.0, 0.0, 0.0),
    )
    speed = rating_audit(trace).check("gearbox input speed")
    assert speed.peak == pytest.approx(3.0 * rating, rel=1.0e-12)
    assert speed.margin == pytest.approx(3.0, rel=1.0e-12)
    assert speed.time_above_s == pytest.approx(4.0 / 3.0, rel=1.0e-12)
    assert speed.share == pytest.approx(4.0 / 9.0, rel=1.0e-12)


def test_a_rating_the_run_never_reaches_reports_no_time_above_it() -> None:
    """A signal held below its rating spends no time above it and has a margin under one."""
    trace = _written_trace(
        times=(0.0, 1.0, 2.0),
        motor_speed_rad_s=(0.0, 0.0, 0.0),
        current_a=(0.0, 0.0, 0.0),
    )
    audit = rating_audit(trace)
    for check in audit.checks:
        assert check.time_above_s == 0.0, check.name
        assert check.share == 0.0, check.name
        assert check.margin < 1.0, check.name
        assert not check.exceeded, check.name
    assert audit.exceedances == ()
    assert audit.within_ratings


def test_the_exceedances_are_ordered_by_how_far_past_the_rating_the_run_went() -> None:
    """Two ratings broken at once are reported worst first, and the worst is the worst.

    Ordering by margin rather than by the order the checks are built in is what makes the
    first row of the report the one to act on.
    """
    plant = build_plant()
    speed_rating = plant.gearbox.max_input_speed_rad_s
    current_rating = plant.motor.catalogue.max_continuous_current_a
    trace = _written_trace(
        times=(0.0, 1.0, 2.0, 3.0),
        motor_speed_rad_s=(0.0, 3.0 * speed_rating, 0.0, 0.0),
        current_a=(2.0 * current_rating,) * 4,
    )
    audit = rating_audit(trace)
    assert tuple(check.name for check in audit.exceedances) == (
        "gearbox input speed",
        "motor continuous current",
    )
    assert audit.worst is audit.exceedances[0]
    current = audit.check("motor continuous current")
    assert current.margin == pytest.approx(2.0, rel=1.0e-12)
    assert current.time_above_s == pytest.approx(audit.duration_s, rel=1.0e-12)
    assert current.share == pytest.approx(1.0, rel=1.0e-12)


def test_an_unknown_check_name_is_rejected() -> None:
    """A mistyped name would otherwise return nothing and read as a rating that passed."""
    audit = rating_audit(_closing())
    with pytest.raises(KeyError, match="no such rating"):
        audit.check("no such rating")


def test_an_audit_needs_a_run_with_a_duration_to_measure_against() -> None:
    """A run of no length gives every share a zero denominator, so it is refused.

    The alternative is a division by zero raised from inside the audit, which says nothing
    about what the caller did wrong.
    """
    trace = _written_trace(
        times=(0.0,),
        motor_speed_rad_s=(0.0,),
        current_a=(0.0,),
    )
    with pytest.raises(ValueError, match="nonzero duration"):
        rating_audit(trace)
