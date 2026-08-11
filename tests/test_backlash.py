"""Tests of the gearhead play, the limitation these tests were written to close.

The model had no backlash. The catalogue publishes it, so leaving it out was a choice
rather than an unknown, and it is the one omission in the design notes that could be
closed without a parameter nobody measures. What follows is the evidence that it is now
modelled, that it is modelled as lost motion and not as a loss, and that it does what the
notes predicted it would do.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from transradial_sim.analysis.energetics import force_chain
from transradial_sim.model.gearbox import MAXON_GP26B_84, lost_motion_m
from transradial_sim.model.system import (
    IDX_MOTOR_ANGLE,
    SystemParameters,
    evaluate_plant,
    initial_state,
    stored_energy,
)
from transradial_sim.model.tendon import PROSTHETIC_TENDON, evaluate_tendon
from transradial_sim.pipeline.scenario import build_plant, current_controller
from transradial_sim.pipeline.simulate import ScenarioConfig, run_scenario

CATALOGUE_BACKLASH_DEG = 1.6
"""Average backlash at no load published for the maxon GP 26 B, in degrees."""


def _without_play(params: SystemParameters) -> SystemParameters:
    """Return the same plant with an ideal gearhead."""
    return replace(params, gearbox=replace(params.gearbox, backlash_rad=0.0))


def test_the_modelled_play_is_the_catalogue_value() -> None:
    """The dead band is the published backlash times the drive pulley radius.

    Nothing here is fitted. The angle is the catalogue number and the radius is the
    reference pulley, so the lost motion is a consequence of two published quantities.
    """
    assert MAXON_GP26B_84.backlash_rad == pytest.approx(
        math.radians(CATALOGUE_BACKLASH_DEG), rel=1.0e-12
    )
    params = build_plant()
    expected = math.radians(CATALOGUE_BACKLASH_DEG) * params.tendon.drive_radius_m
    assert params.lost_motion_m == pytest.approx(expected, rel=1.0e-12)
    assert lost_motion_m(MAXON_GP26B_84, 0.008) == pytest.approx(expected, rel=1.0e-12)
    # Two tenths of a millimetre of cord, which is the number the design notes now quote.
    assert 1.0e3 * params.lost_motion_m == pytest.approx(0.2234, abs=5.0e-4)


def test_a_negative_backlash_is_rejected() -> None:
    """Play is an angle a mechanism has or does not have. It cannot be negative."""
    with pytest.raises(ValueError, match="backlash_rad"):
        replace(MAXON_GP26B_84, backlash_rad=-1.0e-6)


def test_the_dead_band_shifts_the_tension_curve_and_does_not_scale_it() -> None:
    """Tension against extension is the ideal curve translated by the lost motion.

    This is the definition of lost motion as opposed to added compliance: the slope of the
    tension against extension is untouched, only its intercept moves. A model that
    softened the tendon instead would fail the slope comparison.
    """
    play = 2.2e-4
    for extension in (0.0, 1.0e-4, play, 5.0e-4, 2.0e-3, 8.0e-3):
        with_play = evaluate_tendon(PROSTHETIC_TENDON, extension, 0.0, 0.0, 0.0, 0.0, play)
        ideal = evaluate_tendon(PROSTHETIC_TENDON, extension - play, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert with_play.finger_tension_n == pytest.approx(ideal.finger_tension_n, rel=1.0e-12)
        assert with_play.stored_energy_j == pytest.approx(ideal.stored_energy_j, rel=1.0e-12)
        assert with_play.elastic_extension_m == pytest.approx(
            max(0.0, extension - play), abs=1.0e-15
        )


def test_nothing_is_transmitted_inside_the_dead_band() -> None:
    """Until the play is taken up the finger is disconnected from the drive.

    Swept over the whole band including its two edges, and at cord speeds in both
    directions so that the capstan branch is exercised while the tendon is disengaged.
    """
    play = 2.2e-4
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        for rate in (-0.05, 0.0, 0.05):
            state = evaluate_tendon(PROSTHETIC_TENDON, fraction * play, rate, 0.0, 0.0, 1.0, play)
            assert state.finger_tension_n == 0.0
            assert state.drive_tension_n == 0.0
            assert state.stored_energy_j == 0.0
            assert state.friction_power_w == 0.0
            assert state.damper_power_w == 0.0


def test_the_dead_band_dissipates_nothing() -> None:
    """The play is kinematic, so the series element energy identity is unchanged.

    ``T * d(extension)/dt`` must still equal the rate of change of the stored energy plus
    the damper loss, with the stored energy differenced numerically about the operating
    point. The tolerance is set by the difference step, not by an observed error.
    """
    play = 2.2e-4
    stiffness = PROSTHETIC_TENDON.stiffness_n_per_m
    delta = 1.0e-9
    for extension in (play + 1.0e-5, play + 1.0e-4, play + 1.0e-3):
        for rate in (-0.05, 0.0, 0.05):
            state = evaluate_tendon(PROSTHETIC_TENDON, extension, rate, 0.0, 0.0, 0.0, play)
            high = 0.5 * stiffness * (extension + delta - play) ** 2
            low = 0.5 * stiffness * max(0.0, extension - delta - play) ** 2
            stored_rate = (high - low) / (2.0 * delta) * rate
            supplied = state.finger_tension_n * state.extension_rate_m_per_s
            assert supplied == pytest.approx(
                stored_rate + state.damper_power_w, abs=1.0e-9 + 1.0e-6 * abs(supplied)
            )


def test_the_play_costs_travel_and_nothing_else() -> None:
    """The whole plant with play equals the ideal plant wound on by the lost motion.

    The claim being checked is that the backlash enters the model in exactly one place. If
    it had leaked into the gearbox friction, the stored energy or the load reflected onto
    the rotor, this comparison would fail on one of the three quantities asserted here,
    because the shift applied is a pure kinematic offset of the motor angle.
    """
    params = build_plant()
    ideal = _without_play(params)
    offset = params.lost_motion_m * params.gearbox.ratio / params.tendon.drive_radius_m

    angles = (0.20, 0.35, 0.90)
    base = initial_state(params, angles)
    for extra in (0.0, 5.0, 20.0, 60.0):
        with_play = list(base)
        with_play[IDX_MOTOR_ANGLE] += offset + extra
        without = list(base)
        without[IDX_MOTOR_ANGLE] += extra

        left = evaluate_plant(params, with_play, 0.5)
        right = evaluate_plant(ideal, without, 0.5)
        assert left.tendon.finger_tension_n == pytest.approx(
            right.tendon.finger_tension_n, rel=1.0e-12
        )
        assert left.output_torque_nm == pytest.approx(right.output_torque_nm, rel=1.0e-12)
        assert stored_energy(params, with_play).tendon_elastic_j == pytest.approx(
            stored_energy(ideal, without).tendon_elastic_j, rel=1.0e-12
        )


def test_the_play_does_not_move_the_quasi_static_chain() -> None:
    """The design notes predicted the settled grasp force would not change.

    The quasi static chain is a force balance and carries no displacement, so the play
    cannot appear in it. Asserting that directly is what makes the pinned settled grasp
    values in the regression tier, recorded before the play was modelled and unchanged
    after it, evidence rather than coincidence.
    """
    with_play = force_chain(build_plant())
    ideal = force_chain(_without_play(build_plant()))
    assert with_play.finger_tension_n == pytest.approx(ideal.finger_tension_n, rel=1.0e-15)
    assert with_play.drive_tension_n == pytest.approx(ideal.drive_tension_n, rel=1.0e-15)
    assert with_play.transmission_efficiency == pytest.approx(
        ideal.transmission_efficiency, rel=1.0e-15
    )


def test_a_run_shows_the_dead_band_in_its_trace() -> None:
    """On a real run the tendon carries load only once the recorded extension clears it.

    The run is deliberately short: the play is taken up in the first few milliseconds, so
    a fiftieth of a second contains both sides of the transition and nothing else is
    needed to see it.
    """
    params = build_plant()
    config = ScenarioConfig(
        name="engagement",
        params=params,
        duration_s=0.05,
        step_s=5.0e-5,
        sample_stride=1,
    )
    trace = run_scenario(config, current_controller(params))
    play = params.lost_motion_m

    disengaged = 0
    for extension, tension in zip(trace.tendon_extension_m, trace.finger_tension_n, strict=True):
        if float(extension) <= play:
            assert float(tension) == 0.0
            disengaged += 1
        else:
            assert float(tension) > 0.0
    assert disengaged > 0, "the run must contain the phase before the play is taken up"
    assert float(trace.tendon_extension_m[-1]) > play
