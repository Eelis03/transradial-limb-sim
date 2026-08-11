"""Property and invariant tests for the tendon transmission.

Two properties define a tendon and both are tested adversarially: it cannot push, and the
tension across a wrapped guide follows the capstan relation.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from transradial_sim.model.tendon import (
    PROSTHETIC_ROUTING,
    PROSTHETIC_TENDON,
    RoutingSegment,
    capstan_ratio,
    evaluate_tendon,
    total_wrap_angle,
)

TENDON = PROSTHETIC_TENDON


def test_wrap_angle_is_the_sum_of_the_segments() -> None:
    """The routing wrap angle is the sum of the individual guides."""
    assert TENDON.wrap_angle_rad == pytest.approx(3.50, abs=1.0e-12)
    assert total_wrap_angle(PROSTHETIC_ROUTING) == pytest.approx(3.50, abs=1.0e-12)


def test_capstan_ratio_matches_the_published_relation() -> None:
    """The tension ratio is ``exp(-mu theta)`` for a known wrap angle.

    Checked against the closed form of the Euler-Eytelwein relation at a half turn, where
    the published result for a coefficient of 0.147 is ``exp(-0.147 pi) = 0.63014``. The
    tolerance is the double precision round off of the exponential, not a fitted band.
    """
    half_turn = capstan_ratio(0.147, math.pi)
    assert half_turn == pytest.approx(math.exp(-0.147 * math.pi), rel=1.0e-15)
    assert half_turn == pytest.approx(0.63014, abs=1.0e-5)

    assert capstan_ratio(0.0, 5.0) == 1.0
    assert capstan_ratio(0.3, 0.0) == 1.0
    for wrap in (0.5, 1.0, 3.5, 6.0):
        assert 0.0 < capstan_ratio(0.147, wrap) <= 1.0


def test_capstan_ratio_is_multiplicative_over_the_path() -> None:
    """Splitting the routing into more guides with the same total angle changes nothing.

    This is the property that makes the wrap angle, rather than the number of guides, the
    parameter of the model.
    """
    combined = capstan_ratio(0.147, total_wrap_angle(PROSTHETIC_ROUTING))
    product = 1.0
    for segment in PROSTHETIC_ROUTING:
        product *= capstan_ratio(0.147, segment.wrap_angle_rad)
    assert combined == pytest.approx(product, rel=1.0e-14)


def test_reference_transmission_ratio() -> None:
    """The reference routing passes about 60 percent of the drive tension.

    Reported so that a change of routing geometry or friction coefficient cannot silently
    move the headline efficiency number quoted in the README.
    """
    ratio = capstan_ratio(TENDON.friction_coefficient, TENDON.wrap_angle_rad)
    assert ratio == pytest.approx(0.5978, abs=5.0e-4)


def test_tension_is_never_negative_against_an_adversarial_push() -> None:
    """A tendon pulls and cannot push, for any commanded compression or rate.

    The drive is asked to compress the cord by up to a metre and to do it at up to a metre
    per second in either direction. No combination may produce a negative tension at
    either end.
    """
    for compression in (0.0, 1.0e-9, 1.0e-4, 1.0e-2, 1.0):
        for rate in (-1.0, -1.0e-3, 0.0, 1.0e-3, 1.0):
            state = evaluate_tendon(
                TENDON,
                drive_displacement_m=-compression,
                drive_velocity_m_per_s=-rate,
                finger_displacement_m=0.0,
                finger_velocity_m_per_s=0.0,
            )
            assert state.finger_tension_n >= 0.0
            assert state.drive_tension_n >= 0.0
            assert state.extension_m <= 0.0


def test_a_slack_cord_transmits_nothing() -> None:
    """With the cord slack neither the spring nor the damper carries load."""
    state = evaluate_tendon(TENDON, -1.0e-3, 0.5, 0.0, 0.0)
    assert state.finger_tension_n == 0.0
    assert state.drive_tension_n == 0.0
    assert state.stored_energy_j == 0.0
    assert state.friction_power_w == 0.0


def test_loss_powers_are_never_negative() -> None:
    """Capstan friction and the series damper only ever remove energy."""
    for extension in (-1.0e-4, 0.0, 1.0e-5, 1.0e-3, 5.0e-3):
        for drive_rate in (-0.2, -1.0e-4, 0.0, 1.0e-4, 0.2):
            for finger_rate in (-0.2, 0.0, 0.2):
                state = evaluate_tendon(TENDON, extension, drive_rate, 0.0, finger_rate)
                assert state.friction_power_w >= -1.0e-18
                assert state.damper_power_w >= -1.0e-18


def test_series_element_energy_balance_is_exact() -> None:
    """``T * extension_rate`` equals the stored energy rate plus the damper loss.

    Verified by finite difference of the stored energy against the analytic identity, with
    the tolerance set by the difference step rather than by observation.
    """
    stiffness = TENDON.stiffness_n_per_m
    delta = 1.0e-9
    for extension in (1.0e-5, 1.0e-4, 1.0e-3):
        for rate in (-0.05, 0.0, 0.05):
            state = evaluate_tendon(TENDON, extension, rate, 0.0, 0.0)
            stored_high = 0.5 * stiffness * (extension + delta) ** 2
            stored_low = 0.5 * stiffness * max(0.0, extension - delta) ** 2
            stored_rate = (stored_high - stored_low) / (2.0 * delta) * rate
            supplied = state.finger_tension_n * state.extension_rate_m_per_s
            assert supplied == pytest.approx(
                stored_rate + state.damper_power_w, abs=1.0e-9 + 1.0e-6 * abs(supplied)
            )


def test_pulling_costs_more_than_paying_out() -> None:
    """The drive has to apply more tension when pulling in than when letting out.

    This asymmetry is the whole point of the capstan term, and its size is the published
    exponential factor. The cord speed is set well above the stick band so that the test
    exercises the sliding branch, where the direction comes from the measured speed.
    """
    speed = 10.0 * TENDON.stick_velocity_m_per_s
    pulling = evaluate_tendon(TENDON, 1.0e-3, speed, 0.0, 0.0)
    paying_out = evaluate_tendon(TENDON, 1.0e-3, -speed, 0.0, 0.0)
    assert pulling.drive_tension_n > pulling.finger_tension_n
    assert paying_out.drive_tension_n < paying_out.finger_tension_n
    ratio = capstan_ratio(TENDON.friction_coefficient, TENDON.wrap_angle_rad)
    assert pulling.finger_tension_n / pulling.drive_tension_n == pytest.approx(ratio, rel=1.0e-6)


def test_routing_rejects_a_negative_wrap_angle() -> None:
    """A guide cannot subtend a negative angle."""
    with pytest.raises(ValueError, match="wrap_angle_rad"):
        RoutingSegment(name="bad", wrap_angle_rad=-0.1)


def test_tendon_rejects_impossible_parameters() -> None:
    """Non physical radii, stiffnesses and friction coefficients are rejected."""
    with pytest.raises(ValueError, match="drive_radius_m"):
        replace(TENDON, drive_radius_m=0.0)
    with pytest.raises(ValueError, match="stiffness"):
        replace(TENDON, stiffness_n_per_m=-1.0)
    with pytest.raises(ValueError, match="friction_coefficient"):
        replace(TENDON, friction_coefficient=-0.01)


def test_stick_band_keeps_the_full_loss_when_the_cord_has_stopped() -> None:
    """A settled grasp keeps the capstan loss instead of losing it as the cord stops.

    Coulomb friction does not fall off with speed, so a routing that has stopped sliding
    still holds the tension difference it had when it stopped. Inside the stick band the
    direction therefore comes from the direction the drive is pushing rather than from the
    measured speed. Without this the tension ratio would go to one at stall and the model
    would report a grasp force free of any routing loss.
    """
    crawl = 0.1 * TENDON.stick_velocity_m_per_s
    settled = evaluate_tendon(TENDON, 5.0e-3, crawl, 0.0, 0.0, impending_direction=1.0)
    ratio = capstan_ratio(TENDON.friction_coefficient, TENDON.wrap_angle_rad)
    assert settled.finger_tension_n / settled.drive_tension_n == pytest.approx(ratio, rel=1.0e-9)
    assert settled.friction_power_w >= 0.0


def test_stick_band_stays_dissipative_when_the_cord_backs_up() -> None:
    """Inside the stick band a cord that momentarily reverses still dissipates energy."""
    crawl = -0.1 * TENDON.stick_velocity_m_per_s
    state = evaluate_tendon(TENDON, 5.0e-3, crawl, 0.0, 0.0, impending_direction=1.0)
    assert state.friction_power_w >= 0.0
