"""Property and invariant tests for the brushed motor model.

The strongest checks available here are external: the model is built from catalogue
parameters and then asked to reproduce catalogue operating points it was not given.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from transradial_sim.model.gearbox import (
    MAXON_GP26B_84,
    current_limit_from_gearbox,
    output_speed,
)
from transradial_sim.model.motor import (
    CATALOGUE_SPEED_CONSTANT_RPM_PER_V,
    MAXON_RE25_118752,
    electromagnetic_torque,
    friction_from_no_load_point,
    friction_torque,
    holding_voltage,
    predicted_no_load_current,
    predicted_no_load_speed,
    predicted_stall_torque,
)
from transradial_sim.model.units import (
    gcm2_to_kgm2,
    kgm2_to_gcm2,
    mnm_per_a_to_nm_per_a,
    nm_to_mnm,
    rad_s_to_rpm,
    rpm_to_rad_s,
    smooth_sign,
)

MOTOR = MAXON_RE25_118752
CATALOGUE = MOTOR.catalogue

CATALOGUE_TOLERANCE = 0.03
"""Relative agreement required with a catalogue operating point.

maxon publishes the torque constant to three significant figures and the no load speed to
three, and the two are mutually consistent only to about two percent: the speed implied by
the listed torque constant and terminal resistance differs from the listed no load speed by
that much. Three percent is the smallest band that contains the catalogue's own internal
inconsistency, so it is what a model built from the listed constants can be held to.
"""


def test_speed_constant_matches_catalogue() -> None:
    """The torque constant and the back emf constant are the same number in SI units.

    Confusing mNm/A with Nm/A, or rpm/V with rad/(V s), is the most common unit error in
    motor modelling and it is invisible until a speed is computed. The catalogue publishes
    the speed constant independently of the torque constant, so comparing the reciprocal of
    the back emf constant against it is an external check rather than a restatement.
    Tolerance is the rounding of the published three figure value, 0.5 in 408.
    """
    predicted = rad_s_to_rpm(1.0 / MOTOR.back_emf_constant_v_s_per_rad)
    assert predicted == pytest.approx(CATALOGUE_SPEED_CONSTANT_RPM_PER_V, abs=0.5)
    assert MOTOR.back_emf_constant_v_s_per_rad == MOTOR.torque_constant_nm_per_a


def test_no_load_speed_matches_catalogue() -> None:
    """The steady state unloaded speed reproduces the catalogue no load speed."""
    predicted = predicted_no_load_speed(MOTOR, CATALOGUE.nominal_voltage_v)
    assert predicted == pytest.approx(CATALOGUE.no_load_speed_rad_s, rel=CATALOGUE_TOLERANCE)


def test_no_load_current_matches_catalogue() -> None:
    """The steady state unloaded current reproduces the catalogue no load current."""
    predicted = predicted_no_load_current(MOTOR, CATALOGUE.nominal_voltage_v)
    assert predicted == pytest.approx(CATALOGUE.no_load_current_a, rel=CATALOGUE_TOLERANCE)


def test_stall_torque_matches_catalogue() -> None:
    """The torque at zero speed reproduces the catalogue stall torque."""
    predicted = predicted_stall_torque(MOTOR, CATALOGUE.nominal_voltage_v)
    assert predicted == pytest.approx(CATALOGUE.stall_torque_nm, rel=CATALOGUE_TOLERANCE)


def test_stall_current_matches_catalogue() -> None:
    """Ohm's law on the armature reproduces the catalogue stall current."""
    predicted = CATALOGUE.nominal_voltage_v / MOTOR.resistance_ohm
    assert predicted == pytest.approx(CATALOGUE.stall_current_a, rel=CATALOGUE_TOLERANCE)


def test_friction_is_always_dissipative() -> None:
    """Friction torque carries the sign of the speed at every speed.

    If it did not, the friction power would be negative somewhere and the energy balance
    could close only by accident.
    """
    for speed in (-2000.0, -50.0, -1.0, -1.0e-9, 1.0e-9, 1.0, 50.0, 2000.0):
        assert friction_torque(MOTOR, speed) * speed > 0.0
    assert friction_torque(MOTOR, 0.0) == 0.0


def test_friction_split_reproduces_the_no_load_point() -> None:
    """The derived friction coefficients balance the catalogue no load torque exactly."""
    viscous, coulomb = friction_from_no_load_point(
        torque_constant_nm_per_a=MOTOR.torque_constant_nm_per_a,
        no_load_current_a=CATALOGUE.no_load_current_a,
        no_load_speed_rad_s=CATALOGUE.no_load_speed_rad_s,
        coulomb_share=0.35,
    )
    total = viscous * CATALOGUE.no_load_speed_rad_s + coulomb
    expected = MOTOR.torque_constant_nm_per_a * CATALOGUE.no_load_current_a
    assert total == pytest.approx(expected, rel=1.0e-12)


def test_friction_split_rejects_invalid_shares() -> None:
    """A share outside the unit interval is rejected rather than silently clamped."""
    with pytest.raises(ValueError, match="coulomb_share"):
        friction_from_no_load_point(0.02, 0.03, 1000.0, 1.5)
    with pytest.raises(ValueError, match="no_load_speed"):
        friction_from_no_load_point(0.02, 0.03, 0.0, 0.5)


def test_holding_voltage_zeroes_the_current_derivative() -> None:
    """The holding voltage is exactly the value at which di/dt vanishes."""
    from transradial_sim.model.motor import current_derivative

    for current, speed in ((1.0, 100.0), (-0.5, -300.0), (0.0, 0.0)):
        voltage = holding_voltage(MOTOR, current, speed)
        assert current_derivative(MOTOR, voltage, current, speed) == pytest.approx(0.0, abs=1.0e-9)


def test_electromagnetic_torque_is_linear_in_current() -> None:
    """Air gap torque is proportional to current with the catalogue torque constant."""
    assert electromagnetic_torque(MOTOR, 2.0) == pytest.approx(2.0 * MOTOR.torque_constant_nm_per_a)
    assert electromagnetic_torque(MOTOR, -1.0) == pytest.approx(-MOTOR.torque_constant_nm_per_a)


def test_gearbox_sets_the_usable_current_not_the_motor() -> None:
    """The current at the gearbox torque rating is below the motor's continuous rating.

    This is a design conclusion rather than a modelling one, and it is asserted so that a
    change of gearbox or motor cannot silently invalidate the statement made in the README.
    """
    limit = current_limit_from_gearbox(MAXON_GP26B_84, MOTOR.torque_constant_nm_per_a)
    assert limit < CATALOGUE.max_continuous_current_a
    assert limit == pytest.approx(1.121, abs=0.001)


def test_gearbox_loss_uses_the_catalogue_efficiency_convention() -> None:
    """In steady motion the transmitted torque is exactly ``eta * N`` times motor torque.

    The coefficient of the loss term is ``1 / eta - 1``. Writing it as ``1 - eta`` would
    pass a smoke test and understate the loss by the factor ``eta``, so the relation is
    asserted directly against the definition of efficiency.
    """
    from transradial_sim.model.gearbox import loss_torque_on_motor, reflected_load_torque

    gearbox = MAXON_GP26B_84
    output_torque = 1.30
    speed = 500.0
    motor_torque = reflected_load_torque(gearbox, output_torque) + loss_torque_on_motor(
        gearbox, output_torque, speed
    )
    assert output_torque == pytest.approx(
        gearbox.efficiency * gearbox.ratio * motor_torque, rel=1.0e-9
    )


def test_gearbox_loss_is_dissipative_in_both_directions() -> None:
    """The gear loss power is non negative whichever way the shaft turns."""
    from transradial_sim.model.gearbox import loss_torque_on_motor

    for speed in (-800.0, -1.0, 1.0, 800.0):
        assert loss_torque_on_motor(MAXON_GP26B_84, 1.3, speed) * speed > 0.0


def test_gearbox_rejects_impossible_parameters() -> None:
    """Ratios and efficiencies outside their physical range are rejected."""
    with pytest.raises(ValueError, match="ratio"):
        replace(MAXON_GP26B_84, ratio=0.0)
    with pytest.raises(ValueError, match="efficiency"):
        replace(MAXON_GP26B_84, efficiency=1.5)


def test_electrical_time_constant_is_finite_and_small() -> None:
    """The armature pole is fast enough to bound the integration step meaningfully."""
    assert MOTOR.electrical_time_constant_s == pytest.approx(
        MOTOR.inductance_h / MOTOR.resistance_ohm
    )
    assert 1.0e-5 < MOTOR.electrical_time_constant_s < 1.0e-3
    assert not math.isnan(MOTOR.electrical_time_constant_s)


def test_the_output_turns_at_the_input_speed_divided_by_the_ratio() -> None:
    """The reduction is kinematic and exact, in both directions of rotation."""
    for speed in (-900.0, 0.0, 900.0):
        assert output_speed(MAXON_GP26B_84, speed) == pytest.approx(
            speed / MAXON_GP26B_84.ratio, rel=1.0e-12
        )
    assert output_speed(MAXON_GP26B_84, MAXON_GP26B_84.max_input_speed_rad_s) < 10.0


def test_the_catalogue_conversions_round_trip() -> None:
    """Each unit conversion inverts its partner, so no catalogue value can drift.

    The conversions are used once each, where catalogue data is entered, so the failure
    they guard against is a factor of a thousand entered in the wrong direction rather
    than an arithmetic error.
    """
    assert nm_to_mnm(MOTOR.torque_constant_nm_per_a) == pytest.approx(23.4, rel=1.0e-9)
    assert mnm_per_a_to_nm_per_a(nm_to_mnm(0.0234)) == pytest.approx(0.0234, rel=1.0e-12)
    assert kgm2_to_gcm2(MOTOR.rotor_inertia_kgm2) == pytest.approx(10.8, rel=1.0e-9)
    assert gcm2_to_kgm2(kgm2_to_gcm2(1.5e-6)) == pytest.approx(1.5e-6, rel=1.0e-12)
    assert rad_s_to_rpm(rpm_to_rad_s(9560.0)) == pytest.approx(9560.0, rel=1.0e-12)


def test_the_regularised_sign_needs_a_positive_scale() -> None:
    """A zero or negative blending scale has no meaning and is rejected.

    Without the guard the friction terms would divide by zero and return a silent NaN that
    propagates through the whole state vector.
    """
    assert smooth_sign(1.0, 1.0e-3) == pytest.approx(1.0, abs=1.0e-9)
    assert smooth_sign(-1.0, 1.0e-3) == pytest.approx(-1.0, abs=1.0e-9)
    assert smooth_sign(0.0, 1.0e-3) == 0.0
    with pytest.raises(ValueError, match="scale must be strictly positive"):
        smooth_sign(1.0, 0.0)
