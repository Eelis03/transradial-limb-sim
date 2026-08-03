"""Energy conservation and energy balance closure.

These are the strongest correctness tests available for this simulation. Every power term
in the plant is written so that it is individually signed correctly, so the balance is an
identity that holds pointwise, and a nonzero residual can only come from integration error.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.conftest import (
    REFERENCE_STEP_S,
    lossless_natural_rate_rad_s,
    lossless_plant,
    oscillatory_bound,
    truncation_bound,
)
from transradial_sim.algorithm.controllers import ConstantDuty
from transradial_sim.model.finger import joint_origins, point_on_phalanx
from transradial_sim.model.system import (
    ACCUMULATOR_NAMES,
    IDX_MOTOR_SPEED,
    INTERNAL_LOSS_NAMES,
    energy_residual,
    evaluate_plant,
    initial_state,
    object_energy_residual,
    state_derivative,
    stored_energy,
)
from transradial_sim.pipeline.scenario import (
    LARGE_CYLINDER,
    STIFF_CONTACT,
    build_plant,
    grasp_controller,
)
from transradial_sim.pipeline.simulate import ScenarioConfig, run_scenario

LOSSLESS_DURATION_S = 0.04
BALANCE_DURATION_S = 0.25
INITIAL_SPEED_RAD_S = 400.0
"""Rotor speed the lossless run starts with, in rad/s."""


def test_energy_is_conserved_in_the_lossless_configuration() -> None:
    """With every dissipative term removed the total energy must not drift.

    The plant is started with the rotor spinning and the bridge switched off, so the only
    exchanges are between the armature inductance, the two inertias and the three springs.
    The tolerance is the accumulated truncation error of the fourth order scheme evaluated
    at the electromechanical resonance of the lossless plant, which is the fastest mode
    present.
    """
    params = lossless_plant()
    config = ScenarioConfig(
        name="lossless",
        params=params,
        duration_s=LOSSLESS_DURATION_S,
        step_s=REFERENCE_STEP_S,
        sample_stride=100,
        initial_motor_speed_rad_s=INITIAL_SPEED_RAD_S,
    )
    trace = run_scenario(config, ConstantDuty(duty=0.0, sample_period_s=1.0e-4))
    initial_total = trace.initial_stored.internal_total_j
    final_total = trace.final_stored.internal_total_j
    assert initial_total > 0.0

    bound = oscillatory_bound(
        lossless_natural_rate_rad_s(params), REFERENCE_STEP_S, LOSSLESS_DURATION_S
    )
    drift = abs(final_total - initial_total) / initial_total
    assert drift < bound, f"energy drift {drift:.3e} exceeds the bound {bound:.3e}"


def test_lossless_plant_records_no_dissipation() -> None:
    """Every loss channel stays at zero when every loss coefficient is zero."""
    params = lossless_plant()
    state = initial_state(params)
    state[IDX_MOTOR_SPEED] = 400.0
    derivative = evaluate_plant(params, state, 0.0).derivative
    offset = params.accumulator_offset
    for index, name in enumerate(ACCUMULATOR_NAMES):
        if name in INTERNAL_LOSS_NAMES:
            assert derivative[offset + index] == pytest.approx(0.0, abs=1.0e-18), name


def test_energy_balance_closes_with_every_loss_enabled() -> None:
    """With friction present the balance closes once the dissipated terms are counted.

    The residual is ``battery energy - internal losses - work on the object - change in
    stored energy``. Its tolerance is the accumulated fourth order truncation error at the
    armature rate, which is the fastest pole of the reference plant.
    """
    params = build_plant(obstacle=LARGE_CYLINDER, contact=STIFF_CONTACT)
    config = ScenarioConfig(
        name="balance",
        params=params,
        duration_s=BALANCE_DURATION_S,
        step_s=REFERENCE_STEP_S,
        sample_stride=100,
    )
    trace = run_scenario(config, grasp_controller(params, squeeze_start_s=0.1))
    zero = [0.0] * params.state_size
    residual = energy_residual(params, zero, list(trace.final_state))
    supplied = trace.final_accumulator("battery_energy_j")
    assert supplied > 0.0

    armature_rate = params.motor.resistance_ohm / params.motor.inductance_h
    bound = truncation_bound(armature_rate, REFERENCE_STEP_S)
    relative = abs(residual) / supplied
    assert relative < bound, f"balance residual {relative:.3e} exceeds the bound {bound:.3e}"


def test_object_energy_splits_exactly_into_stored_and_dissipated() -> None:
    """Work delivered to the object equals its stored elastic energy plus its dissipation.

    Checked separately from the system balance so that a fault in the contact law cannot
    hide inside the rest of the model. The tolerance is the same fourth order bound.
    """
    params = build_plant(obstacle=LARGE_CYLINDER, contact=STIFF_CONTACT)
    config = ScenarioConfig(
        name="object balance",
        params=params,
        duration_s=BALANCE_DURATION_S,
        step_s=REFERENCE_STEP_S,
        sample_stride=100,
    )
    trace = run_scenario(config, grasp_controller(params, squeeze_start_s=0.1))
    zero = [0.0] * params.state_size
    residual = object_energy_residual(params, zero, list(trace.final_state))
    work = trace.final_accumulator("object_work_j")
    assert work > 0.0
    armature_rate = params.motor.resistance_ohm / params.motor.inductance_h
    bound = truncation_bound(armature_rate, REFERENCE_STEP_S)
    assert abs(residual) / work < bound


def test_every_loss_channel_is_monotonically_non_decreasing() -> None:
    """A loss integral can only grow. A term that falls is energy created from nothing."""
    params = build_plant(obstacle=LARGE_CYLINDER, contact=STIFF_CONTACT)
    config = ScenarioConfig(
        name="monotone losses",
        params=params,
        duration_s=BALANCE_DURATION_S,
        step_s=REFERENCE_STEP_S,
        sample_stride=20,
    )
    trace = run_scenario(config, grasp_controller(params, squeeze_start_s=0.1))
    for name in INTERNAL_LOSS_NAMES:
        channel = trace.accumulator(name)
        differences = channel[1:] - channel[:-1]
        assert differences.min() >= -1.0e-12, name


def test_stored_energy_terms_are_non_negative() -> None:
    """Every storage element holds a non negative amount of energy."""
    params = build_plant(obstacle=LARGE_CYLINDER, contact=STIFF_CONTACT)
    state = initial_state(params, (0.6, 0.7, 0.5))
    state[IDX_MOTOR_SPEED] = 300.0
    stored = stored_energy(params, state)
    assert stored.inductor_j >= 0.0
    assert stored.rotor_kinetic_j > 0.0
    assert stored.finger_kinetic_j >= 0.0
    assert stored.tendon_elastic_j >= 0.0
    assert stored.return_spring_j > 0.0
    assert stored.joint_limit_j >= 0.0
    assert stored.contact_elastic_j >= 0.0
    assert stored.total_j >= stored.internal_total_j


def test_initial_state_rejects_the_wrong_number_of_joints() -> None:
    """A posture with the wrong length is rejected rather than padded."""
    params = build_plant()
    with pytest.raises(ValueError, match="one entry per joint"):
        initial_state(params, (0.1, 0.2))


def test_the_gravitational_term_is_the_negative_potential_of_the_masses() -> None:
    """With gravity on, the stored energy carries the potential of every phalanx.

    Compared against the potential computed from the centres of mass directly, which is a
    second route to the same quantity, and asserted to vanish when the gravity vector is
    zero so that the default configuration reports no potential at all.
    """
    gravity = (0.0, -9.81)
    angles = (0.3, 0.5, 0.4)
    with_gravity = replace(build_plant(), gravity_m_per_s2=gravity)
    without = build_plant()

    state = initial_state(with_gravity, angles)
    origins, directions, _ = joint_origins(with_gravity.finger, angles)
    expected = 0.0
    for index, phalanx in enumerate(with_gravity.finger.phalanges):
        com = point_on_phalanx(origins, directions, index, phalanx.com_distance_m)
        expected -= phalanx.mass_kg * (gravity[0] * com[0] + gravity[1] * com[1])

    assert stored_energy(with_gravity, state).gravitational_j == pytest.approx(
        expected, rel=1.0e-12
    )
    assert expected > 0.0
    assert stored_energy(without, initial_state(without, angles)).gravitational_j == 0.0


def test_the_derivative_helper_returns_the_plant_derivative() -> None:
    """The integrator entry point and the diagnostic evaluation agree exactly.

    ``state_derivative`` exists so an integrator need not carry the diagnostics, and the
    two would drift apart silently if one were changed without the other.
    """
    params = build_plant(obstacle=LARGE_CYLINDER, contact=STIFF_CONTACT)
    state = initial_state(params, (0.4, 0.5, 0.6))
    state[IDX_MOTOR_SPEED] = 120.0
    assert state_derivative(params, state, 0.7) == evaluate_plant(
        params, state, 0.7
    ).derivative
