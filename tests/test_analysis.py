"""Tests of the analysis layer: efficiency bounds, adaptivity, and force scaling."""

from __future__ import annotations

import math
from functools import cache

import pytest

from tests.conftest import REFERENCE_STEP_S
from transradial_sim.analysis.energetics import (
    energy_budget,
    force_chain,
    frictionless_tension_n,
)
from transradial_sim.analysis.metrics import closing_time_s, grasp_summary
from transradial_sim.analysis.sensitivity import relative_spread
from transradial_sim.model.contact import ContactGeometry
from transradial_sim.model.tendon import capstan_ratio
from transradial_sim.pipeline.scenario import (
    APPROACH_SPEED_LIMIT_RAD_S,
    FLAT_PLATE,
    LARGE_CYLINDER,
    SMALL_CYLINDER,
    STIFF_CONTACT,
    build_plant,
    current_controller,
    grasp_controller,
)
from transradial_sim.pipeline.simulate import ScenarioConfig, run_scenario
from transradial_sim.pipeline.trace import SimulationTrace

GRASP_DURATION_S = 1.70
GRASP_STEP_S = 5.0e-5
SQUEEZE_START_S = 1.00
"""Grasp runs settle by 1.7 s with the squeeze starting at 1.0 s.

At this duration the settled quantities agree to one part in ten thousand across
three step halvings, so the runs below report a converged fixed point rather than a
snapshot of a transient.
"""


@cache
def _grasp(
    name: str,
    obstacle: ContactGeometry,
    current_limit_a: float | None = None,
) -> SimulationTrace:
    params = build_plant(
        obstacle=obstacle,
        contact=STIFF_CONTACT,
        current_limit_a=current_limit_a,
    )
    config = ScenarioConfig(
        name=name,
        params=params,
        duration_s=GRASP_DURATION_S,
        step_s=GRASP_STEP_S,
        sample_stride=10,
    )
    return run_scenario(
        config, grasp_controller(params, squeeze_start_s=SQUEEZE_START_S)
    )


def test_every_stage_efficiency_lies_between_zero_and_one() -> None:
    """No stage of the chain may create effort, and none may destroy all of it."""
    params = build_plant()
    chain = force_chain(params)
    for stage in chain.stages:
        assert 0.0 < stage.efficiency <= 1.0, stage.name
    assert 0.0 < chain.transmission_efficiency < 1.0


def test_the_chain_is_monotonically_lossy_in_force() -> None:
    """The tension reaching the finger is below the tension applied at the drive."""
    params = build_plant()
    chain = force_chain(params)
    assert chain.finger_tension_n < chain.drive_tension_n
    assert chain.finger_tension_n > 0.0


def test_chain_force_is_proportional_to_current_above_the_friction_offset() -> None:
    """Doubling the current more than doubles the tension, by the Coulomb offset.

    The transmission is linear in current apart from the motor's own Coulomb friction,
    which is subtracted once. That makes the chain an affine function with a negative
    intercept, and the test asserts exactly that rather than pure proportionality.
    """
    params = build_plant()
    low = force_chain(params, 0.5 * params.current_limit_a)
    high = force_chain(params, params.current_limit_a)
    assert high.finger_tension_n > 2.0 * low.finger_tension_n

    slope = (high.finger_tension_n - low.finger_tension_n) / (
        high.current_a - low.current_a
    )
    predicted = (
        params.motor.torque_constant_nm_per_a
        * params.gearbox.efficiency
        * params.gearbox.ratio
        / params.tendon.drive_radius_m
        * capstan_ratio(params.tendon.friction_coefficient, params.tendon.wrap_angle_rad)
    )
    assert slope == pytest.approx(predicted, rel=1.0e-9)


def test_energy_budget_shares_are_bounded_and_sum_to_one() -> None:
    """Every share lies in the unit interval and the shares account for the input.

    The tolerance on the sum is the relative balance residual of the same run, which is
    itself reported, so the test asserts the accounting against its own stated accuracy.
    """
    trace = _grasp("60 mm cylinder", LARGE_CYLINDER)
    budget = energy_budget(trace)
    assert budget.battery_energy_j > 0.0
    for row in budget.losses:
        assert row.energy_j >= -1.0e-12, row.name
        assert -1.0e-9 <= row.share <= 1.0, row.name
    total = (
        sum(row.share for row in budget.losses)
        + budget.object_share
        + budget.stored_share
    )
    # Twice the reported residual, so the assertion cannot sit on its own boundary: the
    # shares miss unity by exactly the residual, and a tolerance equal to it would be a
    # coin toss on the last bit.
    assert total == pytest.approx(1.0, abs=2.0 * max(1.0e-9, budget.relative_residual))
    assert budget.relative_residual < 1.0e-3


def test_the_named_losses_cover_every_stage_of_the_chain() -> None:
    """Each physical loss mechanism has its own row and none of them is zero."""
    trace = _grasp("60 mm cylinder", LARGE_CYLINDER)
    budget = energy_budget(trace)
    for name in (
        "battery_internal_loss",
        "driver_loss",
        "motor_copper_loss",
        "motor_friction_loss",
        "gearbox_loss",
        "capstan_loss",
        "joint_damping_loss",
    ):
        assert budget.loss(name).energy_j > 0.0, name
    with pytest.raises(KeyError):
        budget.loss("no such loss")


def test_the_finger_conforms_differently_to_a_flat_and_a_round_object() -> None:
    """The settled posture and the force distribution both depend on the object shape.

    Nothing in the controller knows which object is present, so any difference between the
    runs comes from the mechanism. The thresholds are stated in physical units: ten degrees
    of posture difference is well above the one degree the end stops deflect under load, and
    one newton of force difference is well above the millinewton scale of the contact
    model's sampling error.
    """
    flat = grasp_summary(_grasp("flat plate", FLAT_PLATE))
    large = grasp_summary(_grasp("60 mm cylinder", LARGE_CYLINDER))
    small = grasp_summary(_grasp("40 mm cylinder", SMALL_CYLINDER))

    for summary in (flat, large, small):
        assert summary.contact_points > 0, summary.name
        assert summary.total_force_n > 1.0, summary.name

    for left, right in ((flat, large), (flat, small), (large, small)):
        posture = math.sqrt(
            sum(
                (a - b) ** 2
                for a, b in zip(left.joint_angles_deg, right.joint_angles_deg, strict=True)
            )
            / len(left.joint_angles_deg)
        )
        assert posture > 10.0, f"{left.name} against {right.name}: {posture:.1f} deg"

    assert large.phalanx_forces_n[2] > large.phalanx_forces_n[0]
    assert small.phalanx_forces_n[0] > small.phalanx_forces_n[2]
    assert flat.phalanx_forces_n[0] < 1.0


def test_one_command_produces_different_force_distributions() -> None:
    """One command produces three different distributions of the same tendon load.

    Underactuation means the actuator sets a single scalar. The test asserts both halves
    of that statement: the tendon tension agrees to within five percent across the three
    objects, and the share of the total force carried by each phalanx differs by at least
    fifteen percentage points between every pair of them. Comparing shares rather than
    absolute forces means the second half cannot be satisfied by the finger simply pulling
    harder on one object than another.
    """
    summaries = [
        grasp_summary(_grasp(name, obstacle))
        for name, obstacle in (
            ("flat plate", FLAT_PLATE),
            ("60 mm cylinder", LARGE_CYLINDER),
            ("40 mm cylinder", SMALL_CYLINDER),
        )
    ]
    tensions = [summary.finger_tension_n for summary in summaries]
    spread = (max(tensions) - min(tensions)) / (sum(tensions) / len(tensions))
    assert spread < 0.05, f"tendon tension spread {spread:.3f}"

    shares = [
        tuple(force / summary.total_force_n for force in summary.phalanx_forces_n)
        for summary in summaries
    ]
    for left in range(len(shares)):
        for right in range(left + 1, len(shares)):
            difference = max(
                abs(a - b)
                for a, b in zip(shares[left], shares[right], strict=True)
            )
            assert difference > 0.15, (
                f"{summaries[left].name} against {summaries[right].name}: "
                f"{difference:.3f}"
            )


def test_fingertip_force_scales_with_motor_current_as_the_transmission_predicts() -> None:
    """Measured tendon tension is bracketed by the transmission at every current.

    Two bounds follow from the transmission alone. The lower one is the quasi static
    chain, in which the gearbox and the routing lose their full sliding friction. The
    upper one is the same chain with both losses removed. A stalled drive in this model
    creeps rather than holding, so the sliding friction is only partly active and the
    measured tension sits inside that band. Both bounds are proportional to current above
    the Coulomb offset, so the test is a statement about scaling and not about a fitted
    number. Monotonicity of the grasp force in current is asserted alongside it.
    """
    params = build_plant()
    tensions = []
    forces = []
    for fraction in (0.5, 1.0):
        current = fraction * params.current_limit_a
        trace = _grasp(f"force at {fraction}", LARGE_CYLINDER, current_limit_a=current)
        settled = grasp_summary(trace)
        lower = force_chain(trace.params, current).finger_tension_n
        upper = frictionless_tension_n(trace.params, current)
        assert 0.0 < lower < upper
        assert lower <= settled.finger_tension_n <= upper, (
            f"tension {settled.finger_tension_n:.2f} outside [{lower:.2f}, {upper:.2f}]"
        )
        tensions.append(settled.finger_tension_n)
        forces.append(settled.total_force_n)
    assert tensions[1] >= tensions[0]
    assert forces[1] > forces[0]


def test_measured_tendon_tension_tracks_the_chain_prediction() -> None:
    """The simulated settled tension agrees with the hand calculation.

    The measured tension is never below the quasi static value and never more than one
    impact above it. The upper bound is derived, not observed: the rotor arrives at the
    object carrying ``0.5 * J * omega_limit^2`` of kinetic energy, and a tendon that
    absorbed all of it would stretch to ``sqrt(2 * k * E)``. A tendon cannot push that
    stretch back out and the gearbox sticks rather than back driving, so whatever fraction
    of the impact ends up in the cord stays there and adds to the tension the motor holds.
    """
    trace = _grasp("60 mm cylinder", LARGE_CYLINDER)
    settled = grasp_summary(trace)
    params = trace.params
    chain = force_chain(params)
    kinetic_j = 0.5 * params.rotor_inertia_kgm2 * APPROACH_SPEED_LIMIT_RAD_S**2
    impact_n = math.sqrt(2.0 * params.tendon.stiffness_n_per_m * kinetic_j)
    assert chain.finger_tension_n <= settled.finger_tension_n
    assert settled.finger_tension_n <= chain.finger_tension_n + impact_n
    assert chain.drive_tension_n <= settled.drive_tension_n
    assert settled.drive_tension_n <= chain.drive_tension_n + impact_n


def test_closing_time_is_measured_against_the_final_posture() -> None:
    """The closing time is the first sample reaching the stated fraction of travel."""
    params = build_plant()
    config = ScenarioConfig(
        name="closing",
        params=params,
        duration_s=0.50,
        step_s=REFERENCE_STEP_S,
        sample_stride=20,
    )
    trace = run_scenario(config, current_controller(params))
    time_s = closing_time_s(trace)
    assert 0.0 < time_s < 0.50
    assert closing_time_s(trace, 0.5) < time_s
    with pytest.raises(ValueError, match="fraction"):
        closing_time_s(trace, 0.0)


def test_relative_spread_rejects_an_empty_sweep() -> None:
    """A spread over no points is an error rather than zero."""
    with pytest.raises(ValueError, match="no points"):
        relative_spread((), "closing_time_s")
