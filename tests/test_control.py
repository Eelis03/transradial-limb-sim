"""Controller tests, in particular that the current limit is never exceeded.

The limiter is exact in continuous time: at the trip level the applied voltage is reduced
to the value that makes ``di/dt`` zero. Any excursion above the limit in a simulated run is
therefore a discretisation artefact, and the tolerances below are derived from the step
rather than from an observed overshoot.
"""

from __future__ import annotations

import pytest

from tests.conftest import REFERENCE_STEP_S
from transradial_sim.algorithm.controllers import (
    ConstantDuty,
    CurrentController,
    ForceController,
    PlantMeasurement,
    PositionController,
    current_loop_gains,
)
from transradial_sim.pipeline.scenario import (
    COMPLIANT_CONTACT,
    FLAT_PLATE,
    build_plant,
    current_controller,
)
from transradial_sim.pipeline.simulate import ScenarioConfig, run_scenario

ADVERSARIAL_STEP_S = 1.0e-6
ADVERSARIAL_DURATION_S = 4.0e-3


def test_current_limit_holds_under_an_adversarial_open_loop_command() -> None:
    """Full duty applied open loop must not drive the current past the limit.

    This bypasses the controller entirely, so only the hardware style limiter inside the
    plant is protecting the motor. In continuous time the limiter is exact: at the trip
    level the applied voltage is reduced to the value that makes ``di/dt`` zero, so the
    current cannot cross. In discrete time a step that starts just below the trip level
    can carry the current above it, by at most ``h * V / L``, the largest change one step
    can produce at the maximum available ``di/dt``. The test asserts that bound at two
    steps, and because the bound is proportional to the step, passing at the finer step is
    the stronger statement of the two.
    """
    params = build_plant()
    previous_excess = None
    for step_s in (2.0 * ADVERSARIAL_STEP_S, ADVERSARIAL_STEP_S):
        config = ScenarioConfig(
            name="adversarial duty",
            params=params,
            duration_s=ADVERSARIAL_DURATION_S,
            step_s=step_s,
            sample_stride=1,
        )
        trace = run_scenario(config, ConstantDuty(duty=1.0, sample_period_s=1.0e-4))
        excess = float(trace.current_a.max()) - params.current_limit_a
        bound = step_s * params.supply.open_circuit_voltage_v / params.motor.inductance_h
        assert excess <= bound, f"excess {excess:.4e} exceeds bound {bound:.4e}"
        if previous_excess is not None:
            assert excess <= previous_excess
        previous_excess = excess


def test_current_limit_holds_under_a_reversed_adversarial_command() -> None:
    """The limiter is symmetric: full reverse duty is bounded the same way."""
    params = build_plant()
    config = ScenarioConfig(
        name="adversarial reverse duty",
        params=params,
        duration_s=ADVERSARIAL_DURATION_S,
        step_s=ADVERSARIAL_STEP_S,
        sample_stride=1,
    )
    trace = run_scenario(config, ConstantDuty(duty=-1.0, sample_period_s=1.0e-4))
    trough = float(trace.current_a.min())
    overshoot_bound = (
        ADVERSARIAL_STEP_S * params.supply.open_circuit_voltage_v / params.motor.inductance_h
    )
    assert trough >= -params.current_limit_a - overshoot_bound


def test_current_limit_holds_when_the_reference_is_commanded_far_beyond_it() -> None:
    """A reference ten times the limit is clamped and slew limited by the controller.

    With the reference clamped and slew limited the current cannot move faster than the
    slew rate, so one step can move it by at most ``h * S``. The tolerance is that product
    relative to the limit.
    """
    params = build_plant()
    controller = current_controller(params, reference_a=10.0 * params.current_limit_a)
    config = ScenarioConfig(
        name="over commanded reference",
        params=params,
        duration_s=0.10,
        step_s=REFERENCE_STEP_S,
        sample_stride=5,
    )
    trace = run_scenario(config, controller)
    bound = REFERENCE_STEP_S * controller.slew_limit_a_per_s
    assert float(trace.current_a.max()) <= params.current_limit_a + bound
    assert controller.applied_reference_a <= params.current_limit_a


def test_duty_never_leaves_the_bridge_range() -> None:
    """The applied duty ratio stays inside the physically available range."""
    params = build_plant()
    config = ScenarioConfig(
        name="duty range",
        params=params,
        duration_s=0.05,
        step_s=REFERENCE_STEP_S,
        sample_stride=1,
    )
    trace = run_scenario(config, ConstantDuty(duty=5.0, sample_period_s=1.0e-4))
    assert float(trace.duty.max()) <= params.supply.max_duty + 1.0e-12
    assert float(trace.duty.min()) >= -params.supply.max_duty - 1.0e-12


def test_current_loop_gains_follow_from_pole_placement() -> None:
    """The regulator gains are the motor parameters times the chosen bandwidth."""
    params = build_plant()
    proportional, integral = current_loop_gains(params.motor, 500.0)
    omega = 2.0 * 3.141592653589793 * 500.0
    assert proportional == pytest.approx(params.motor.inductance_h * omega, rel=1.0e-12)
    assert integral == pytest.approx(params.motor.resistance_ohm * omega, rel=1.0e-12)
    assert integral / proportional == pytest.approx(
        1.0 / params.motor.electrical_time_constant_s, rel=1.0e-12
    )


def test_current_loop_rejects_a_non_positive_bandwidth() -> None:
    """A bandwidth of zero or below has no pole placement interpretation."""
    params = build_plant()
    with pytest.raises(ValueError, match="bandwidth_hz"):
        current_loop_gains(params.motor, 0.0)


def test_controllers_reset_their_integrators() -> None:
    """Resetting clears accumulated state in the outer and the inner loop alike."""
    params = build_plant()
    inner = CurrentController(
        motor=params.motor,
        supply_voltage_v=params.supply.open_circuit_voltage_v,
        current_limit_a=params.current_limit_a,
    )
    outer = PositionController(inner=inner, proportional_a_per_rad=0.01, integral_a_per_rad_s=0.1)
    measurement = PlantMeasurement(0.0, 0.0, 0.0, 0.0, 0.0)
    outer.target_rad = 100.0
    for _ in range(20):
        outer.update(measurement)
    first = outer.update(measurement)
    outer.reset()
    after_reset = outer.update(measurement)
    assert after_reset != first
    assert inner.applied_reference_a == pytest.approx(
        inner.slew_limit_a_per_s * inner.sample_period_s
    )


def test_force_controller_reaches_its_set_point() -> None:
    """The outer force loop removes the steady state error against a compliant object."""
    params = build_plant(obstacle=FLAT_PLATE, contact=COMPLIANT_CONTACT)
    controller = ForceController(
        inner=current_controller(params, 0.0),
        proportional_a_per_n=0.0010,
        integral_a_per_n_s=0.050,
        target_n=8.0,
        integral_limit_a=params.current_limit_a,
    )
    config = ScenarioConfig(
        name="force step",
        params=params,
        duration_s=2.00,
        step_s=5.0e-5,
        sample_stride=20,
    )
    trace = run_scenario(config, controller)
    settled = float(trace.fingertip_force_n[-1])
    assert settled == pytest.approx(8.0, abs=2.0)
    assert float(trace.current_a.max()) <= params.current_limit_a * (1.0 + 1.0e-3)


def test_scheduled_controller_follows_its_schedule() -> None:
    """The two phase profile applies the approach current then the grasp current."""
    from transradial_sim.pipeline.scenario import APPROACH_CURRENT_A, grasp_controller

    params = build_plant()
    controller = grasp_controller(params, squeeze_start_s=0.2)
    controller.reset()
    controller.update(PlantMeasurement(0.0, 0.0, 0.0, 0.0, 0.0))
    assert controller.inner.reference_a == pytest.approx(APPROACH_CURRENT_A)
    controller.update(PlantMeasurement(0.5, 0.0, 0.0, 0.0, 0.0))
    assert controller.inner.reference_a == pytest.approx(params.current_limit_a)
