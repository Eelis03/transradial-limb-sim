"""Convergence and order of the fixed step integrator.

The convergence study is run on the smooth part of the plant, with the tendon taut and no
object present. Contact activation and the one way tendon clamp are only piecewise smooth,
so an order study over them would measure the corner rather than the scheme.
"""

from __future__ import annotations

import math

import pytest

from tests.conftest import CONTROL_PERIOD_S, truncation_bound
from transradial_sim.algorithm.integrators import (
    ForwardEuler,
    RungeKutta4,
    integrate_fixed_step,
    richardson_order,
)
from transradial_sim.model.system import IDX_JOINTS, evaluate_plant
from transradial_sim.pipeline.scenario import build_plant, current_controller
from transradial_sim.pipeline.simulate import ScenarioConfig, run_scenario

CONVERGENCE_DURATION_S = 0.05
"""Length of the convergence runs, chosen to stay inside the smooth acceleration phase."""

COARSE_STEP_S = 5.0e-5
"""Coarsest step of the ladder. Halved twice by the tests below."""


def _final_flexion(step_s: float) -> float:
    params = build_plant()
    config = ScenarioConfig(
        name="convergence",
        params=params,
        duration_s=CONVERGENCE_DURATION_S,
        step_s=step_s,
        sample_stride=1000,
    )
    trace = run_scenario(config, current_controller(params))
    return float(trace.total_flexion_rad[-1])


def test_halving_the_step_changes_the_result_by_less_than_the_stated_tolerance() -> None:
    """Halving the step must move the answer by less than the fourth order bound.

    The tolerance is not an observed error. It is ``(lambda h)^4`` with ``lambda`` the
    fastest pole of the plant, the armature rate ``R / L``, and ``h`` the step. That is the
    global error of a fourth order scheme on a dissipative system, so the test asserts the
    scheme behaves as its order says it should.
    """
    params = build_plant()
    armature_rate = params.motor.resistance_ohm / params.motor.inductance_h
    coarse = _final_flexion(COARSE_STEP_S)
    medium = _final_flexion(COARSE_STEP_S / 2.0)
    bound = truncation_bound(armature_rate, COARSE_STEP_S)
    change = abs(coarse - medium) / abs(medium)
    assert change < bound, f"relative change {change:.3e} exceeds the bound {bound:.3e}"


def test_observed_order_is_high() -> None:
    """Three step halvings recover an order close to four.

    The observed order is measured by Richardson extrapolation on the same quantity. The
    threshold is two and a half rather than four because the plant is only piecewise
    smooth: the duty ratio is held constant across each control period, and the friction
    terms are regularised sign functions whose higher derivatives are large near zero
    speed. Both cost a fraction of an order, and the point of the test is to distinguish a
    high order scheme from a first or second order one, which this threshold does.
    """
    coarse = _final_flexion(COARSE_STEP_S)
    medium = _final_flexion(COARSE_STEP_S / 2.0)
    fine = _final_flexion(COARSE_STEP_S / 4.0)
    order = richardson_order(coarse, medium, fine)
    assert order >= 2.5, f"observed order {order:.2f}"


def test_richardson_order_rejects_identical_results() -> None:
    """An order cannot be estimated when the differences vanish."""
    with pytest.raises(ValueError, match="identical"):
        richardson_order(1.0, 1.0, 1.0)


def test_runge_kutta_is_exact_for_a_cubic() -> None:
    """The scheme integrates a cubic in time exactly, as a fourth order method must.

    A direct check of the scheme itself, independent of the plant. The tolerance is double
    precision round off on the accumulated sum.
    """

    def derivative(state: list[float], time_s: float) -> list[float]:
        del state
        return [3.0 * time_s * time_s]

    scheme = RungeKutta4()
    final = integrate_fixed_step(scheme, derivative, [0.0], 0.0, 100, 0.01)
    assert final[0] == pytest.approx(1.0, rel=1.0e-13)


def test_forward_euler_is_first_order_on_the_same_problem() -> None:
    """The reference first order scheme shows the expected order, which validates the study.

    If the order estimator itself were broken this test would not report one.
    """

    def derivative(state: list[float], time_s: float) -> list[float]:
        del time_s
        return [-state[0]]

    scheme = ForwardEuler()
    exact = math.exp(-1.0)
    errors = []
    for steps in (100, 200, 400):
        final = integrate_fixed_step(scheme, derivative, [1.0], 0.0, steps, 1.0 / steps)
        errors.append(abs(final[0] - exact))
    order = math.log2(errors[0] / errors[1])
    assert 0.9 < order < 1.1, f"observed order {order:.2f}"


def test_integrator_reports_its_own_order_and_stage_count() -> None:
    """The scheme metadata used by the tolerance derivations is correct."""
    assert RungeKutta4().order == 4
    assert RungeKutta4().stages == 4
    assert ForwardEuler().order == 1
    assert ForwardEuler().stages == 1


def test_control_period_must_be_a_multiple_of_the_step() -> None:
    """A step that does not divide the control period is rejected, not rounded."""
    params = build_plant()
    config = ScenarioConfig(
        name="bad step", params=params, duration_s=0.001, step_s=3.7e-5, sample_stride=1
    )
    with pytest.raises(ValueError, match="integer multiple"):
        run_scenario(config, current_controller(params))


def test_derivative_is_finite_everywhere_on_the_reference_plant() -> None:
    """The right hand side stays finite at rest, at speed and past the joint limits."""
    params = build_plant()
    state = [0.0] * params.state_size
    for duty in (-1.0, 0.0, 1.0):
        for joint_angle in (0.0, 1.0, 3.0):
            state[IDX_JOINTS] = joint_angle
            derivative = evaluate_plant(params, state, duty).derivative
            assert all(math.isfinite(value) for value in derivative)


def test_control_period_is_the_documented_value() -> None:
    """The suite and the scenarios agree on the controller rate."""
    params = build_plant()
    assert current_controller(params).sample_period_s == CONTROL_PERIOD_S
