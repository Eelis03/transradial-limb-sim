"""Property and invariant tests for the finger kinematics, dynamics and contact model."""

from __future__ import annotations

import math

import pytest

from transradial_sim.model.contact import (
    CircularObject,
    ContactModel,
    FlatSurface,
    evaluate_contact,
    no_contact,
)
from transradial_sim.model.finger import (
    INDEX_FINGER,
    bias_torques,
    inverse_dynamics,
    joint_limit_torques,
    joint_origins,
    kinetic_energy,
    mass_matrix,
    point_on_phalanx,
    point_velocity,
    solve_accelerations,
    spring_energy,
    spring_torques,
    tendon_displacement,
)

FINGER = INDEX_FINGER
POSTURES = (
    (0.0, 0.0, 0.0),
    (0.3, 0.7, 0.2),
    (1.2, 1.5, 1.0),
    (-0.2, 0.4, -0.1),
)


def test_forward_kinematics_at_full_extension() -> None:
    """With every joint at zero the finger lies along the x axis."""
    origins, directions, absolute = joint_origins(FINGER, (0.0, 0.0, 0.0))
    assert origins[0] == (0.0, 0.0)
    assert origins[-1][0] == pytest.approx(FINGER.total_length_m)
    assert origins[-1][1] == pytest.approx(0.0)
    assert all(angle == 0.0 for angle in absolute)
    assert all(direction == pytest.approx((1.0, 0.0)) for direction in directions)


def test_absolute_angles_accumulate() -> None:
    """Each phalanx angle in the plane is the running sum of the joint angles."""
    angles = (0.3, 0.7, 0.2)
    _, _, absolute = joint_origins(FINGER, angles)
    assert absolute == pytest.approx((0.3, 1.0, 1.2))


def test_mass_matrix_agrees_with_recursive_newton_euler() -> None:
    """The closed form mass matrix reproduces the one implied by inverse dynamics.

    The two are computed by completely different routes, a Jacobian product and a
    recursive Newton-Euler sweep, so agreement is a strong internal check on the dynamics.
    The tolerance is double precision round off amplified by the condition of the matrix,
    taken here as 1e-12 relative.
    """
    for angles in POSTURES:
        origins, directions, _ = joint_origins(FINGER, angles)
        closed_form = mass_matrix(FINGER, origins, directions)
        zero = (0.0, 0.0, 0.0)
        for column in range(FINGER.joint_count):
            unit = tuple(1.0 if k == column else 0.0 for k in range(FINGER.joint_count))
            torques = inverse_dynamics(FINGER, angles, zero, unit)
            for row in range(FINGER.joint_count):
                assert torques[row] == pytest.approx(closed_form[row][column], rel=1.0e-12)


def test_mass_matrix_is_symmetric_and_positive_definite() -> None:
    """Symmetry and positive definiteness hold at every posture tested."""
    for angles in POSTURES:
        origins, directions, _ = joint_origins(FINGER, angles)
        matrix = mass_matrix(FINGER, origins, directions)
        for row in range(FINGER.joint_count):
            for column in range(FINGER.joint_count):
                assert matrix[row][column] == pytest.approx(matrix[column][row], rel=1.0e-14)
        solution = solve_accelerations(matrix, [1.0, 0.0, 0.0])
        assert solution[0] > 0.0


def test_solve_rejects_a_non_positive_definite_matrix() -> None:
    """A degenerate mass matrix is reported rather than producing a silent result."""
    with pytest.raises(ValueError, match="positive definite"):
        solve_accelerations([[0.0, 1.0], [1.0, 0.0]], [1.0, 1.0])


def test_kinetic_energy_is_non_negative_and_quadratic() -> None:
    """Kinetic energy vanishes at rest and scales with the square of the rates."""
    angles = (0.3, 0.7, 0.2)
    origins, directions, _ = joint_origins(FINGER, angles)
    matrix = mass_matrix(FINGER, origins, directions)
    assert kinetic_energy(matrix, (0.0, 0.0, 0.0)) == 0.0
    single = kinetic_energy(matrix, (1.0, -2.0, 0.5))
    doubled = kinetic_energy(matrix, (2.0, -4.0, 1.0))
    assert single > 0.0
    assert doubled == pytest.approx(4.0 * single, rel=1.0e-12)


def test_bias_torques_vanish_at_rest_without_gravity() -> None:
    """With no gravity and no motion there are no velocity product torques."""
    for angles in POSTURES:
        assert bias_torques(FINGER, angles, (0.0, 0.0, 0.0)) == pytest.approx(
            [0.0, 0.0, 0.0], abs=1.0e-18
        )


def test_gravity_torque_matches_the_potential_gradient() -> None:
    """The gravity term equals the gradient of the gravitational potential energy.

    Checked by central difference of the potential with a step of 1e-6 rad, so the
    tolerance follows from the second order difference error, ``O(step^2)`` relative.
    """
    gravity = (0.0, -9.81)
    angles = (0.4, 0.6, 0.3)
    torques = bias_torques(FINGER, angles, (0.0, 0.0, 0.0), gravity)

    def potential(state: tuple[float, ...]) -> float:
        origins, directions, _ = joint_origins(FINGER, state)
        total = 0.0
        for index, phalanx in enumerate(FINGER.phalanges):
            com = point_on_phalanx(origins, directions, index, phalanx.com_distance_m)
            total -= phalanx.mass_kg * (gravity[0] * com[0] + gravity[1] * com[1])
        return total

    step = 1.0e-6
    for joint in range(FINGER.joint_count):
        high = list(angles)
        low = list(angles)
        high[joint] += step
        low[joint] -= step
        gradient = (potential(tuple(high)) - potential(tuple(low))) / (2.0 * step)
        assert torques[joint] == pytest.approx(gradient, rel=1.0e-6, abs=1.0e-12)


def test_point_velocity_matches_finite_difference() -> None:
    """The planar point Jacobian reproduces a finite difference of the position."""
    angles = (0.4, 0.6, 0.3)
    rates = (1.5, -0.8, 2.0)
    step = 1.0e-7
    ahead = tuple(a + step * r for a, r in zip(angles, rates, strict=True))
    behind = tuple(a - step * r for a, r in zip(angles, rates, strict=True))
    origins, directions, _ = joint_origins(FINGER, angles)
    point = point_on_phalanx(origins, directions, 2, FINGER.phalanges[2].length_m)
    velocity = point_velocity(origins, point, 2, rates)

    origins_ahead, directions_ahead, _ = joint_origins(FINGER, ahead)
    origins_behind, directions_behind, _ = joint_origins(FINGER, behind)
    ahead_point = point_on_phalanx(
        origins_ahead, directions_ahead, 2, FINGER.phalanges[2].length_m
    )
    behind_point = point_on_phalanx(
        origins_behind, directions_behind, 2, FINGER.phalanges[2].length_m
    )
    for axis in (0, 1):
        expected = (ahead_point[axis] - behind_point[axis]) / (2.0 * step)
        assert velocity[axis] == pytest.approx(expected, rel=1.0e-6, abs=1.0e-12)


def test_tendon_displacement_is_the_moment_arm_weighted_sum() -> None:
    """One tendon couples the joints through a single scalar length."""
    angles = (0.5, 0.4, 0.3)
    expected = sum(
        phalanx.moment_arm_m * angle
        for phalanx, angle in zip(FINGER.phalanges, angles, strict=True)
    )
    assert tendon_displacement(FINGER, angles) == pytest.approx(expected, rel=1.0e-14)


def test_return_springs_oppose_flexion_and_store_energy() -> None:
    """Spring torque opposes flexion and its energy is the integral of that torque."""
    angles = (0.5, 0.4, 0.3)
    torques = spring_torques(FINGER, angles)
    assert all(torque < 0.0 for torque in torques)
    expected = sum(
        0.5 * phalanx.spring_stiffness_nm_per_rad * angle**2
        for phalanx, angle in zip(FINGER.phalanges, angles, strict=True)
    )
    assert spring_energy(FINGER, angles) == pytest.approx(expected, rel=1.0e-14)


def test_joint_limits_only_push_inwards() -> None:
    """An end stop never pulls the joint further out of range."""
    beyond = tuple(phalanx.upper_limit_rad + 0.05 for phalanx in FINGER.phalanges)
    for rate in (-50.0, -1.0, 0.0, 1.0, 50.0):
        torques, stored = joint_limit_torques(FINGER, beyond, (rate, rate, rate))
        assert all(torque <= 0.0 for torque in torques)
        assert stored > 0.0

    below = tuple(phalanx.lower_limit_rad - 0.05 for phalanx in FINGER.phalanges)
    for rate in (-50.0, 0.0, 50.0):
        torques, stored = joint_limit_torques(FINGER, below, (rate, rate, rate))
        assert all(torque >= 0.0 for torque in torques)
        assert stored > 0.0


def test_no_joint_limit_torque_inside_the_range() -> None:
    """Inside its range a joint feels nothing from its stops."""
    torques, stored = joint_limit_torques(FINGER, (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
    assert torques == [0.0, 0.0, 0.0]
    assert stored == 0.0


def test_contact_force_is_never_attractive() -> None:
    """An object pushes the finger and never pulls it, whatever the approach rate."""
    model = ContactModel(
        stiffness_n_per_m_pow=4.0e4, exponent=1.5, damping_s_per_m=8.0, samples_per_phalanx=3
    )
    surface = FlatSurface(point_m=(0.0, 0.010), normal=(0.0, -1.0))
    angles = (0.6, 0.6, 0.4)
    origins, directions, _ = joint_origins(FINGER, angles)
    for rate in (-50.0, -5.0, 0.0, 5.0, 50.0):
        result = evaluate_contact(
            FINGER, model, surface, origins, directions, (rate, rate, rate)
        )
        assert result.total_normal_force_n >= 0.0
        assert all(force >= 0.0 for force in result.phalanx_forces_n)
        assert result.dissipated_power_w >= -1.0e-15


def test_contact_torque_power_matches_the_work_on_the_object() -> None:
    """The power the contact removes from the finger equals the power into the object.

    This is the Jacobian transpose identity, and it is what makes the energy balance close
    once contact is active. The tolerance is double precision round off relative to the
    larger of the two quantities.
    """
    model = ContactModel(
        stiffness_n_per_m_pow=4.0e4, exponent=1.5, damping_s_per_m=8.0, samples_per_phalanx=3
    )
    obstacle = CircularObject(centre_m=(0.020, 0.038), radius_m=0.030)
    angles = (0.5, 0.6, 0.5)
    rates = (1.2, -0.4, 0.9)
    origins, directions, _ = joint_origins(FINGER, angles)
    result = evaluate_contact(FINGER, model, obstacle, origins, directions, rates)
    assert result.contact_count > 0
    delivered = sum(
        torque * rate
        for torque, rate in zip(result.joint_torques_nm, rates, strict=True)
    )
    assert delivered == pytest.approx(-result.power_into_object_w, rel=1.0e-12, abs=1.0e-15)


def test_contact_model_rejects_invalid_parameters() -> None:
    """A contact law needs at least one sample point and a positive exponent."""
    with pytest.raises(ValueError, match="samples_per_phalanx"):
        ContactModel(1.0, 1.5, 1.0, 0)
    with pytest.raises(ValueError, match="exponent"):
        ContactModel(1.0, 0.0, 1.0, 3)


def test_no_contact_is_a_zero_result() -> None:
    """The free motion result carries no force, no torque and no energy."""
    result = no_contact(3)
    assert result.total_normal_force_n == 0.0
    assert result.fingertip_force_n == 0.0
    assert result.joint_torques_nm == (0.0, 0.0, 0.0)


def test_circle_penetration_is_geometric() -> None:
    """Penetration into a disc is the radius less the distance from the centre."""
    disc = CircularObject(centre_m=(0.0, 0.0), radius_m=0.010)
    depth, normal = disc.penetration((0.004, 0.0))
    assert depth == pytest.approx(0.006)
    assert normal == pytest.approx((1.0, 0.0))
    depth, _ = disc.penetration((0.020, 0.0))
    assert depth < 0.0
    depth, normal = disc.penetration((0.0, 0.0))
    assert depth == pytest.approx(0.010)
    assert math.hypot(*normal) == pytest.approx(1.0)
