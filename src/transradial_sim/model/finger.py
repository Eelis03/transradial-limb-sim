"""Underactuated planar finger: geometry, rigid body dynamics, return springs, limits.

The finger is a planar serial chain of three phalanges driven by a single tendon that
runs over an idler at every joint. One actuator therefore drives three degrees of
freedom, which is the definition of underactuation. The joints are not kinematically
coupled: the tendon applies the same tension to every joint through its own moment arm,
and an extension spring at each joint opposes flexion. When one phalanx meets an object
and stops, the cord that continues to be paid in is taken up by the joints that are still
free, so the finger wraps around whatever shape it meets. Nothing in the code selects
that behaviour; it is a consequence of the force balance.

Rigid body dynamics use the standard planar recursive Newton-Euler formulation. The mass
matrix is available in closed form from the same kinematic quantities, and the two are
checked against each other in the test suite, which is a strong internal consistency
check on the dynamics.

Sign convention: joint angles are measured from full extension and increase with flexion.
Gravity is zero by default because a prosthetic finger is light enough that the tendon
and spring torques dominate, and because the reported metrics are then independent of
hand orientation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

__all__ = [
    "INDEX_FINGER",
    "FingerGeometry",
    "Phalanx",
    "bias_torques",
    "joint_limit_torques",
    "joint_origins",
    "kinetic_energy",
    "mass_matrix",
    "point_on_phalanx",
    "solve_accelerations",
    "spring_torques",
    "tendon_displacement",
]

Vector2 = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Phalanx:
    """One rigid segment of the finger together with the joint that drives it."""

    name: str
    """Anatomical name of the segment."""

    length_m: float
    """Distance from this joint to the next, in m."""

    mass_kg: float
    """Segment mass in kg."""

    com_distance_m: float
    """Distance from this joint to the segment centre of mass, in m."""

    inertia_kgm2: float
    """Segment inertia about its own centre of mass, in kg m^2."""

    moment_arm_m: float
    """Tendon moment arm about this joint, in m."""

    spring_stiffness_nm_per_rad: float
    """Extension return spring stiffness in Nm/rad."""

    damping_nms_per_rad: float
    """Viscous joint damping in Nm s/rad."""

    lower_limit_rad: float
    """Hyperextension stop, in rad."""

    upper_limit_rad: float
    """Flexion stop, in rad."""


@dataclass(frozen=True, slots=True)
class FingerGeometry:
    """A complete underactuated finger."""

    phalanges: tuple[Phalanx, ...]
    """Segments ordered from the palm outwards."""

    limit_stiffness_nm_per_rad: float
    """Stiffness of the unilateral joint end stops, in Nm/rad."""

    limit_damping_nms_per_rad: float
    """Damping of the unilateral joint end stops, in Nm s/rad."""

    @property
    def joint_count(self) -> int:
        """Number of joints, which equals the number of phalanges."""
        return len(self.phalanges)

    @property
    def total_length_m(self) -> float:
        """Summed phalanx length, in m."""
        return math.fsum(p.length_m for p in self.phalanges)


INDEX_FINGER: Final[FingerGeometry] = FingerGeometry(
    phalanges=(
        Phalanx(
            name="proximal",
            length_m=0.045,
            mass_kg=0.020,
            com_distance_m=0.0225,
            inertia_kgm2=3.375e-6,
            moment_arm_m=8.0e-3,
            spring_stiffness_nm_per_rad=0.030,
            damping_nms_per_rad=2.0e-3,
            lower_limit_rad=0.0,
            upper_limit_rad=math.radians(90.0),
        ),
        Phalanx(
            name="middle",
            length_m=0.028,
            mass_kg=0.012,
            com_distance_m=0.014,
            inertia_kgm2=7.84e-7,
            moment_arm_m=6.0e-3,
            spring_stiffness_nm_per_rad=0.020,
            damping_nms_per_rad=1.2e-3,
            lower_limit_rad=0.0,
            upper_limit_rad=math.radians(100.0),
        ),
        Phalanx(
            name="distal",
            length_m=0.022,
            mass_kg=0.008,
            com_distance_m=0.011,
            inertia_kgm2=3.227e-7,
            moment_arm_m=4.5e-3,
            spring_stiffness_nm_per_rad=0.012,
            damping_nms_per_rad=8.0e-4,
            lower_limit_rad=0.0,
            upper_limit_rad=math.radians(80.0),
        ),
    ),
    limit_stiffness_nm_per_rad=60.0,
    limit_damping_nms_per_rad=1.2e-2,
)
"""Index finger sized from adult anthropometric segment lengths and masses.

Phalanx inertias are those of uniform slender rods about their own centres of mass,
``m L^2 / 12``. Return spring rates are chosen so that a tendon tension of about 6 N
carries every joint to its flexion stop, which sets the free closing behaviour, while the
transmission can deliver more than an order of magnitude more tension for grasping.
"""


def joint_origins(
    finger: FingerGeometry, angles_rad: tuple[float, ...]
) -> tuple[tuple[Vector2, ...], tuple[Vector2, ...], tuple[float, ...]]:
    """Return joint positions, phalanx unit vectors, and absolute phalanx angles.

    Args:
        finger: Finger geometry.
        angles_rad: Joint angles measured from full extension, in rad.

    Returns:
        A triple ``(origins, directions, absolute_angles)``. ``origins`` has one more
        entry than there are joints; the last entry is the fingertip.
    """
    origins: list[Vector2] = [(0.0, 0.0)]
    directions: list[Vector2] = []
    absolute: list[float] = []
    angle = 0.0
    x = 0.0
    y = 0.0
    for phalanx, joint_angle in zip(finger.phalanges, angles_rad, strict=True):
        angle += joint_angle
        absolute.append(angle)
        unit = (math.cos(angle), math.sin(angle))
        directions.append(unit)
        x += phalanx.length_m * unit[0]
        y += phalanx.length_m * unit[1]
        origins.append((x, y))
    return tuple(origins), tuple(directions), tuple(absolute)


def point_on_phalanx(
    origins: tuple[Vector2, ...],
    directions: tuple[Vector2, ...],
    index: int,
    distance_m: float,
) -> Vector2:
    """Return the position of a point ``distance_m`` along phalanx ``index``."""
    base = origins[index]
    unit = directions[index]
    return (base[0] + distance_m * unit[0], base[1] + distance_m * unit[1])


def point_velocity(
    origins: tuple[Vector2, ...],
    point: Vector2,
    index: int,
    rates_rad_s: tuple[float, ...],
) -> Vector2:
    """Return the velocity of a point fixed to phalanx ``index``.

    For a planar chain the velocity of a point is the sum over the joints proximal to it
    of the joint rate times the vector from that joint to the point, rotated by a quarter
    turn.
    """
    vx = 0.0
    vy = 0.0
    for joint in range(index + 1):
        rate = rates_rad_s[joint]
        if rate == 0.0:
            continue
        origin = origins[joint]
        vx -= rate * (point[1] - origin[1])
        vy += rate * (point[0] - origin[0])
    return (vx, vy)


def point_jacobian_transpose(
    origins: tuple[Vector2, ...],
    point: Vector2,
    index: int,
    force: Vector2,
    joint_count: int,
) -> list[float]:
    """Return the joint torques produced by ``force`` applied at ``point``.

    The planar result is ``tau_k = (p - o_k) x F`` for every joint ``k`` proximal to the
    phalanx carrying the point, and zero for the joints distal to it.
    """
    torques = [0.0] * joint_count
    for joint in range(index + 1):
        origin = origins[joint]
        rx = point[0] - origin[0]
        ry = point[1] - origin[1]
        torques[joint] = rx * force[1] - ry * force[0]
    return torques


def mass_matrix(
    finger: FingerGeometry,
    origins: tuple[Vector2, ...],
    directions: tuple[Vector2, ...],
) -> list[list[float]]:
    """Return the joint space mass matrix ``M(q)``.

    Built from the partial derivatives of each centre of mass with respect to each joint
    angle. For a planar chain the derivative of the centre of mass of phalanx ``i`` with
    respect to joint ``k <= i`` is the quarter turn rotation of the vector from joint
    ``k`` to that centre of mass, so ``M[k][l] = sum_i m_i w_ik . w_il + I_i``, summed
    over the phalanges distal to both joints.
    """
    count = finger.joint_count
    matrix = [[0.0] * count for _ in range(count)]
    for index, phalanx in enumerate(finger.phalanges):
        com = point_on_phalanx(origins, directions, index, phalanx.com_distance_m)
        levers: list[Vector2] = []
        for joint in range(index + 1):
            origin = origins[joint]
            levers.append((-(com[1] - origin[1]), com[0] - origin[0]))
        for row in range(index + 1):
            lever_row = levers[row]
            for column in range(row, index + 1):
                lever_column = levers[column]
                value = phalanx.mass_kg * (
                    lever_row[0] * lever_column[0] + lever_row[1] * lever_column[1]
                )
                value += phalanx.inertia_kgm2
                matrix[row][column] += value
                if row != column:
                    matrix[column][row] += value
    return matrix


def inverse_dynamics(
    finger: FingerGeometry,
    angles_rad: tuple[float, ...],
    rates_rad_s: tuple[float, ...],
    accelerations_rad_s2: tuple[float, ...],
    gravity_m_per_s2: Vector2 = (0.0, 0.0),
) -> list[float]:
    """Return the joint torques required to produce a given acceleration.

    Planar recursive Newton-Euler. Gravity enters through the base acceleration, the
    standard device of accelerating the root upwards by ``-g`` so that the inertial
    forces on every body already contain the weight.
    """
    count = finger.joint_count
    _, directions, _ = joint_origins(finger, angles_rad)

    absolute_rate = 0.0
    absolute_acceleration = 0.0
    origin_ax = -gravity_m_per_s2[0]
    origin_ay = -gravity_m_per_s2[1]

    forces: list[Vector2] = []
    moments: list[float] = []
    com_offsets: list[Vector2] = []

    for index, phalanx in enumerate(finger.phalanges):
        absolute_rate += rates_rad_s[index]
        absolute_acceleration += accelerations_rad_s2[index]
        unit = directions[index]
        centripetal = absolute_rate * absolute_rate

        com_distance = phalanx.com_distance_m
        swing = absolute_acceleration * com_distance
        pull = centripetal * com_distance
        com_ax = origin_ax - swing * unit[1] - pull * unit[0]
        com_ay = origin_ay + swing * unit[0] - pull * unit[1]
        forces.append((phalanx.mass_kg * com_ax, phalanx.mass_kg * com_ay))
        moments.append(phalanx.inertia_kgm2 * absolute_acceleration)
        com_offsets.append((com_distance * unit[0], com_distance * unit[1]))

        swing = absolute_acceleration * phalanx.length_m
        pull = centripetal * phalanx.length_m
        origin_ax = origin_ax - swing * unit[1] - pull * unit[0]
        origin_ay = origin_ay + swing * unit[0] - pull * unit[1]

    torques = [0.0] * count
    force_x = 0.0
    force_y = 0.0
    moment = 0.0
    for index in range(count - 1, -1, -1):
        phalanx = finger.phalanges[index]
        unit = directions[index]
        link_x = phalanx.length_m * unit[0]
        link_y = phalanx.length_m * unit[1]
        # Moment of the force handed on by the distal neighbour, about this joint.
        moment += link_x * force_y - link_y * force_x
        body_force = forces[index]
        offset = com_offsets[index]
        moment += offset[0] * body_force[1] - offset[1] * body_force[0]
        moment += moments[index]
        force_x += body_force[0]
        force_y += body_force[1]
        torques[index] = moment
    return torques


def bias_torques(
    finger: FingerGeometry,
    angles_rad: tuple[float, ...],
    rates_rad_s: tuple[float, ...],
    gravity_m_per_s2: Vector2 = (0.0, 0.0),
) -> list[float]:
    """Return the velocity product and gravity torques, ``C(q, qdot) qdot + g(q)``."""
    zero = (0.0,) * finger.joint_count
    return inverse_dynamics(finger, angles_rad, rates_rad_s, zero, gravity_m_per_s2)


def solve_accelerations(matrix: list[list[float]], right_hand_side: list[float]) -> list[float]:
    """Solve ``M x = b`` for a symmetric positive definite ``M`` by Cholesky factorisation.

    The mass matrix of a serial chain with positive link masses is positive definite, so
    Cholesky is both the cheapest and the most numerically stable choice, and a failure
    of the factorisation is a genuine signal that the geometry is degenerate.

    Raises:
        ValueError: If ``matrix`` is not positive definite.
    """
    size = len(matrix)
    lower = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            total = matrix[row][column]
            for k in range(column):
                total -= lower[row][k] * lower[column][k]
            if row == column:
                if total <= 0.0:
                    raise ValueError("mass matrix is not positive definite")
                lower[row][column] = math.sqrt(total)
            else:
                lower[row][column] = total / lower[column][column]

    forward = [0.0] * size
    for row in range(size):
        total = right_hand_side[row]
        for k in range(row):
            total -= lower[row][k] * forward[k]
        forward[row] = total / lower[row][row]

    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        total = forward[row]
        for k in range(row + 1, size):
            total -= lower[k][row] * solution[k]
        solution[row] = total / lower[row][row]
    return solution


def kinetic_energy(matrix: list[list[float]], rates_rad_s: tuple[float, ...]) -> float:
    """Return ``0.5 * qdot^T M qdot`` in J."""
    total = 0.0
    for row, rate_row in enumerate(rates_rad_s):
        if rate_row == 0.0:
            continue
        row_values = matrix[row]
        partial = 0.0
        for column, rate_column in enumerate(rates_rad_s):
            partial += row_values[column] * rate_column
        total += rate_row * partial
    return 0.5 * total


def tendon_displacement(finger: FingerGeometry, angles_rad: tuple[float, ...]) -> float:
    """Return the cord length consumed by the joints at the given angles, in m.

    Each joint takes up ``r_j * theta_j`` of cord, so a single tendon shared between the
    joints couples them through this one scalar. This is where the underactuation enters
    the equations.
    """
    total = 0.0
    for phalanx, angle in zip(finger.phalanges, angles_rad, strict=True):
        total += phalanx.moment_arm_m * angle
    return total


def spring_torques(finger: FingerGeometry, angles_rad: tuple[float, ...]) -> list[float]:
    """Return the extension spring torques, in Nm. Negative values oppose flexion."""
    return [
        -phalanx.spring_stiffness_nm_per_rad * angle
        for phalanx, angle in zip(finger.phalanges, angles_rad, strict=True)
    ]


def spring_energy(finger: FingerGeometry, angles_rad: tuple[float, ...]) -> float:
    """Return the energy stored in the extension return springs, in J."""
    total = 0.0
    for phalanx, angle in zip(finger.phalanges, angles_rad, strict=True):
        total += 0.5 * phalanx.spring_stiffness_nm_per_rad * angle * angle
    return total


def joint_limit_torques(
    finger: FingerGeometry,
    angles_rad: tuple[float, ...],
    rates_rad_s: tuple[float, ...],
) -> tuple[list[float], float]:
    """Return the unilateral end stop torques and the energy they store.

    Each stop is a one sided spring and damper. The torque is clamped so that a stop can
    only push the joint back inside its range, never pull it further out, which is what
    makes it a stop rather than a bilateral spring.
    """
    torques = [0.0] * finger.joint_count
    stored = 0.0
    stiffness = finger.limit_stiffness_nm_per_rad
    damping = finger.limit_damping_nms_per_rad
    for index, phalanx in enumerate(finger.phalanges):
        angle = angles_rad[index]
        rate = rates_rad_s[index]
        if angle > phalanx.upper_limit_rad:
            penetration = angle - phalanx.upper_limit_rad
            stored += 0.5 * stiffness * penetration * penetration
            torques[index] = min(0.0, -stiffness * penetration - damping * rate)
        elif angle < phalanx.lower_limit_rad:
            penetration = phalanx.lower_limit_rad - angle
            stored += 0.5 * stiffness * penetration * penetration
            torques[index] = max(0.0, stiffness * penetration - damping * rate)
    return torques, stored


def limit_spring_torques(
    finger: FingerGeometry, angles_rad: tuple[float, ...]
) -> list[float]:
    """Return only the elastic part of the end stop torques, in Nm.

    Separating the elastic part from the damped part lets the energy accounting split the
    work done on the stops into a stored and a dissipated share exactly, without assuming
    which of the two dominates.
    """
    torques = [0.0] * finger.joint_count
    stiffness = finger.limit_stiffness_nm_per_rad
    for index, phalanx in enumerate(finger.phalanges):
        angle = angles_rad[index]
        if angle > phalanx.upper_limit_rad:
            torques[index] = -stiffness * (angle - phalanx.upper_limit_rad)
        elif angle < phalanx.lower_limit_rad:
            torques[index] = stiffness * (phalanx.lower_limit_rad - angle)
    return torques
