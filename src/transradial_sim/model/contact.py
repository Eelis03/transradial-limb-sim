"""Compliant contact between the finger and a grasped object.

Contact uses the Hunt and Crossley law, a nonlinear elastic term with a damping term
proportional to the same nonlinear stiffness,

    ``f = k * delta^n * (1 + a * delta_dot)``  clamped at zero,

which is preferred over a linear spring and damper because it makes the contact force
start and end at zero rather than jumping at the instant of touchdown and pulling the
bodies together on release. The exponent ``n = 3/2`` is the Hertzian value for smooth
convex bodies.

Each phalanx is sampled at several points along its length so that a phalanx lying flat
against a surface produces a distributed reaction rather than a single point load. That
discretisation is the main approximation in this module and is stated in the design notes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from transradial_sim.model.finger import (
    FingerGeometry,
    Vector2,
    point_jacobian_transpose,
    point_on_phalanx,
    point_velocity,
)

__all__ = [
    "CircularObject",
    "ContactGeometry",
    "ContactModel",
    "ContactResult",
    "FlatSurface",
    "evaluate_contact",
    "no_contact",
]


@runtime_checkable
class ContactGeometry(Protocol):
    """A rigid object shape that can report how far a point has entered it."""

    def penetration(self, point: Vector2) -> tuple[float, Vector2]:
        """Return the penetration depth and the outward unit normal at ``point``.

        The depth is positive inside the object and zero or negative outside. The normal
        points out of the object, so it is the direction in which the object pushes the
        finger.
        """
        ...


@dataclass(frozen=True, slots=True)
class FlatSurface:
    """A half plane, used to represent a flat object such as a card or a table edge."""

    point_m: Vector2
    """Any point on the surface, in m."""

    normal: Vector2
    """Unit normal pointing away from the material, into free space."""

    def penetration(self, point: Vector2) -> tuple[float, Vector2]:
        """Return the penetration depth and outward normal for ``point``."""
        offset_x = point[0] - self.point_m[0]
        offset_y = point[1] - self.point_m[1]
        signed = offset_x * self.normal[0] + offset_y * self.normal[1]
        return -signed, self.normal


@dataclass(frozen=True, slots=True)
class CircularObject:
    """A disc, used to represent a cylindrical or spherical object."""

    centre_m: Vector2
    """Centre of the disc, in m."""

    radius_m: float
    """Radius of the disc, in m."""

    def penetration(self, point: Vector2) -> tuple[float, Vector2]:
        """Return the penetration depth and outward normal for ``point``."""
        offset_x = point[0] - self.centre_m[0]
        offset_y = point[1] - self.centre_m[1]
        distance = math.hypot(offset_x, offset_y)
        if distance <= 1.0e-12:
            return self.radius_m, (1.0, 0.0)
        return self.radius_m - distance, (offset_x / distance, offset_y / distance)


@dataclass(frozen=True, slots=True)
class ContactModel:
    """Parameters of the compliant contact law."""

    stiffness_n_per_m_pow: float
    """Hunt and Crossley stiffness ``k``, in N/m^n with ``n`` the exponent."""

    exponent: float
    """Force exponent ``n``. The Hertzian value for smooth convex bodies is 1.5."""

    damping_s_per_m: float
    """Hunt and Crossley damping factor ``a``, in s/m."""

    samples_per_phalanx: int
    """Number of contact sample points distributed along each phalanx."""

    def __post_init__(self) -> None:
        if self.samples_per_phalanx < 1:
            raise ValueError("samples_per_phalanx must be at least one")
        if self.exponent <= 0.0:
            raise ValueError("exponent must be strictly positive")


@dataclass(frozen=True, slots=True)
class ContactResult:
    """Aggregated result of evaluating every contact sample point."""

    joint_torques_nm: tuple[float, ...]
    """Torque contributed to each joint by the contact forces, in Nm."""

    stored_energy_j: float
    """Elastic energy held in the compliant object, in J."""

    power_into_object_w: float
    """Rate at which the finger does work on the object, in W."""

    dissipated_power_w: float
    """Rate at which the object dissipates energy, in W. Never negative."""

    total_normal_force_n: float
    """Sum of the magnitudes of all contact normal forces, in N."""

    phalanx_forces_n: tuple[float, ...]
    """Summed normal force on each phalanx, in N. The distribution across this tuple is
    the measurable signature of an adaptive grasp."""

    contact_count: int = field(default=0)
    """Number of sample points currently in contact."""

    @property
    def fingertip_force_n(self) -> float:
        """Total normal force on the distal phalanx, in N.

        Defined over the whole distal phalanx rather than a single sample point so that
        the reported value does not depend on how finely the phalanx is sampled.
        """
        return self.phalanx_forces_n[-1] if self.phalanx_forces_n else 0.0


def _sample_distances(length_m: float, count: int) -> tuple[float, ...]:
    """Return sample positions along a phalanx, ending exactly at the tip."""
    return tuple(length_m * (index + 1) / count for index in range(count))


def evaluate_contact(
    finger: FingerGeometry,
    model: ContactModel,
    obstacle: ContactGeometry,
    origins: tuple[Vector2, ...],
    directions: tuple[Vector2, ...],
    rates_rad_s: tuple[float, ...],
) -> ContactResult:
    """Evaluate the contact forces between the finger and one object.

    Args:
        finger: Finger geometry.
        model: Contact law parameters.
        obstacle: Object shape.
        origins: Joint positions from :func:`~transradial_sim.model.finger.joint_origins`.
        directions: Phalanx unit vectors from the same call.
        rates_rad_s: Joint angular rates, in rad/s.

    Returns:
        The aggregated contact result. Every force is clamped at zero, so the object can
        push the finger and never pull it.
    """
    joint_count = finger.joint_count
    torques = [0.0] * joint_count
    per_phalanx = [0.0] * joint_count
    stored = 0.0
    power_in = 0.0
    dissipated = 0.0
    total_force = 0.0
    contacts = 0

    for index, phalanx in enumerate(finger.phalanges):
        for distance in _sample_distances(phalanx.length_m, model.samples_per_phalanx):
            point = point_on_phalanx(origins, directions, index, distance)
            depth, normal = obstacle.penetration(point)
            if depth <= 0.0:
                continue
            velocity = point_velocity(origins, point, index, rates_rad_s)
            depth_rate = -(velocity[0] * normal[0] + velocity[1] * normal[1])

            elastic_force = model.stiffness_n_per_m_pow * depth**model.exponent
            damped_force = elastic_force * model.damping_s_per_m * depth_rate
            normal_force = max(0.0, elastic_force + damped_force)

            stored += elastic_force * depth / (model.exponent + 1.0)
            power_in += normal_force * depth_rate
            dissipated += (normal_force - elastic_force) * depth_rate
            total_force += normal_force
            per_phalanx[index] += normal_force
            contacts += 1

            force = (normal[0] * normal_force, normal[1] * normal_force)
            contribution = point_jacobian_transpose(origins, point, index, force, joint_count)
            for joint in range(index + 1):
                torques[joint] += contribution[joint]

    return ContactResult(
        joint_torques_nm=tuple(torques),
        stored_energy_j=stored,
        power_into_object_w=power_in,
        dissipated_power_w=dissipated,
        total_normal_force_n=total_force,
        phalanx_forces_n=tuple(per_phalanx),
        contact_count=contacts,
    )


def no_contact(joint_count: int) -> ContactResult:
    """Return the result that represents free motion with no object present."""
    zeros = (0.0,) * joint_count
    return ContactResult(
        joint_torques_nm=zeros,
        stored_energy_j=0.0,
        power_into_object_w=0.0,
        dissipated_power_w=0.0,
        total_normal_force_n=0.0,
        phalanx_forces_n=zeros,
        contact_count=0,
    )
