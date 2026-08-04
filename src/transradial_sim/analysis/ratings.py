"""Catalogue rating audit: what a run asks of each component against what it is rated for.

The reference drive train is sized against two published ratings by hand. The current limit
is the current at which the gearhead reaches its continuous output torque, and it is checked
against the motor's own continuous current. Both statements are quasi static and both are
made about a chain that is not. A finger that arrives at an object carrying the kinetic
energy of the rotor stretches the cord past the tension the chain predicts, and a drive that
is not speed limited runs the gearhead near the free running speed of the motor on the way
there. Neither shows up in a force balance.

This module reads a trace and compares it against the ratings the parameter objects already
carry. Each check reports the peak the run reached, the margin against the rating, and how
long the run spent above it, because a rating is a statement about duration as much as about
magnitude: a gearhead may be taken to its intermittently permissible torque and not held
there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from transradial_sim.pipeline.trace import SimulationTrace

__all__ = ["RatingAudit", "RatingCheck", "rating_audit"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RatingCheck:
    """One published rating measured against what a run asked of it."""

    name: str
    """What is being checked."""

    component: str
    """Manufacturer order number of the part that publishes the rating."""

    rating: float
    """Published limit, in the unit named by :attr:`unit`."""

    peak: float
    """Largest magnitude the run reached, in the same unit."""

    unit: str
    """Unit of :attr:`rating` and :attr:`peak`."""

    time_above_s: float
    """Time the run spent above the rating, in s."""

    share: float
    """Fraction of the run spent above the rating, between zero and one."""

    @property
    def margin(self) -> float:
        """Peak divided by the rating, greater than one when the rating was exceeded."""
        return self.peak / self.rating

    @property
    def exceeded(self) -> bool:
        """Whether the run went above the rating at any point."""
        return self.peak > self.rating


@dataclass(frozen=True, slots=True)
class RatingAudit:
    """Every published rating a run can touch, checked against that run."""

    name: str
    duration_s: float
    checks: tuple[RatingCheck, ...]

    @property
    def exceedances(self) -> tuple[RatingCheck, ...]:
        """The ratings the run went above, the largest margin first.

        Ordered by margin rather than by the order the checks are built in, so that the
        first entry is the one worth acting on.
        """
        exceeded = [check for check in self.checks if check.exceeded]
        exceeded.sort(key=lambda check: check.margin, reverse=True)
        return tuple(exceeded)

    @property
    def within_ratings(self) -> bool:
        """Whether the run stayed inside every rating it was checked against."""
        return not self.exceedances

    @property
    def worst(self) -> RatingCheck:
        """The check with the largest margin, whether or not it was exceeded."""
        return max(self.checks, key=lambda check: check.margin)

    def check(self, name: str) -> RatingCheck:
        """Return one check by name.

        Raises:
            KeyError: If no check has that name.
        """
        for item in self.checks:
            if item.name == name:
                return item
        raise KeyError(name)


def _time_above(times: FloatArray, values: FloatArray, rating: float) -> float:
    """Return the time a sampled signal spends above a rating, in s.

    The samples are read as the piecewise linear signal between them and each crossing is
    placed by interpolating the two that bracket it. Counting whole sample intervals
    instead would quantise the answer to the sample stride, and for a rating that is
    crossed briefly that quantisation is the whole of the answer.
    """
    parts: list[float] = []
    for index in range(1, int(times.shape[0])):
        span = float(times[index] - times[index - 1])
        start = float(values[index - 1]) - rating
        end = float(values[index]) - rating
        if start > 0.0 and end > 0.0:
            parts.append(span)
        elif start > 0.0:
            parts.append(span * start / (start - end))
        elif end > 0.0:
            parts.append(span * end / (end - start))
    return math.fsum(parts)


def _rating_check(
    name: str,
    component: str,
    rating: float,
    unit: str,
    times: FloatArray,
    values: FloatArray,
    duration_s: float,
) -> RatingCheck:
    """Measure one signal against one rating over a run of a stated length."""
    time_above = _time_above(times, values, rating)
    return RatingCheck(
        name=name,
        component=component,
        rating=rating,
        peak=float(np.max(values)),
        unit=unit,
        time_above_s=time_above,
        share=time_above / duration_s,
    )


def rating_audit(trace: SimulationTrace) -> RatingAudit:
    """Return the catalogue rating audit of one run.

    The signals are taken as magnitudes, because a rating bounds how hard a part is worked
    and not which way it turns.

    Args:
        trace: Recorded run.

    Returns:
        One check per published rating the run can touch: the motor continuous current,
        the two gearbox output torque ratings, and the gearbox recommended input speed.

    Raises:
        ValueError: If the run has no duration, which leaves nothing to measure against.
    """
    duration_s = trace.duration_s
    if duration_s <= 0.0:
        raise ValueError("a rating audit needs a run of nonzero duration")
    motor = trace.params.motor
    gearbox = trace.params.gearbox
    times = trace.time_s
    output_torque = np.abs(trace.output_torque_nm)
    checks = (
        _rating_check(
            "motor continuous current",
            motor.catalogue.part_number,
            motor.catalogue.max_continuous_current_a,
            "A",
            times,
            np.abs(trace.current_a),
            duration_s,
        ),
        _rating_check(
            "gearbox continuous output torque",
            gearbox.part_number,
            gearbox.max_continuous_output_torque_nm,
            "Nm",
            times,
            output_torque,
            duration_s,
        ),
        _rating_check(
            "gearbox intermittent output torque",
            gearbox.part_number,
            gearbox.max_intermittent_output_torque_nm,
            "Nm",
            times,
            output_torque,
            duration_s,
        ),
        _rating_check(
            "gearbox input speed",
            gearbox.part_number,
            gearbox.max_input_speed_rad_s,
            "rad/s",
            times,
            np.abs(trace.motor_speed_rad_s),
            duration_s,
        ),
    )
    return RatingAudit(name=trace.name, duration_s=duration_s, checks=checks)
