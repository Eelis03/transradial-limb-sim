"""Energy accounting: the quasi static force chain and the measured loss budget.

Two complementary views of the same machine.

The force chain is the calculation an engineer does by hand: start at the battery, apply
each stage's ratio and each stage's loss, and arrive at a tendon tension. It has no
dynamics in it and it is the number a design is sized against.

The energy budget is measured from a simulation run. It reports where the energy drawn
from the battery actually went during a complete grasp, one row per physical mechanism.
The two disagree only where the dynamics matter, and the size of that disagreement is
itself a reported result.
"""

from __future__ import annotations

from dataclasses import dataclass

from transradial_sim.model.system import INTERNAL_LOSS_NAMES, SystemParameters
from transradial_sim.model.tendon import capstan_ratio
from transradial_sim.pipeline.trace import SimulationTrace

__all__ = [
    "ChainStage",
    "EnergyBudget",
    "ForceChain",
    "LossRow",
    "energy_budget",
    "force_chain",
    "frictionless_tension_n",
]


@dataclass(frozen=True, slots=True)
class ChainStage:
    """One stage of the quasi static chain from the battery to the tendon."""

    name: str
    """What the stage is."""

    quantity: str
    """Name of the quantity leaving the stage."""

    value: float
    """Value of that quantity in the unit named by :attr:`unit`."""

    unit: str
    """Unit of :attr:`value`."""

    efficiency: float
    """Fraction of the incoming effort that survives the stage, one when lossless."""

    note: str
    """Where the number for this stage comes from."""


@dataclass(frozen=True, slots=True)
class ForceChain:
    """The quasi static chain from battery to tendon tension at the finger."""

    stages: tuple[ChainStage, ...]
    current_a: float
    drive_tension_n: float
    finger_tension_n: float

    @property
    def transmission_efficiency(self) -> float:
        """Product of every stage efficiency between the motor shaft and the finger."""
        product = 1.0
        for stage in self.stages:
            product *= stage.efficiency
        return product


def force_chain(params: SystemParameters, current_a: float | None = None) -> ForceChain:
    """Return the quasi static force chain at a given armature current.

    Args:
        params: Plant parameters.
        current_a: Armature current. The drive current limit by default.

    Returns:
        The chain, one stage at a time, ending at the tendon tension the finger sees.
    """
    current = current_a if current_a is not None else params.current_limit_a
    motor = params.motor
    gearbox = params.gearbox
    tendon = params.tendon

    electromagnetic = motor.torque_constant_nm_per_a * current
    shaft = electromagnetic - motor.coulomb_friction_nm
    output = gearbox.efficiency * gearbox.ratio * shaft
    drive_tension = output / tendon.drive_radius_m
    ratio = capstan_ratio(tendon.friction_coefficient, tendon.wrap_angle_rad)
    finger_tension = drive_tension * ratio

    stages = (
        ChainStage(
            name="battery and bridge",
            quantity="armature current",
            value=current,
            unit="A",
            efficiency=1.0,
            note="current limited by the gearbox continuous torque rating",
        ),
        ChainStage(
            name="motor electromagnetic",
            quantity="air gap torque",
            value=electromagnetic,
            unit="Nm",
            efficiency=1.0,
            note="torque constant from the catalogue",
        ),
        ChainStage(
            name="motor brush and bearing friction",
            quantity="shaft torque",
            value=shaft,
            unit="Nm",
            efficiency=shaft / electromagnetic if electromagnetic else 1.0,
            note="Coulomb share of the catalogue no load torque",
        ),
        ChainStage(
            name="gearbox",
            quantity="output torque",
            value=output,
            unit="Nm",
            efficiency=gearbox.efficiency,
            note="catalogue maximum efficiency",
        ),
        ChainStage(
            name="drive pulley",
            quantity="tendon tension at the drive",
            value=drive_tension,
            unit="N",
            efficiency=1.0,
            note="kinematic, no loss modelled at the pulley itself",
        ),
        ChainStage(
            name="capstan friction in the routing",
            quantity="tendon tension at the finger",
            value=finger_tension,
            unit="N",
            efficiency=ratio,
            note=f"exp(-mu theta) with mu {tendon.friction_coefficient} "
            f"and theta {tendon.wrap_angle_rad:.2f} rad",
        ),
    )
    return ForceChain(
        stages=stages,
        current_a=current,
        drive_tension_n=drive_tension,
        finger_tension_n=finger_tension,
    )


def frictionless_tension_n(params: SystemParameters, current_a: float | None = None) -> float:
    """Return the tendon tension a lossless transmission would give, in N.

    Upper bound on what the drive can produce at a given current: the gearbox passes all
    of the torque and the routing passes all of the tension. The measured tension must lie
    between :func:`force_chain` and this value, and where it lies inside that band is a
    direct measure of how much of the sliding friction is active at the operating point.
    """
    current = current_a if current_a is not None else params.current_limit_a
    shaft = params.motor.torque_constant_nm_per_a * current - params.motor.coulomb_friction_nm
    return params.gearbox.ratio * shaft / params.tendon.drive_radius_m


@dataclass(frozen=True, slots=True)
class LossRow:
    """One named destination of the energy drawn from the battery."""

    name: str
    energy_j: float
    share: float
    """Fraction of the battery energy, between zero and one."""


@dataclass(frozen=True, slots=True)
class EnergyBudget:
    """Where the energy taken from the battery went during one run."""

    battery_energy_j: float
    losses: tuple[LossRow, ...]
    object_work_j: float
    stored_change_j: float
    residual_j: float

    @property
    def object_share(self) -> float:
        """Fraction of the battery energy delivered to the grasped object."""
        return self.object_work_j / self.battery_energy_j if self.battery_energy_j else 0.0

    @property
    def stored_share(self) -> float:
        """Fraction of the battery energy still held in the mechanism at the end."""
        return self.stored_change_j / self.battery_energy_j if self.battery_energy_j else 0.0

    @property
    def relative_residual(self) -> float:
        """Closure error of the balance as a fraction of the battery energy."""
        return abs(self.residual_j) / self.battery_energy_j if self.battery_energy_j else 0.0

    def loss(self, name: str) -> LossRow:
        """Return one loss row by name.

        Raises:
            KeyError: If no row has that name.
        """
        for row in self.losses:
            if row.name == name:
                return row
        raise KeyError(name)


def energy_budget(trace: SimulationTrace) -> EnergyBudget:
    """Return the measured energy budget of a run.

    The rows are the integrated loss channels of the plant, plus the work delivered to the
    grasped object and the change in stored energy. Their sum equals the battery energy up
    to the integration error, which is reported as the residual so it can be judged
    against the numbers rather than hidden inside them.
    """
    battery = trace.final_accumulator("battery_energy_j") - float(
        trace.accumulator("battery_energy_j")[0]
    )
    rows: list[LossRow] = []
    consumed = 0.0
    for name in INTERNAL_LOSS_NAMES:
        channel = trace.accumulator(name)
        value = float(channel[-1] - channel[0])
        consumed += value
        rows.append(
            LossRow(
                name=name.removesuffix("_j"),
                energy_j=value,
                share=value / battery if battery else 0.0,
            )
        )
    work_channel = trace.accumulator("object_work_j")
    work = float(work_channel[-1] - work_channel[0])
    stored = trace.final_stored.internal_total_j - trace.initial_stored.internal_total_j
    residual = battery - consumed - work - stored
    return EnergyBudget(
        battery_energy_j=battery,
        losses=tuple(rows),
        object_work_j=work,
        stored_change_j=stored,
        residual_j=residual,
    )
