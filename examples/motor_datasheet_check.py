"""Compare the motor model against the published catalogue operating points.

Runs no simulation. It evaluates the steady state of the motor equations from the
catalogue parameters and prints the result next to the catalogue values, which is an
external check that the parameters were entered and converted correctly.
"""

from __future__ import annotations

import math

from _common import parse_options

from transradial_sim.model.gearbox import (
    MAXON_GP26B_84,
    current_limit_from_gearbox,
    lost_motion_m,
)
from transradial_sim.model.motor import (
    CATALOGUE_SPEED_CONSTANT_RPM_PER_V,
    MAXON_RE25_118752,
    predicted_no_load_current,
    predicted_no_load_speed,
    predicted_stall_torque,
)
from transradial_sim.model.tendon import PROSTHETIC_TENDON
from transradial_sim.model.units import nm_to_mnm, rad_s_to_rpm


def main() -> None:
    """Print the datasheet comparison table."""
    parse_options(__doc__ or "motor datasheet check")
    motor = MAXON_RE25_118752
    catalogue = motor.catalogue
    voltage = catalogue.nominal_voltage_v

    rows = [
        (
            "no load speed, rpm",
            rad_s_to_rpm(predicted_no_load_speed(motor, voltage)),
            rad_s_to_rpm(catalogue.no_load_speed_rad_s),
        ),
        (
            "no load current, mA",
            1.0e3 * predicted_no_load_current(motor, voltage),
            1.0e3 * catalogue.no_load_current_a,
        ),
        (
            "stall torque, mNm",
            nm_to_mnm(predicted_stall_torque(motor, voltage)),
            nm_to_mnm(catalogue.stall_torque_nm),
        ),
        (
            "stall current, A",
            voltage / motor.resistance_ohm,
            catalogue.stall_current_a,
        ),
        (
            "speed constant, rpm/V",
            rad_s_to_rpm(1.0 / motor.back_emf_constant_v_s_per_rad),
            CATALOGUE_SPEED_CONSTANT_RPM_PER_V,
        ),
    ]

    print(f"motor: {catalogue.part_number} at {voltage:.1f} V")
    print(f"{'quantity':<24}{'model':>12}{'catalogue':>12}{'error':>10}")
    for name, model_value, catalogue_value in rows:
        error = 100.0 * (model_value / catalogue_value - 1.0)
        print(f"{name:<24}{model_value:>12.4g}{catalogue_value:>12.4g}{error:>9.2f}%")

    print()
    print(f"electrical time constant   {1.0e6 * motor.electrical_time_constant_s:.1f} us")
    print(f"torque constant            {nm_to_mnm(motor.torque_constant_nm_per_a):.3f} mNm/A")
    print(
        "back emf constant          "
        f"{motor.back_emf_constant_v_s_per_rad:.5f} V s/rad, equal to the torque constant"
    )
    print(f"viscous friction           {motor.viscous_friction_nms:.4e} Nm s/rad")
    print(f"Coulomb friction           {nm_to_mnm(motor.coulomb_friction_nm):.4f} mNm")

    print()
    limit = current_limit_from_gearbox(MAXON_GP26B_84, motor.torque_constant_nm_per_a)
    play_m = lost_motion_m(MAXON_GP26B_84, PROSTHETIC_TENDON.drive_radius_m)
    print(f"gearbox: {MAXON_GP26B_84.part_number}")
    print(f"current at the gearbox continuous torque rating   {limit:.3f} A")
    print(
        "motor maximum continuous current                  "
        f"{catalogue.max_continuous_current_a:.3f} A"
    )
    print("the gearbox sets the usable current, not the motor")
    print(
        "average backlash at no load                       "
        f"{math.degrees(MAXON_GP26B_84.backlash_rad):.1f} deg at the output"
    )
    print(
        f"lost motion on the {1.0e3 * PROSTHETIC_TENDON.drive_radius_m:.0f} mm drive pulley"
        f"                {1.0e3 * play_m:.3f} mm of cord"
    )


if __name__ == "__main__":
    main()
