# Transradial Limb Sim

Physics simulation of transradial prosthesis actuation with tendon and motor models.

[![CI](https://github.com/Eelis03/transradial-limb-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/Eelis03/transradial-limb-sim/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

This library simulates the complete actuation path of one finger of a tendon driven
transradial prosthesis, from a lithium ion battery through a bridge, a brushed direct
current motor, a planetary gearbox and a routed tendon to the force the fingertip applies
to a grasped object. It reports where the energy goes at every stage, how the underactuated
finger conforms to different object shapes, and how the achievable closing time, grasp force
and control bandwidth depend on the tendon compliance and the routing friction.

It is written for someone sizing or reviewing a prosthetic hand drive train who needs to
know whether a chosen motor and gearbox will deliver a usable grasp force, how much of the
battery charge a grasp costs, and which component in the chain is the one worth improving.

## Problem

A tendon driven prosthetic hand has to close in under a second, hold tens of newtons at the
fingertip, conform to whatever shape it meets with one motor per finger, and run all day on
a battery small enough to wear. Those requirements pull against each other through a
transmission whose losses are not obvious.

Three properties of that transmission set the answer and none of them can be captured by a
gear ratio alone.

A tendon pulls and cannot push, so the finger needs return springs, the drive can lose the
object by backing off, and any stretch put into the cord by an impact stays there.

Where the tendon runs over a guide it loses tension exponentially in the wrap angle, so the
loss is multiplicative and cannot be reduced by pulling harder. For a routing typical of a
forearm mounted drive this removes more of the actuation force than the gearbox does.

One motor drives three joints, so the posture the finger settles into is not commanded. It
is the solution of a force balance between the tendon, the return springs and the contact,
and it is different for every object.

The problem this project solves is to predict, from published component data rather than
from measurement, the fingertip force, the closing time and the energy cost of a grasp, and
to say how confident those predictions are given that the tendon stiffness and the routing
friction of a hand that does not yet exist are not known.

## Approach

The plant is a coupled electromechanical model integrated as one system. The motor is the
standard two state lumped brushed machine, an armature resistance and inductance in series
with a back electromotive force, driving a rotor inertia against viscous and Coulomb
friction. Its parameters are the catalogue values of the maxon RE 25, order number 118752,
and the model is checked by asking it to reproduce catalogue operating points it was not
given. The gearbox is the maxon GP 26 B at 84:1, and its catalogue efficiency enters as a
load dependent Coulomb loss with a Karnopp stick band, so that a stalled drive holds rather
than creeping.

The tendon is a stainless steel rope in a metal routing, carrying a lumped series
compliance, a hard clamp at zero tension, and capstan friction over the total wrap angle of
the routing, following the Euler-Eytelwein relation. The rope construction is specified to
match the one on which the friction coefficient used here was measured, rather than assuming
a coefficient for a polymer cord, for which no directly measured value is published.

The finger is a planar three phalanx serial chain with full rigid body dynamics, a return
spring and an end stop at each joint, and a single tendon acting through a moment arm at
each joint. Contact with the grasped object uses the Hunt and Crossley law. The whole system
is advanced by a fixed step fourth order Runge-Kutta scheme under a zero order hold
controller running at 10 kHz, in a cascade of a current regulator placed by pole assignment
on the armature and an outer position or force loop.

Every power term in the model is written so that it is individually signed correctly, which
makes the energy balance an identity that holds at every instant rather than on average.
Integrating the loss channels alongside the physical states then turns the balance residual
into a direct measure of integration error, and it is reported next to every energy result.

The alternatives that were considered and rejected, including an adaptive stiff solver,
kinematic coupling between the joints, and a bristle friction model, are recorded in
[docs/design-notes.md](docs/design-notes.md) together with what each would have cost and
bought.

## Installation

Requires Python 3.12 or later.

```bash
git clone https://github.com/Eelis03/transradial-limb-sim.git
cd transradial-limb-sim
uv sync
```

Using pip instead of uv:

```bash
python -m venv .venv
.venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

```python
from transradial_sim.analysis.energetics import force_chain
from transradial_sim.pipeline.scenario import build_plant

plant = build_plant()
chain = force_chain(plant)

for stage in chain.stages:
    print(f"{stage.name:<34}{stage.value:>10.4g} {stage.unit:<3}{stage.efficiency:>8.4f}")
print(f"tendon tension at the finger: {chain.finger_tension_n:.1f} N")
```

```text
battery and bridge                     1.121 A   1.0000
motor electromagnetic                0.02623 Nm  1.0000
motor brush and bearing friction     0.02593 Nm  0.9885
gearbox                                1.285 Nm  0.5900
drive pulley                           160.6 N   1.0000
capstan friction in the routing        96.02 N   0.5978
tendon tension at the finger: 96.0 N
```

Runnable examples live in `examples/`:

```bash
uv run python examples/motor_datasheet_check.py
uv run python examples/finger_closing.py
uv run python examples/grasp_adaptivity.py
uv run python examples/efficiency_chain.py
uv run python examples/fingertip_force_curve.py
uv run python examples/force_control.py
uv run python examples/sensitivity_study.py
```

Every example accepts `--quick`, which shortens the runs for smoke testing, and
`--no-figures`, which suppresses figure output. Figures are written to `outputs/`, which is
not tracked.

## Results

All numbers below are produced by the commands shown above, on the reference configuration:
a maxon RE 25 118752 driving a maxon GP 26 B 84:1 into an 8 mm drive pulley, a tendon of
2.0e4 N/m series stiffness routed over 3.50 rad of total wrap with a measured friction
coefficient of 0.147, an index finger of 45, 28 and 22 mm phalanges, and a 22.2 V six cell
battery. The integration step is 5e-5 s except where stated and the controller runs at
10 kHz.

### Motor model against the manufacturer catalogue

From `uv run python examples/motor_datasheet_check.py`. The model is built from the
catalogue terminal resistance, inductance, torque constant and rotor inertia, and is then
asked to reproduce five operating points it was not given.

| Quantity | Model | Catalogue | Error |
| --- | --- | --- | --- |
| No load speed, rpm | 9759 | 9560 | +2.08% |
| No load current, mA | 37.4 | 36.9 | +1.35% |
| Stall torque, mNm | 241.8 | 243 | -0.51% |
| Stall current, A | 10.34 | 10.4 | -0.53% |
| Speed constant, rpm/V | 408.1 | 408 | +0.02% |

The speed constant agrees to 0.02 percent, which confirms that the conversion of the torque
constant from mNm/A into SI is right, since the catalogue publishes the two independently.
The 2.08 percent on no load speed is the catalogue's own internal inconsistency: the speed
implied by its listed torque constant and terminal resistance differs from its listed no
load speed by that amount.

The usable drive current is 1.121 A. It is not chosen. It is the current at which the
gearbox reaches its catalogue continuous output torque of 1.3 Nm, and it sits below the
motor's own continuous rating of 1.16 A. The gearbox, not the motor, sets the limit.

### The efficiency chain from battery to fingertip

From `uv run python examples/efficiency_chain.py`. This is the quasi static calculation, one
stage at a time, at the drive current limit.

| Stage | Output | Stage efficiency | Loss mechanism |
| --- | --- | --- | --- |
| Battery and bridge | 1.121 A | 1.0000 | current limited by the gearbox torque rating |
| Motor electromagnetic | 26.23 mNm | 1.0000 | torque constant, catalogue |
| Motor brush and bearing friction | 25.93 mNm | 0.9885 | Coulomb share of the no load torque |
| Gearbox, 84:1 three stage | 1.285 Nm | 0.5900 | gear tooth friction, catalogue |
| Drive pulley, 8 mm | 160.6 N | 1.0000 | kinematic |
| Capstan friction, 3.50 rad wrap | 96.0 N | 0.5978 | cord sliding on the routing guides |
| Product, motor shaft to finger | | 0.3486 | |

Two thirds of the actuation effort is lost between the motor shaft and the finger, and the
routing loses slightly more of it than the gearbox does. Only 34.9 percent of the shaft
torque reaches the cord that pulls the finger.

The measured energy budget for one complete grasp of a rigid 60 mm cylinder, integrated over
the whole 1.7 s run, is below. This is where the battery charge goes, not where the force
goes, so the shares are different.

| Destination | Energy, J | Share |
| --- | --- | --- |
| Motor copper loss, i squared R | 2.1689 | 54.81% |
| Bridge conduction and quiescent draw | 0.4998 | 12.63% |
| Gearbox tooth friction | 0.4858 | 12.28% |
| Capstan friction in the routing | 0.2820 | 7.13% |
| Motor brush and bearing friction | 0.0967 | 2.44% |
| Joint damping | 0.0097 | 0.24% |
| Battery internal resistance | 0.0054 | 0.14% |
| Tendon viscoelastic damping | 0.0003 | 0.01% |
| Joint end stops | 0.0000 | 0.00% |
| Work delivered to the object | 0.0072 | 0.18% |
| Still stored in springs and cord at the end | 0.4019 | 10.16% |
| Total drawn from the battery | 3.9570 | 100% |
| Balance residual | -6.19e-04 | 0.0156% |

The dominant loss is resistive heating in the motor winding, because a stalled grasp draws
full current and produces no mechanical power. That is the central engineering conclusion of
the loss table: a hand that holds a grasp electrically burns 1.3 W doing nothing, and the
place to attack it is a non backdrivable transmission or a mechanical latch, not a better
motor. The 10.16 percent still stored is the energy in the return springs and the stretched
cord, which is returned when the grasp is released.

The balance residual of 0.0156 percent is the integration error alone. The identity it
tests holds exactly in the continuous model, and the residual falls with the step as the
scheme's order requires.

### Fingertip force, closing time, no load speed and stall force

From `examples/finger_closing.py` and `examples/fingertip_force_curve.py`.

| Metric | Value |
| --- | --- |
| Closing time to 95 percent of travel, current limited | 0.358 s |
| Closing time with the drive closing speed limit in force | 0.80 s |
| Peak motor speed while closing | 8817 rpm |
| Motor no load speed at 22.2 V | 9026 rpm |
| Peak tendon speed | 87.9 mm/s |
| Peak fingertip speed | 766 mm/s |
| Battery energy for one free closure | 4.155 J |
| Stall tendon tension at the drive | 207.6 N |
| Stall tendon tension at the finger | 124.1 N |
| Stall grasp force, summed over the finger | 38.3 N |
| Stall fingertip force, distal phalanx | 23.1 N |

Fingertip force against motor current, measured on a rigid 60 mm cylinder. The sliding
column is the quasi static chain with the gearbox and capstan losses fully applied. The
lossless column is the same chain with both removed. The measured tension must lie between
them and does.

| Current, A | Sliding, N | Measured, N | Lossless, N | Grasp force, N | Fingertip force, N |
| --- | --- | --- | --- | --- | --- |
| 0.4484 | 37.74 | 57.20 | 107.00 | 17.25 | 10.41 |
| 0.6726 | 57.17 | 58.31 | 162.08 | 17.59 | 10.62 |
| 0.8968 | 76.59 | 95.88 | 217.17 | 29.36 | 17.73 |
| 1.1210 | 96.02 | 124.12 | 272.25 | 38.29 | 23.09 |

Grasp force rises at 31.3 N per ampere. The measured tension sits above the quasi static
value because the finger arrives at the object carrying the rotor's kinetic energy, the
impact stretches the cord, and neither the cord nor the stuck gearbox can push that stretch
back out. At the 400 rad/s closing speed limit the rotor holds 0.090 J, and a cord absorbing
all of it would stretch to 59.9 N, which is the upper bound the test suite asserts.

### Adaptive grasp, measured across three object shapes

From `uv run python examples/grasp_adaptivity.py`. The controller is identical in all three
runs and knows nothing about the object.

| Object | Proximal, N | Middle, N | Distal, N | Total, N | Contact points | Settled posture, deg |
| --- | --- | --- | --- | --- | --- | --- |
| Flat plate | 0.00 | 15.79 | 17.13 | 32.92 | 4 | 9.1, 101.1, 80.7 |
| 60 mm cylinder | 12.22 | 6.00 | 20.78 | 39.00 | 7 | 25.0, 37.3, 80.1 |
| 40 mm cylinder | 47.64 | 13.20 | 10.01 | 70.85 | 7 | 58.8, 100.3, 80.4 |

The tendon tension at the finger is 120.4, 122.6 and 120.3 N for the three objects, a spread
of 1.9 percent. One actuator sets one scalar. What the mechanism varies is how that scalar
is shared out, and the share carried by the proximal phalanx runs from zero on the flat
plate to 67 percent on the small cylinder. The root mean square difference in posture is
38.0 degrees between the flat plate and the 60 mm cylinder, 28.7 degrees between the flat
plate and the 40 mm cylinder, and 41.3 degrees between the two cylinders. A kinematically
coupled finger would return three identical rows.

### Closed loop position and force control

From `uv run python examples/force_control.py`. The armature corner frequency is 1551 Hz,
the current loop is placed at 500 Hz, and the controller samples at 10 kHz.

| Loop | Target | Settled | Error | Overshoot | Time to 90 percent |
| --- | --- | --- | --- | --- | --- |
| Motor position | 250.0 rad | 252.39 rad | +0.96% | 32.0% | 0.307 s |
| Fingertip force | 4.0 N | 3.92 N | -0.08 N | to 8.00 N peak | 0.628 s |
| Fingertip force | 8.0 N | 9.34 N | +1.34 N | to 9.69 N peak | 0.486 s |
| Fingertip force | 12.0 N | 11.86 N | -0.14 N | to 12.34 N peak | 0.575 s |

The force loop runs an integral gain of 0.05 A per newton second, which is three orders of
magnitude below what a rigid actuator would allow. Two things bound it: the current loop
underneath it, and the series compliance of the tendon, which puts a lightly damped mode
between the motor and the fingertip. The rise times also contain the approach, because no
force exists until the finger arrives.

### Sensitivity to tendon elasticity and capstan friction

From `uv run python examples/sensitivity_study.py`. These are the two parameters least well
known before a prototype exists.

Tendon series stiffness, swept over a factor of forty:

| Stiffness, N/m | Closing time, s | Grasp force, N | Fingertip force, N | Tendon tension, N | Capstan loss, J |
| --- | --- | --- | --- | --- | --- |
| 5 000 | 0.368 | 33.73 | 20.35 | 109.72 | 0.8308 |
| 10 000 | 0.362 | 35.63 | 21.49 | 115.71 | 0.4719 |
| 20 000 | 0.358 | 38.29 | 23.09 | 124.12 | 0.2820 |
| 50 000 | 0.356 | 37.07 | 22.36 | 120.28 | 0.1215 |
| 200 000 | 0.358 | 32.80 | 19.80 | 106.78 | 0.0465 |

Capstan friction coefficient:

| Coefficient | Closing time, s | Grasp force, N | Fingertip force, N | Tendon tension, N | Transmission ratio |
| --- | --- | --- | --- | --- | --- |
| 0.000 | 0.360 | 61.49 | 36.93 | 196.63 | 1.000 |
| 0.050 | 0.360 | 52.30 | 31.46 | 168.04 | 0.839 |
| 0.100 | 0.358 | 44.49 | 26.80 | 143.60 | 0.705 |
| 0.147 | 0.358 | 38.29 | 23.09 | 124.12 | 0.598 |
| 0.200 | 0.358 | 32.38 | 19.54 | 105.45 | 0.497 |

Closing time is insensitive to both, varying by 3.3 percent across the stiffness sweep and
0.6 percent across the friction sweep, because free closing is speed limited by the back
electromotive force and not force limited. Grasp force is insensitive to stiffness, varying
by 15.5 percent without a monotone trend, and strongly sensitive to friction, varying by
63.6 percent. At the reference coefficient of 0.147 the routing alone removes 36.9 percent
of the tendon tension that would otherwise reach the finger.

The engineering conclusion is that the routing is the first thing to improve. Reducing the
friction coefficient from 0.147 to 0.05, which is what a low friction liner or rolling
element idlers would give, raises the grasp force from 38.3 N to 52.3 N at the same current,
a gain no change of motor at the same package size could match.

## Architecture

| Module | Responsibility |
| --- | --- |
| `src/transradial_sim/model/units.py` | SI conversions from catalogue units and the regularised sign function |
| `src/transradial_sim/model/motor.py` | Brushed motor electrical and mechanical dynamics, catalogue parameters, predicted operating points |
| `src/transradial_sim/model/gearbox.py` | Reduction, reflected inertia, load dependent friction with a stick band |
| `src/transradial_sim/model/tendon.py` | Series compliance, one way tension clamp, capstan friction over the routing |
| `src/transradial_sim/model/finger.py` | Planar chain kinematics, mass matrix, recursive Newton-Euler, return springs, end stops |
| `src/transradial_sim/model/contact.py` | Object shapes and the Hunt and Crossley contact law |
| `src/transradial_sim/model/system.py` | The assembled plant, the state layout and the exact energy accounting |
| `src/transradial_sim/algorithm/protocols.py` | Structural interfaces for controllers and integrators |
| `src/transradial_sim/algorithm/integrators.py` | Fixed step Runge-Kutta and Euler schemes, Richardson order estimation |
| `src/transradial_sim/algorithm/controllers.py` | Current, position and force loops with anti windup, slew and speed limiting |
| `src/transradial_sim/pipeline/trace.py` | The structured record a run produces |
| `src/transradial_sim/pipeline/simulate.py` | The two rate simulation driver |
| `src/transradial_sim/pipeline/scenario.py` | The reference prosthesis, the test objects and the standard scenarios |
| `src/transradial_sim/analysis/energetics.py` | Quasi static force chain and the measured loss budget |
| `src/transradial_sim/analysis/metrics.py` | Closing time, settled grasp summary, headline performance figures |
| `src/transradial_sim/analysis/sensitivity.py` | Sweeps over tendon stiffness and capstan friction |
| `src/transradial_sim/analysis/figures.py` | Figure builders, the only module that imports matplotlib |
| `examples/` | Thin wiring scripts, no logic of their own |

Each layer imports only from the ones above it. The model layer performs no input or
output and holds no mutable state, the algorithm layer does no plotting, and the analysis
layer reads a trace and nothing else.

## Testing

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

The suite has three tiers: property and invariant tests covering the mathematics,
regression tests pinning recorded behaviour, and integration tests running each
example script under a reduced iteration count.

The invariant tier includes the checks that would fail loudest if the physics were wrong:
the motor reproduces five catalogue operating points, the torque and back electromotive
force constants agree with the independently published speed constant, energy is conserved
to 5.1e-8 relative over 40 ms in a configuration with every loss coefficient set to zero,
the energy balance closes to 2.7e-9 relative over 250 ms with every loss enabled and to
1.6e-4 over the full 1.7 s grasp reported above, tendon tension stays at zero
against an adversarial command to compress the cord by a metre at a metre per second, the
capstan relation reproduces `exp(-mu theta)` at a known wrap angle, the closed form mass
matrix agrees with the recursive Newton-Euler algorithm to 1e-12, the finger settles into
measurably different postures on a flat and a round object, and the drive current never
exceeds its limit under a full duty open loop command.

Every tolerance is derived from a measurement scale rather than from an observed error. The
two scales that appear are the integration step with the order of the scheme, and the
sample interval of a recorded trace. The derivation is stated in the docstring of each test
and the policy is recorded in [docs/design-notes.md](docs/design-notes.md).

The full suite runs in about 80 seconds.

## References

### Component data

- maxon motor ag. *RE 25, 25 mm, Graphite Brushes, 20 Watt*, order number 118752, catalogue
  page, May 2017 edition. Terminal resistance 2.32 ohm, terminal inductance 0.238 mH, torque
  constant 23.4 mNm/A, speed constant 408 rpm/V, rotor inertia 10.8 g cm squared, no load
  speed 9560 rpm, no load current 36.9 mA, stall torque 243 mNm, stall current 10.4 A,
  maximum continuous current 1.16 A.
  <https://www.maxongroup.com/maxon/view/product/motor/dcmotor/re/re25/118752>
- maxon motor ag. *Planetary Gearhead GP 26 B, 26 mm, 0.5 to 2.0 Nm*, order number 144039,
  catalogue page, May 2008 edition. Reduction 84:1, three stages, maximum efficiency
  59 percent, maximum continuous output torque 1.3 Nm, intermittently permissible 1.9 Nm,
  mass inertia 0.4 g cm squared referred to the motor shaft, average no load backlash
  1.6 degrees, recommended maximum input speed 8000 rpm.
  <https://www.maxongroup.com/maxon/view/product/gear/planetary/GP-Sonderprogramm/144039>

### Capstan friction

- Lubarda, V. A. "The Mechanics of Belt Friction Revisited." *International Journal of
  Mechanical Engineering Education*, volume 42, number 2, pages 97 to 112, 2014.
  DOI: 10.7227/IJMEE.0002. Rigorous derivation of the capstan relation, the conditions under
  which the exponential form holds, and the historical attribution.
- Euler, L. "Remarques sur l'effet du frottement dans l'equilibre." *Memoires de l'academie
  des sciences de Berlin*, volume 18, volume year 1762, printed 1769, pages 265 to 278.
  Eneström index E382. The original statement of the belt friction law.
  <https://scholarlycommons.pacific.edu/euler-works/382/>
- Jung, J. H., Pan, N., and Kang, T. J. "Capstan equation including bending rigidity and
  non-linear frictional behavior." *Mechanism and Machine Theory*, volume 43, number 6, pages
  661 to 675, 2008. DOI: 10.1016/j.mechmachtheory.2007.06.002. The corrections to the
  exponential law that a stiff cord introduces, which this model omits.
- Stuart, I. M. "Capstan equation for strings with rigidity." *British Journal of Applied
  Physics*, volume 12, number 10, pages 559 to 562, 1961. DOI: 10.1088/0508-3443/12/10/309.

### Measured tendon transmission

- Agrawal, V., Peine, W. J., and Yao, B. "Modeling of Transmission Characteristics Across a
  Cable-Conduit System." *IEEE Transactions on Robotics*, volume 26, number 5, pages 914 to
  924, 2010. DOI: 10.1109/TRO.2010.2064014. Source of the friction coefficient of 0.147 used
  here, measured on a 0.52 mm uncoated stainless steel 7 by 19 rope in a 1.2 mm inner
  diameter stainless conduit over 12 mm pulleys, at pretensions from 0.7 N to 7.3 N. A model
  fit to the same data back calculates 0.156. The reference tendon in this project is
  specified as the same construction so that the measurement applies to it directly.
- Nahvi, A., Hollerbach, J. M., Xu, Y., and Hunter, I. W. "An investigation of the
  transmission system of a tendon driven robot hand." *Proceedings of the 1994 IEEE/RSJ
  International Conference on Intelligent Robots and Systems*, volume 1, pages 202 to 208,
  1994. DOI: 10.1109/IROS.1994.407390. Reports 0.13 over routing pulleys. Measured on 12.4 mm
  pulleys with a 3 mm bore, so it is an effective coefficient dominated by the bearing rather
  than by sliding of the cord.
- Li, Z., Chen, X., Ringwald, J., Pozo Fortunic, E., Ganguly, A., and Haddadin, S.
  "Investigation of the tendon-driven transmission for anthropomorphic robotic hands."
  *Proceedings of the 2024 IEEE/RSJ International Conference on Intelligent Robots and
  Systems*, pages 8330 to 8337, 2024. DOI: 10.1109/IROS58592.2024.10801538. Measured force
  transmission efficiency of 60 percent at 10 N, which corroborates the 0.598 ratio this
  model computes for its routing.
- Grosu, V., Grosu, S., Vanderborght, B., Lefeber, D., and Rodriguez-Guerrero, C. "Design of
  Smart Modular Variable Stiffness Actuators for Robotic-Assistive Devices." *Frontiers in
  Robotics and AI*, volume 5, article 105, 2018. DOI: 10.3389/frobt.2018.00105. Cable
  transmission efficiency of 64 to 76 percent.
- Friedl, W., Chalon, M., Reinecke, J., and Grebenstein, M. "FRCEF: The new friction reduced
  and coupling enhanced finger for the Awiwi hand." *Proceedings of the 2015 IEEE-RAS
  International Conference on Humanoid Robots*, pages 140 to 147, 2015.
  DOI: 10.1109/HUMANOIDS.2015.7363527. Reports polymer tendon losses as a percentage per
  routing element rather than as a friction coefficient, about 1.1 percent for steel and 2.5
  to 4.75 percent for Dyneema, reaching 30 percent across seven pulleys. Cited here because
  no peer reviewed source reports a directly measured friction coefficient for ultra high
  molecular weight polyethylene cord on steel or aluminium, which is why the reference tendon
  in this project is a steel rope rather than a polymer cord.
- Palli, G., and Melchiorri, C. "Model and control of tendon-sheath transmission systems."
  *Proceedings of the 2006 IEEE International Conference on Robotics and Automation*, pages
  988 to 993, 2006. DOI: 10.1109/ROBOT.2006.1641838. The tendon and sheath alternative
  rejected in the design notes.
- Kaneko, M., Yamashita, T., and Tanie, K. "Basic considerations on transmission
  characteristics for tendon drive robots." *Proceedings of the Fifth International
  Conference on Advanced Robotics*, pages 827 to 832, 1991. DOI: 10.1109/ICAR.1991.240572.

### Underactuated hands

- Birglen, L., Laliberte, T., and Gosselin, C. *Underactuated Robotic Hands*. Springer Tracts
  in Advanced Robotics, volume 40, Springer, 2008. DOI: 10.1007/978-3-540-77459-4. The force
  balance formulation of underactuated fingers that the finger model follows. The series
  volume number comes from the Springer series listing rather than from the Crossref record.
- Laliberte, T., and Gosselin, C. "Simulation and design of underactuated mechanical hands."
  *Mechanism and Machine Theory*, volume 33, number 1 to 2, pages 39 to 57, 1998.
  DOI: 10.1016/S0094-114X(97)00020-7. Shape adaptation as a consequence of the force balance
  rather than of a commanded trajectory.
- Dollar, A. M., and Howe, R. D. "The Highly Adaptive SDM Hand: Design and Performance
  Evaluation." *The International Journal of Robotics Research*, volume 29, number 5, pages
  585 to 597, 2010. DOI: 10.1177/0278364909360852. Measured conformance of an underactuated
  hand across object shapes, the behaviour reproduced in the adaptivity example.
- Belter, J. T., Segil, J. L., Dollar, A. M., and Weir, R. F. "Mechanical design and
  performance specifications of anthropomorphic prosthetic hands: A review." *Journal of
  Rehabilitation Research and Development*, volume 50, number 5, pages 599 to 618, 2013.
  DOI: 10.1682/JRRD.2011.10.0188. Note that its Table 2 tabulates fingertip speeds and not
  closing times for the commercial hands, so any closing time attributed to those hands in
  this project is derived from their tabulated speed and is labelled as derived.

### Numerical methods

- Hunt, K. H., and Crossley, F. R. E. "Coefficient of Restitution Interpreted as Damping in
  Vibroimpact." *Journal of Applied Mechanics*, volume 42, number 2, pages 440 to 445, 1975.
  DOI: 10.1115/1.3423596. The contact law used in `model/contact.py`.
- Karnopp, D. "Computer Simulation of Stick-Slip Friction in Mechanical Dynamic Systems."
  *Journal of Dynamic Systems, Measurement, and Control*, volume 107, number 1, pages 100 to
  103, 1985. DOI: 10.1115/1.3140698. The stick band formulation used for the gearbox and the
  routing friction.
- Featherstone, R. *Rigid Body Dynamics Algorithms*. Springer US, 2008.
  DOI: 10.1007/978-1-4899-7560-7, print ISBN 978-0-387-74314-1. The recursive Newton-Euler
  algorithm and the composite rigid body method used to form the mass matrix. The DOI
  10.1007/978-0-387-74315-8, which circulates widely for this title, resolves to a different
  book by the same author, *Robot Dynamics Algorithms*, 1987.
- Hairer, E., Norsett, S. P., and Wanner, G. *Solving Ordinary Differential Equations I:
  Nonstiff Problems*, second revised edition. Springer Series in Computational Mathematics,
  volume 8, Springer, 1993. DOI: 10.1007/978-3-540-78862-1. Order conditions and the
  stability region of the classical fourth order Runge-Kutta scheme, and Richardson
  extrapolation for the observed order.
- Dormand, J. R., and Prince, P. J. "A family of embedded Runge-Kutta formulae." *Journal of
  Computational and Applied Mathematics*, volume 6, number 1, pages 19 to 26, 1980.
  DOI: 10.1016/0771-050X(80)90013-3.

### Dependencies

| Package | Purpose | Licence |
| --- | --- | --- |
| numpy | Array storage for simulation traces and the analysis layer | BSD 3-Clause |
| scipy | Available for numerical utilities in the analysis layer | BSD 3-Clause |
| matplotlib | Figure generation in `analysis/figures.py` | Matplotlib licence, PSF based |
| pytest | Test runner for all three tiers | MIT |
| ruff | Linting and import ordering | MIT |
| mypy | Static type checking under strict mode | MIT |

## License

Released under the MIT license. See [LICENSE](LICENSE).
