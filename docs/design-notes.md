# Design notes for Transradial Limb Sim

## Method selection

### A lumped parameter dynamic model, not a quasi static one

The question this project answers is where the actuation effort of a tendon driven
prosthetic finger goes, and what limits how fast and how hard it can close. Both parts
need dynamics. A quasi static force chain, which is the calculation an engineer does by
hand, gives the grasp force at a given current but says nothing about closing time, about
the bandwidth a force controller can achieve, or about how much of the battery energy is
spent accelerating the rotor rather than squeezing the object. The model therefore carries
state: armature current, motor shaft angle and speed, three joint angles and their rates,
and a set of energy accumulators.

The quasi static chain is still computed, in `analysis/energetics.py`, and the two are
compared in the Results section of the README. Where they disagree is informative rather
than embarrassing, and the disagreement is quantified.

### Brushed direct current motor, two state lumped model

The armature is a resistance and an inductance in series with a back electromotive force
proportional to speed, and the shaft carries the rotor inertia against viscous and Coulomb
friction. This is the standard model and it has the property that matters here: every
parameter in it is published by the manufacturer, so the model can be checked against
operating points it was not given. The reproduction of the catalogue no load speed, no
load current, stall torque, stall current and speed constant is the strongest external
validation available without hardware. Parameters are those of the maxon RE 25, order
number 118752, and the gearbox is the maxon GP 26 B at 84:1.

The one parameter the manufacturer does not publish separately is the split of the no load
torque between viscous and Coulomb friction. Their sum is fixed by the catalogue no load
current at the catalogue no load speed. The split is set by a stated assumption, 35 percent
Coulomb, which is documented in `model/motor.py` and exposed as an argument rather than
hidden as a constant.

The torque constant and the back electromotive force constant are not carried as separate
parameters. For an ideal machine they are the same number in SI units, and treating them
as one removes the possibility of a unit error in the conversion from mNm/A and rpm/V
producing a self consistent but wrong model. The catalogue publishes the speed constant
independently, so the conversion is checked against it.

### Gearbox efficiency as a load dependent Coulomb loss

The catalogue quotes a maximum efficiency, which is a statement about steady forward
motion: the output torque is `eta` times the ratio times the motor torque. The loss torque
referred to the motor shaft that produces this is `(1 / eta - 1)` times the transmitted
torque referred to the same shaft. Writing the coefficient as `1 - eta` instead is a common
slip and understates the loss by the factor `eta`, which for a three stage gearhead at
59 percent is a 41 percent error in the loss term. The model uses the first form and a test
asserts the catalogue relation directly.

Below a small speed the friction switches from sliding to sticking, following Karnopp: it
takes whatever value inside its capacity cancels the net driving torque. Without stiction a
regularised sign function returns zero friction at zero speed, a stalled drive keeps
turning, and the tendon tension climbs towards the frictionless limit, which for this
transmission is 2.8 times the correct value. That failure mode was observed during
development and is the reason the stick branch exists. The switch follows Karnopp.

### Capstan friction for the routing

Where a cord runs over a guide it presses on the guide and the friction changes the tension
along the contact arc. Integrating around the arc gives the capstan relation, tension ratio
`exp(-mu theta)`. Two properties make it the right model here. It depends only on the total
wrap angle, so the routing is described by one number rather than by a list of guide radii,
and the loss is multiplicative rather than additive, so it cannot be reduced by pulling
harder. Both properties are asserted in the test suite.

The same stiction treatment is applied. Coulomb friction does not fall off with speed, so a
routing that has stopped sliding still holds the tension difference it had when it stopped.
Inside the stick band the direction of the friction is taken from the direction the drive is
pushing rather than from the measured speed.

The reference tendon is specified as a stainless steel rope of the construction Agrawal,
Peine and Yao measured, because their coefficient of 0.147 then applies to the modelled
geometry directly. A polymer cord was considered and not used as the reference, because no
peer reviewed source reports a directly measured friction coefficient for ultra high
molecular weight polyethylene on steel or aluminium. The figures that circulate for it, near
0.085 on aluminium, trace only to vendor pages. Friedl and colleagues instead report polymer
losses per routing element, which is a different parameterisation and cannot be converted
into a wrap angle law without knowing their geometry.

### Planar rigid body dynamics by recursive Newton-Euler

The finger is a planar serial chain of three phalanges. The equations of motion are formed
by computing the mass matrix in closed form from the centre of mass Jacobians and the bias
torques by a recursive Newton-Euler sweep, then solving the linear system by Cholesky
factorisation. The two routes to the mass matrix are independent, and the test suite asserts
that a unit acceleration passed through the recursive algorithm reproduces the closed form
matrix column by column. That agreement is a strong internal check on the dynamics.

### Hunt and Crossley contact

Contact uses a nonlinear elastic term with a damping term proportional to the same
nonlinear stiffness. It is preferred to a linear spring and damper because the contact
force starts and ends at zero rather than jumping at touchdown and pulling the bodies
together on release, which a linear damper does. The exponent is the Hertzian value of 3/2.

### Fixed step integration

A fixed step fourth order Runge-Kutta scheme is used rather than an adaptive one because
the plant is driven by a zero order hold controller at a fixed rate. With an adaptive solver
the control instants would have to be imposed as event boundaries, and the reported control
bandwidth would depend on the solver tolerance rather than on the drive. A fixed step also
makes the convergence study a direct statement about the reported results: halving the step
is the only change made.

## Rejected alternatives

### An adaptive stiff solver

`scipy.integrate.solve_ivp` with Radau or LSODA would take far fewer steps for the same
accuracy, because the armature pole at 9748 rad/s is three orders of magnitude faster than
the mechanical motion of interest. It was rejected for the reason above, and for a second
reason: the plant is only piecewise smooth. The tendon clamps at zero tension, contact
points appear and disappear, and the friction terms switch between sliding and sticking. An
adaptive solver meeting a tight tolerance across those events spends most of its effort
resolving them, and the step rejection pattern becomes part of the answer. What it would
have bought is roughly a tenfold reduction in run time.

### Modelling the tendon as a bidirectional rod

Removing the clamp at zero tension would make the plant smooth, remove the need for the
stiction treatment, and roughly halve the development effort. It was rejected because the
one way property is the defining feature of tendon actuation. A finger driven by a rod does
not need return springs, does not lose the object when the drive backs off, and does not
show the residual tension after an impact that this model reports. Every one of those is a
real behaviour of a tendon driven hand.

### Kinematic coupling between the joints

Many underactuated fingers are modelled by imposing a fixed ratio between the joint angles,
which reduces three degrees of freedom to one and removes the need for contact forces to
determine the posture. It was rejected because it assumes away the result. A coupled finger
reaches the same posture on every object, and the measurement in
`examples/grasp_adaptivity.py` would return three identical rows. What it would have bought
is a much cheaper simulation and a much simpler contact model.

### A Dahl or LuGre bristle friction model

Both replace the Karnopp switch with a continuous friction state and give a smooth right
hand side, which would restore full fourth order convergence through the stick to slip
transition. They were rejected because each adds a stiff state per friction element, three
in total, and because the bristle stiffness is a parameter nobody publishes for a planetary
gearhead. Karnopp's switch needs only a velocity threshold, and the threshold can be chosen
from a quantity that is measurable, namely the cord speed below which the drive is not
usefully moving. What a bristle model would have bought is a cleaner order study.

### A tendon and sheath transmission instead of pulleys

Palli and Melchiorri model a tendon inside a flexible conduit, where the wrap angle varies
with the posture of the arm and the transmission characteristic is history dependent. That
is the right model for a cable running through a wrist that moves. It was rejected because
the reference design routes the cord over fixed idlers inside the forearm shell, so the wrap
angle is a property of the mechanism rather than of the arm posture. What it would have
bought is the ability to report how the grasp force changes when the wrist is flexed.

### Finite element structural analysis of the phalanges

Rejected as out of scope. See the limitations below for what it would change.

## Known limitations

This is a lumped parameter model. There is no finite element structural analysis, no
thermal model, and no hardware validation. Each omission is named below with what it would
change.

### No finite element structural analysis

The phalanges are rigid bodies and the only structural compliance in the model is the
lumped tendon series stiffness and the joint end stops. A real finger frame deflects under
a 120 N tendon load, and the idler pulleys move with it, which changes the moment arms and
therefore the joint torque distribution. A finite element model would give the frame
compliance and the moment arm change under load. The expected effect is a reduction in
fingertip force of a few percent and a small change in the posture a given object produces.
It would not change the loss table, because frame deflection is elastic and is returned
when the grasp is released.

### No thermal model

Winding resistance is constant. A copper winding rises about 0.39 percent in resistance per
kelvin, and the motor copper loss in the reported grasp is 2.17 J over 1.7 s, which is 1.3 W
on average against a winding to housing thermal resistance of roughly 14 K/W. A grasp held
for tens of seconds would raise the winding by tens of kelvin and increase the copper loss
by about ten percent at the same current, reducing the fraction of battery energy that
reaches the object. The current limit is set here by the gearbox torque rating rather than
by motor heating, so the reported forces would not change, but the reported efficiency
would fall for a sustained grasp. Nothing in this model can predict how long a grasp may be
held.

### No hardware validation

Every number in the Results section comes from the model. The motor and gearbox parameters
are catalogue values and the friction coefficient is a published measurement on a comparable
cord and routing, but no part of the assembled system has been measured. The parameters
carrying the most uncertainty are the tendon series stiffness, which is swept over a factor
of forty in the sensitivity study, and the capstan friction coefficient, which is swept from
zero to 0.20. The sensitivity study exists because of this limitation and should be read as
the confidence interval on every other reported number.

### The settled grasp tension is above the quasi static chain

The measured tendon tension at the settled grasp is 124.1 N against a quasi static
prediction of 96.0 N. The difference is not numerical. The finger arrives at the object
carrying the kinetic energy of the rotor, the impact stretches the cord, and neither the
cord nor the stuck gearbox can push the stretch back out. The excess is bounded above by
the tension a cord absorbing the whole rotor kinetic energy would reach, which at the
400 rad/s closing speed limit is 59.9 N, and the test suite asserts that bound. Lowering
the closing speed limit reduces the excess and lengthens the closing time in proportion:
at 300 rad/s the excess falls to 21 N and the closing time rises from 0.80 s to 1.04 s.

### The contact surface is sampled, not continuous

Each phalanx is represented by three contact points. A phalanx lying flat against a surface
therefore produces three point loads rather than a distributed pressure. Total forces
converge quickly with the sample count because the contact law is nearly linear over the
indentation range used, but the reported force on an individual phalanx can move by a few
percent when the sampling changes. The fingertip force is reported as the total over the
distal phalanx rather than the force at one sample point for this reason.

### Planar, single finger, no thumb and no palm

The model is one finger in a plane. There is no thumb to oppose it and no palm for the
object to rest against, so the object is fixed in space and the reported forces are the
forces the finger applies to a held object rather than the forces of a complete grasp. A
full hand would add the load sharing between digits, which changes the current each motor
draws but not the transmission chain that this project is about.

### Gravity is off by default

The gravity vector is a parameter and defaults to zero. The finger weighs 40 g and its
weight torque about the proximal joint is at most 12 mNm, against a tendon torque of about
1 Nm at the grasp current, so gravity changes the settled posture by well under a degree.
It is switched off so that the reported metrics do not depend on hand orientation. The
gravity term is implemented and tested against the gradient of the potential energy.

### No backlash

The gearhead has 1.6 degrees of average no load backlash at the output, which is 0.4 mm of
tendon travel. The model has none. Backlash would add a dead zone to any position loop and
would show up as a delay between commanding a current and seeing a force. It does not
affect the settled grasp force, because the backlash is taken up once and stays taken up.

### The controller sees an ideal measurement

Current, motor angle, motor speed and fingertip force are read without noise, quantisation
or delay. A real drive measures current across a shunt at the pulse width modulation rate,
counts encoder edges, and reads a fingertip force sensor with its own bandwidth. Adding
those would lower the achievable loop gains, so the force control bandwidth reported here
should be read as an upper bound.

## Regression and tolerance policy

Two rules govern the numeric tests, both recorded here because they are easy to get wrong.

Only reproducible quantities are pinned. Closed form chain values, converged steady states,
integrated energy balances, counts and orderings are pinned. Raw late run state of a stiff
contact simulation is not, because a difference in the order a summation is reduced grows
over many steps into a visibly different answer, and a baseline recorded on one machine
then fails on another running the same code. The pinned grasp was checked at three
integration steps, 1e-4, 5e-5 and 2.5e-5, and agrees to one part in a thousand at all three.

Every tolerance is derived from a measurement scale. The two scales that appear are the
integration step together with the order of the scheme, which gives `(lambda h)^p` for a
dissipative run and `(lambda h)^p * lambda * T` for a conservative one, and the sample
interval of a recorded trace, which quantises any threshold crossing. Where an assertion
would otherwise sit exactly on its own quantisation boundary, the tolerance is doubled and
the doubling is stated in the test. The energy budget share test is the clearest case: the
shares miss unity by exactly the reported residual, so a tolerance equal to the residual
would be a coin toss on the last bit.
