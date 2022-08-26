import aerosandbox as asb
import aerosandbox.numpy as np
import aerosandbox.tools.units as u

opti = asb.Opti()

wing_method = '3d printing'

# span = 999e-3
# span = opti.variable(init_guess=1, lower_bound=0.1)
span = opti.parameter(1.000)
# root_chord = opti.parameter(156.5e-3) #156.5e-3
root_chord = opti.variable(156.5e-3, lower_bound=0) #156.5e-3
# wing_airfoil = asb.Airfoil("s1091")
wing_airfoil = asb.Airfoil("ag13")
wing_airfoil.generate_polars(
    cache_filename=f"cache/{wing_airfoil.name}.json",
    xfoil_kwargs=dict(
        xfoil_repanel=False
    )
)

tail_airfoil = asb.Airfoil("naca0008")
tail_airfoil.generate_polars(
    cache_filename=f"cache/{tail_airfoil.name}.json",
    xfoil_kwargs=dict(
        xfoil_repanel=True
    )
)


def chord(y):
    spanfrac = y / (span / 2)
    chordfrac = 1 - 0.4 * spanfrac - 0.47 * (1 - (1 - spanfrac ** 2 + 1e-16) ** 0.5)

    # c_over_c_root = 0.1 + 0.9 * (1 - (y / half_span) ** 2) ** 0.5
    return chordfrac * root_chord


def z(y):
    spanfrac = y / (span / 2)
    return y * np.tand(11)
    # return spanfrac ** 2 * np.tand(12)
    # return np.softmax(0, 2 * spanfrac - 1, hardness=7) * np.tand(6)


ys = np.sinspace(0, span / 2, 10, reverse_spacing=True)
cs = chord(ys)
zs = z(ys)

wing = asb.Wing(
    name="Main Wing",
    symmetric=True,
    xsecs=[
        asb.WingXSec(
            xyz_le=[cs[0] - cs[i], ys[i], zs[i]],
            chord=cs[i],
            airfoil=wing_airfoil,
        )
        for i in range(np.length(ys))
    ]
)

nose_length = 4 * u.inch  # Wing LE to Nose
tail_length = 19 * u.inch  # Wing LE to Tail LE

vtail = asb.Wing(
    name="VTail",
    symmetric=True,
    xsecs=[
        asb.WingXSec(
            chord=2.75 * u.inch,
        ),
        asb.WingXSec(
            xyz_le=[0.28 * u.inch, 2 * u.inch, 0],
            chord=2.50 * u.inch,
        ),
        asb.WingXSec(
            xyz_le=[0.63 * u.inch, 4 * u.inch, 0],
            chord=2.06 * u.inch,
        ),
        asb.WingXSec(
            xyz_le=[0.93 * u.inch, 5 * u.inch, 0],
            chord=1.72 * u.inch,
        ),
        asb.WingXSec(
            xyz_le=[1.23 * u.inch, 5.58 * u.inch, 0],
            chord=1.36 * u.inch,
        ),
        asb.WingXSec(
            xyz_le=[1.75 * u.inch, 6 * u.inch, 0],
            chord=0.75 * u.inch,
        )
    ]
).translate([
    tail_length,
    0,
    0
])
# vtail_twist=-7
vtail_twist = opti.variable(init_guess=0, lower_bound=-30, upper_bound=30)

for i in range(len(vtail.xsecs)):
    xyz_le_flat = vtail.xsecs[i].xyz_le
    vtail.xsecs[i].xyz_le = np.array([
        xyz_le_flat[0],
        np.cosd(37) * xyz_le_flat[1],
        np.sind(37) * xyz_le_flat[1]
    ])
    vtail.xsecs[i].airfoil = tail_airfoil
    vtail.xsecs[i].twist=vtail_twist

fuselage = asb.Fuselage(
    name="Fuse",
    xsecs=[
        asb.FuselageXSec(
            xyz_c=[-nose_length, 0, 0],
            radius=7e-3 / 2
        ),
        asb.FuselageXSec(
            xyz_c=[tail_length, 0, 0],
            radius=7e-3 / 2
        )
    ]
)

airplane = asb.Airplane(
    name="Feather",
    wings=[
        wing,
        vtail
    ],
    fuselages=[fuselage]
)

if wing_method == '3d printing':
    wing_mass = asb.MassProperties(
        mass=(14e-3 / (0.200 * 0.150)) * wing.area()
    )
elif wing_method == 'elf':
    wing_mass = asb.MassProperties(
        mass=0.0506 * span ** 2.09,
        x_cg=0.5 * root_chord
    )

tail_mass = asb.MassProperties(
    mass=5e-3 * span ** 2.09,
    x_cg=vtail.xsecs[0].xyz_le[0] + vtail.xsecs[0].chord / 2
)

pod_mass = asb.MassProperties(
    mass=7e-3,
    x_cg=(-nose_length + wing.xsecs[0].chord) / 2
)

control_rods_mass = asb.MassProperties(
    mass=1e-3,
    x_cg=tail_length / 2
)

avionics_mass = asb.MassProperties(
    mass=11.00e-3,
    x_cg=-nose_length + 1.5 * u.inch
)  # Includes RX, 2 linear servos, ESC, BLDC motor, prop
# https://www.buzzardmodels.com/4gram1spro-brick

battery_mass = asb.MassProperties(
    mass=4.56e-3,
    x_cg=-20e-3
)  # 1S 150 mAh

boom_mass = asb.MassProperties(
    mass=7.0e-3 * ((nose_length + tail_length) / 826e-3) * span,
    x_cg=826e-3 / 2
)  # https://www.buzzardmodels.com/p000

ballast_mass = asb.MassProperties(
    mass=opti.variable(init_guess=0., lower_bound=0.,)
)

mass_props = (
        wing_mass +
        tail_mass +
        pod_mass +
        control_rods_mass +
        avionics_mass +
        battery_mass +
        boom_mass +
        ballast_mass
)

airspeed = opti.variable(init_guess=5, lower_bound=0)
alpha = opti.variable(init_guess=4.5, lower_bound=-15, upper_bound=15)
op_point = asb.OperatingPoint(
    velocity=airspeed,
    alpha=alpha
)

analysis = asb.AeroBuildup(
    airplane,
    op_point,
    xyz_ref=mass_props.xyz_cg
)
aero = analysis.run()

opti.subject_to([
    aero["L"] == 9.81 * mass_props.mass,
    aero["Cm"] == 0
])

LD = aero["L"] / aero["D"]
power_loss = aero["D"] * op_point.velocity
sink_rate = power_loss / 9.81 / mass_props.mass

opti.minimize(sink_rate)


# try:
#     sol = opti.solve()
# except RuntimeError as e:
#     print(e)
#     sol = opti.debug
# s = lambda x: sol.value(x)

@np.vectorize
def get_sols(input_value=None):
    if input_value is not None:
        opti.set_value(root_chord, input_value)
    try:
        sol = opti.solve(verbose=False, max_iter=50)
        opti.set_initial_from_sol(sol)
    except RuntimeError:
        sol = opti.debug
    return sol

sol = get_sols()

# inputs = np.linspace(0.05, 0.4, 10)
# sols = get_sols(inputs)

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

def s(x):
    return sol.value(x)

def ss(x):
    return np.array([sol.value(x) if sol.stats()['success'] else np.nan for sol in sols])

def qp(*args):
    p.qp(*[ss(x) for x in args], stacklevel=2)

# qp(root_chord, sink_rate)

