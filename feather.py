import aerosandbox as asb
import aerosandbox.numpy as np
import aerosandbox.tools.units as u
import pandas as pd
import copy

opti = asb.Opti(
    # variable_categories_to_freeze="all",
    # freeze_style="float"
)

make_plots = True

##### Section: Parameters

# wing_method = '3d printing'
wing_method = 'foam'

wing_span = 1
wing_dihedral_angle_deg = 11
vtail_dihedral_angle_deg = 37

airfoils = {
    name: asb.Airfoil(
        name=name,
    ) for name in [
        "ag04",
        # "ag09",
        "ag13",
        "naca0008"
    ]
}

for v in airfoils.values():
    v.generate_polars(
        cache_filename=f"cache/{v.name}.json",
        alphas=np.linspace(-10, 10, 21)
    )

##### Section: Vehicle Overall Specs

op_point = asb.OperatingPoint(
    velocity=opti.variable(
        init_guess=14,
        lower_bound=1,
        log_transform=True
    ),
    alpha=opti.variable(
        init_guess=0,
        lower_bound=-10,
        upper_bound=10
    )
)

design_mass_TOGW = opti.variable(
    init_guess=0.1,
    lower_bound=1e-3
)
design_mass_TOGW = np.maximum(design_mass_TOGW, 1e-3)

LD_cruise = opti.variable(
    init_guess=15,
    lower_bound=0.1,
    log_transform=True
)

g = 9.81

target_climb_angle = 45  # degrees

thrust_cruise = (
        design_mass_TOGW * g / LD_cruise
)

thrust_climb = (
        design_mass_TOGW * g / LD_cruise +
        design_mass_TOGW * g * np.sind(target_climb_angle)
)

##### Section: Vehicle Definition

"""
Coordinate system:

Geometry axes. Datum (0, 0, 0) is coincident with the quarter-chord-point of the centerline cross section of the main 
wing.

"""

### Define x-stations
x_nose = opti.variable(
    init_guess=-0.1,
    upper_bound=1e-3,
)
x_tail = opti.variable(
    init_guess=0.7,
    lower_bound=1e-3
)

### Wing
wing_root_chord = opti.variable(
    init_guess=0.15,
    lower_bound=1e-3
)


def wing_rot(xyz):
    dihedral_rot = np.rotation_matrix_3D(
        angle=np.radians(wing_dihedral_angle_deg),
        axis="X"
    )

    return dihedral_rot @ np.array(xyz)


def wing_chord(y):
    spanfrac = y / (wing_span / 2)
    chordfrac = 1 - 0.4 * spanfrac - 0.47 * (1 - (1 - spanfrac ** 2 + 1e-16) ** 0.5)

    # c_over_c_root = 0.1 + 0.9 * (1 - (y / half_span) ** 2) ** 0.5
    return chordfrac * wing_root_chord


def wing_twist(y):
    return np.zeros_like(y)


wing_ys = np.sinspace(
    0,
    wing_span / 2,
    11,
    reverse_spacing=True
)

wing = asb.Wing(
    name="Main Wing",
    symmetric=True,
    xsecs=[
        asb.WingXSec(
            xyz_le=wing_rot([
                -wing_chord(wing_ys[i]),
                wing_ys[i],
                0
            ]),
            chord=wing_chord(wing_ys[i]),
            airfoil=airfoils["ag13"],
            twist=wing_twist(wing_ys[i]),
        )
        for i in range(np.length(wing_ys))
    ]
).translate([
    0.75 * wing_root_chord,
    0,
    0
])

vtail_twist = opti.variable(init_guess=0, lower_bound=-15, upper_bound=15)

vtail_shape_data = pd.DataFrame(
    {
        "x_le_inches" : [0, 0.28, 0.63, 0.93, 1.25, 1.75],
        "y_le_inches" : [0, 2, 4, 5, 5.58, 6],
        "chord_inches": [2.75, 2.50, 2.06, 1.72, 1.36, 0.75]
    }
)


def vtail_rot(xyz):
    dihedral_rot = np.rotation_matrix_3D(
        angle=np.radians(vtail_dihedral_angle_deg),
        axis="X"
    )

    return dihedral_rot @ np.array(xyz)


vtail = asb.Wing(
    name="VTail",
    symmetric=True,
    xsecs=[
        asb.WingXSec(
            xyz_le=vtail_rot([
                row["x_le_inches"] * u.inch,
                row["y_le_inches"] * u.inch,
                0
            ]),
            chord=row["chord_inches"] * u.inch,
            twist=vtail_twist,
            airfoil=airfoils["naca0008"]
        )
        for i, row in vtail_shape_data.iterrows()
    ]
).translate([
    x_tail - 0.7 * vtail_shape_data.iloc[0, :]["chord_inches"] * u.inch,
    0,
    0
])

fuselage = asb.Fuselage(
    name="Fuse",
    xsecs=[
        asb.FuselageXSec(
            xyz_c=[x_nose, 0, 0],
            radius=7e-3 / 2
        ),
        asb.FuselageXSec(
            xyz_c=[x_tail, 0, 0],
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

##### Section: Internal Geometry and Weights

mass_props = {}

### Lifting bodies
if wing_method == '3d printing':
    # mass_props['wing'] = asb.mass_properties_from_radius_of_gyration(
    #     mass=(12e-3 / (0.200 * 0.150)) * wing.area(),
    #     x_cg=
    # )
    raise ValueError

elif wing_method == 'foam':
    # density = 20.8 # kg/m^3, for Foamular 150
    # density = 38.06 # kg/m^3, for hard blue foam
    # density = 4 * u.lbm / u.foot ** 3
    density = 2 * u.lbm / u.foot ** 3
    mass_props['wing'] = asb.mass_properties_from_radius_of_gyration(
        mass=wing.volume() * density,
        x_cg=(0.500 - 0.25) * wing_root_chord,
        z_cg=(0.03591) * (
                np.sind(wing_dihedral_angle_deg) / np.sind(11)
        ) * (
                     wing_span / 1
             ),
    )
elif wing_method == 'elf':
    # wing_mass = asb.MassProperties(
    #     mass=0.0506 * wing_span ** 2.09,
    #     x_cg=0.5 * root_chord
    # )
    raise ValueError

mass_props["tail_surfaces"] = asb.mass_properties_from_radius_of_gyration(
    mass=vtail.volume() * 80,
    x_cg=vtail.xsecs[0].xyz_le[0] + 0.50 * vtail.xsecs[0].chord,
)

mass_props["linkages"] = asb.MassProperties(
    mass=1e-3,
    x_cg=x_tail / 2
)

### Pod and avionics

mass_props["motor"] = asb.mass_properties_from_radius_of_gyration(
    mass=4.49e-3,  # EX1103 - kv 6000, weight includes 3-phase wiring
    x_cg=x_nose - 0.3 * u.inch
)
mass_props["motor_bolts"] = asb.mass_properties_from_radius_of_gyration(
    mass=4 * 0.075e-3,
    x_cg=x_nose
)

mass_props["propeller"] = asb.mass_properties_from_radius_of_gyration(
    mass=1.54e-3,
    x_cg=x_nose - 0.7 * u.inch
)

mass_props["propeller_band"] = asb.mass_properties_from_radius_of_gyration(
    mass=0.06e-3,
    x_cg=mass_props["propeller"].x_cg
)

mass_props["flight_computer"] = asb.mass_properties_from_radius_of_gyration(
    mass=4.30e-3,
    x_cg=x_nose + 2 * u.inch + (1.3 * u.inch) / 2
)  # Includes RX, 2 linear servos, ESC, BLDC motor, prop
# https://www.buzzardmodels.com/4gram1spro-brick

mass_props["battery"] = asb.mass_properties_from_radius_of_gyration(
    mass=4.61e-3,
    x_cg=x_nose + 2 * u.inch
)

mass_props["pod"] = asb.MassProperties(
    mass=7e-3,
    x_cg=(x_nose + 0.75 * wing_root_chord) / 2
)

mass_props["ballast"] = asb.mass_properties_from_radius_of_gyration(
    mass=opti.variable(init_guess=0, lower_bound=0),
    x_cg=opti.variable(init_guess=0, lower_bound=x_nose, upper_bound=x_tail),
)

### Boom

mass_props["boom"] = asb.mass_properties_from_radius_of_gyration(
    mass=7.0e-3 * ((x_tail - x_nose) / 826e-3),
    x_cg=(x_nose + x_tail) / 2
)  # https://www.buzzardmodels.com/p000

### Summation
mass_props_TOGW = asb.MassProperties(mass=0)
for k, v in mass_props.items():
    mass_props_TOGW = mass_props_TOGW + v

### Add glue weight
mass_props['glue_weight'] = mass_props_TOGW * 0.08
mass_props_TOGW += mass_props['glue_weight']

##### Section: Aerodynamics

ab = asb.AeroBuildup(
    airplane=airplane,
    op_point=op_point,
    xyz_ref=mass_props_TOGW.xyz_cg
)
aero = ab.run_with_stability_derivatives(
    alpha=True,
    beta=True,
    p=False,
    q=False,
    r=False,
)

opti.subject_to([
    aero["L"] >= 9.81 * mass_props_TOGW.mass,
    aero["Cm"] == 0,
])

LD = aero["L"] / aero["D"]
power_loss = aero["D"] * op_point.velocity
sink_rate = power_loss / 9.81 / mass_props_TOGW.mass

##### Section: Stability
static_margin = (aero["x_np"] - mass_props_TOGW.x_cg) / wing.mean_aerodynamic_chord()

opti.subject_to(
    static_margin == 0.08
)

##### Section: Finalize Optimization Problem
objective = sink_rate / 0.3 + mass_props_TOGW.mass / 0.100 * 0.1
penalty = (mass_props["ballast"].x_cg / 1e3) ** 2

opti.minimize(objective + penalty)

### Additional constraint
opti.subject_to([
    LD_cruise == LD,
    design_mass_TOGW == mass_props_TOGW.mass
])

opti.subject_to([
    x_nose < -0.25 * wing_root_chord - 0.5 * u.inch,  # propeller must extend in front of wing
    x_tail - x_nose < 0.826,  # due to the length of carbon tube I have
    vtail.area() * np.cosd(vtail_dihedral_angle_deg) ** 2 * x_tail / (
            wing.area() * wing.mean_aerodynamic_chord()) > 0.25
])

if __name__ == '__main__':
    try:
        sol = opti.solve()
    except RuntimeError:
        sol = opti.debug
    s = lambda x: sol.value(x)

    airplane = sol(airplane)
    op_point = sol(op_point)
    mass_props = sol(mass_props)
    mass_props_TOGW = sol(mass_props_TOGW)
    aero = sol(aero)

    wing_lowres = copy.deepcopy(wing)
    xsecs_to_keep = np.arange(len(wing.xsecs)) % 2 == 0
    xsecs_to_keep[0] = True
    xsecs_to_keep[-1] = True
    wing_lowres.xsecs = np.array(wing_lowres.xsecs)[xsecs_to_keep]

    try:
        avl_aero = asb.AVL(
            airplane=asb.Airplane(
                wings=[
                    wing_lowres,
                    vtail,
                ],
                fuselages=[fuselage]
            ),
            op_point=op_point,
            xyz_ref=mass_props_TOGW.xyz_cg,
            working_directory=r"C:\Users\peter\Downloads\avl_debug"
        ).run()
    except FileNotFoundError:
        class EmptyDict:
            def __getitem__(self, item):
                return "Install AVL to see this."


        avl_aero = EmptyDict()

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p
    from aerosandbox.tools.string_formatting import eng_string

    ##### Section: Printout
    print_title = lambda s: print(s.upper().join(["*" * 20] * 2))


    def fmt(x):
        return f"{s(x):.6g}"


    print_title("Outputs")
    for k, v in {
        "mass_TOGW"             : f"{fmt(mass_props_TOGW.mass)} kg ({fmt(mass_props_TOGW.mass / u.lbm)} lbm)",
        "L/D (actual)"          : fmt(LD_cruise),
        "Cruise Airspeed"       : f"{fmt(op_point.velocity)} m/s",
        "Cruise AoA"            : f"{fmt(op_point.alpha)} deg",
        "Cruise CL"             : fmt(aero['CL']),
        "Sink Rate"             : f"{fmt(sink_rate)} m/s",
        "Cma"                   : fmt(aero['Cma']),
        "Cnb"                   : fmt(aero['Cnb']),
        "Cm"                    : fmt(aero['Cm']),
        "Wing Reynolds Number"  : eng_string(op_point.reynolds(sol(wing.mean_aerodynamic_chord()))),
        "AVL: Cma"              : avl_aero['Cma'],
        "AVL: Cnb"              : avl_aero['Cnb'],
        "AVL: Cm"               : avl_aero['Cm'],
        "AVL: Clb Cnr / Clr Cnb": avl_aero['Clb Cnr / Clr Cnb'],
        "CG location"           : "(" + ", ".join([fmt(xyz) for xyz in mass_props_TOGW.xyz_cg]) + ") m",
        "Wing Span"             : f"{fmt(wing_span)} m ({fmt(wing_span / u.foot)} ft)",
    }.items():
        print(f"{k.rjust(25)} = {v}")

    fmtpow = lambda x: fmt(x) + " W"

    print_title("Mass props")
    for k, v in mass_props.items():
        print(f"{k.rjust(25)} = {v.mass * 1e3:.2f} g ({v.mass / u.oz:.2f} oz)")

    if make_plots:
        ##### Section: Geometry
        airplane.draw_three_view(show=False)
        p.show_plot(tight_layout=False, savefig="figures/three_view.png")

        ##### Section: Mass Budget
        fig, ax = plt.subplots(figsize=(12, 5), subplot_kw=dict(aspect="equal"), dpi=300)

        name_remaps = {
            **{
                k: k.replace("_", " ").title()
                for k in mass_props.keys()
            },
        }

        mass_props_to_plot = copy.deepcopy(mass_props)
        if mass_props_to_plot["ballast"].mass < 1e-6:
            mass_props_to_plot.pop("ballast")
        p.pie(
            values=[
                v.mass
                for v in mass_props_to_plot.values()
            ],
            names=[
                n if n not in name_remaps.keys() else name_remaps[n]
                for n in mass_props_to_plot.keys()
            ],
            center_text=f"$\\bf{{Mass\\ Budget}}$\nTOGW: {s(mass_props_TOGW.mass * 1e3):.2f} g",
            label_format=lambda name, value, percentage: f"{name}, {value * 1e3:.2f} g, {percentage:.1f}%",
            startangle=110,
            arm_length=30,
            arm_radius=20,
            y_max_labels=1.1
        )
        p.show_plot(savefig="figures/mass_budget.png")
