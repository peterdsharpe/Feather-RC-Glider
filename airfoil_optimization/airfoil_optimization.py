from pymoo.algorithms.moo.nsga2 import NSGA2
# from pymoo.problems import get_problem
from pymoo.core.problem import Problem, ElementwiseProblem, ElementwiseEvaluationFunction
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.termination import get_termination

import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.geometry.airfoil import get_kulfan_coordinates


import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

# fig, ax = plt.subplots()

resolution = 20
Re = 35e3

def unpack_x(x):
    alpha = x[0]
    x_camber = x[1:resolution]
    x_thickness = x[resolution:]
    return alpha, x_camber, x_thickness


def pack_x(alpha, x_camber, x_thickness):
    return np.concatenate((
        [alpha],
        np.ones(resolution - 1) * x_camber,
        np.ones(resolution) * x_thickness,
    ))


def make_airfoil(x):
    alpha, x_camber, x_thickness = unpack_x(x)
    x_camber = np.concatenate(([0.], x_camber))
    return asb.Airfoil(
        coordinates=get_kulfan_coordinates(
            upper_weights=x_camber + x_thickness,
            lower_weights=x_camber - x_thickness,
            enforce_continuous_LE_radius=False,
            n_points_per_side=80,
        )
    )


class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=len(pack_x(0, 0, 0)),  # geometry + alpha
            n_obj=2,  # L/D, Cm,
            xl=pack_x(
                x_camber=-0.1,
                x_thickness=0,
                alpha=-5
            ),
            xu=pack_x(
                x_camber=0.40,
                x_thickness=0.40,
                alpha=15
            )
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # print(repr(x))

        alpha, x_camber, x_thickness = unpack_x(x)

        af = make_airfoil(x)

        # fig.clear()
        # af.draw()
        # plt.show()
        # plt.pause(1e-3)

        try:
            xf = asb.XFoil(
                airfoil=af,
                Re=Re,
                mach=0,
                xfoil_repanel=False,
                # verbose=True,
                max_iter=20
            )
            aero = xf.alpha(alpha=alpha)

            LD = aero["CL"] / aero["CD"]
            Cm = aero["CM"]
            if len(LD) == 0:
                raise FileNotFoundError
            try:
                LD = float(LD)
                Cm = float(Cm)
            except TypeError:
                print(repr(aero))
                raise TypeError

            print("\t", LD, Cm)
        except FileNotFoundError:
            LD = 0
            Cm = -1

        out["F"] = [-LD, -Cm]

res = minimize(
    problem=MyProblem(),
    algorithm=NSGA2(pop_size=30),
    termination=get_termination('time', "06:00:00"),
    seed=1,
    verbose=True,
)

fig, ax = plt.subplots()
LDs = -res.F[:, 0]
Cms = -res.F[:, 1]
plt.plot(Cms, LDs, ".")
p.show_plot(
    f"Airfoil Pareto @ $Re = {Re}",
    "$L/D$",
    "$C_m$"
)