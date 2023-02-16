from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.util.ref_dirs import get_reference_directions
# from pymoo.problems import get_problem
from pymoo.core.problem import Problem, ElementwiseProblem, ElementwiseEvaluationFunction
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.termination import get_termination
# from dask.distributed import Client
import multiprocessing
from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization
from pymoo.core.callback import Callback

import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.geometry.airfoil import get_kulfan_coordinates

import matplotlib

matplotlib.rc('figure', max_open_warning=0)
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

op_point = asb.OperatingPoint(
    velocity=4.975614658280702
)
wing_mean_aerodynamic_chord = 0.11079155799884116
mass = 0.08152734674206971

resolution = 16
Re = op_point.reynolds(wing_mean_aerodynamic_chord)
n_procs = 16

CL_trim = (9.81 * mass) / (1 * wing_mean_aerodynamic_chord) / op_point.dynamic_pressure()

alpha_range = 2.5  # +- 2 deg
n_alphas = resolution

population_size = 32

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
            TE_thickness=400e-6 / 150e-3
        )
    )


class MyProblem(ElementwiseProblem):
    def __init__(self, **kwargs):
        super().__init__(
            n_var=len(pack_x(0, 0, 0)),  # geometry + alpha
            n_obj=2,  # L/D, Cm,
            xl=pack_x(
                alpha=0,
                x_camber=-0.4,
                x_thickness=np.concatenate((
                    [0.04],
                    -0.1 * np.ones(resolution - 1)
                )),
            ),
            xu=pack_x(
                alpha=12,
                x_camber=0.60,
                x_thickness=0.40,
            ),
            **kwargs
        )

    def _evaluate(self, x, out, debug=False, *args, **kwargs):

        with open("x.log", "a") as f:
            f.write("\nnp." + repr(x) + ',')

        alpha, x_camber, x_thickness = unpack_x(x)

        af = make_airfoil(x)

        try:
            alphas = np.linspace(alpha - alpha_range, alpha + alpha_range, n_alphas)

            xf = asb.XFoil(
                airfoil=af,
                Re=Re,
                mach=0,
                xfoil_repanel=False,
                # verbose=True,
                max_iter=100,
                timeout=10,
            )

            aero = xf.alpha(
                alpha=alphas,
                start_at=alpha
            )

            if debug:
                print(aero)

            aero['CD'] = np.where(
                aero["CDp"] <= 0,
                1,
                aero["CD"]
            )

            LDs = aero["CL"] / aero["CD"]
            Cms = aero["CM"]

            if len(LDs) == 0:
                raise Exception()

            n_nans = len(alphas) - len(LDs)

            LD = np.mean(np.concatenate((
                LDs,
                np.zeros(n_nans)
            )))
            Cm = np.mean(np.concatenate((
                Cms,
                -1 * np.ones(n_nans)
            )))

            LD = float(LD)
            Cm = float(Cm)

        except BaseException as e:
            print(e)
            LD = 0
            Cm = -10
            n_nans = len(alphas)

        print(f"\t\tLD = {LD:8.2f} | Cm = {Cm:8.3f} | n_nans = {n_nans}")

        out["F"] = [-LD, -Cm]


class MyCallback(Callback):
    def __init__(self):
        super().__init__()

    def notify(self, algorithm):
        if algorithm.n_iter % 10 == 1:
            fig, ax = plt.subplots()
            best_LD_x = algorithm.pop.get("X")[np.argmin(algorithm.pop.get("F")[:, 0]), :]
            af = make_airfoil(best_LD_x)
            af.draw()
            plt.show()


pool = ThreadPool(n_procs)
runner = StarmapParallelization(pool.starmap)

problem = MyProblem(
    elementwise_runner=runner
)

# algorithm = UNSGA3(
#     ref_dirs=get_reference_directions('das-dennis', 2, n_partitions=population_size),
# )

algorithm = NSGA2(pop_size=population_size)

termination = get_termination('time', "00:40:00")

callback = MyCallback()

if __name__ == '__main__':
    res = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=termination,  # 7:29
        callback=callback,
        seed=0,
        verbose=True,
        save_history=True,
    )

    order = np.argsort(-res.F[:, 0])
    afs = [
        make_airfoil(res.X[i, :])
        for i in order
    ]
    alphas = res.X[order, 0]
    LDs = -res.F[order, 0]
    Cms = -res.F[order, 1]

    with open("res.txt", "w+") as f:
        f.write("X = np." + repr(
            res.X[order, :]
        ))

    fig, ax = plt.subplots()
    plt.plot(Cms, LDs, ".")
    p.show_plot(
        f"Airfoil Pareto @ $Re = {Re:.0f}$",
        "$C_m$",
        "$L/D$",
    )

    fig, ax = plt.subplots(figsize=(9, 3))
    colors = plt.cm.get_cmap('rainbow')(np.linspace(0, 1, len(afs)))
    for af, color in zip(afs, colors):
        plt.plot(
            af.x(), af.y(), "-",
            color=color,
            alpha=0.3
        )
    p.equal()
    p.show_plot()

    fig, ax = plt.subplots()
    xf = asb.XFoil(
        airfoil=afs[-1],

    )