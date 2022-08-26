import aerosandbox as asb
import aerosandbox.numpy as np

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import pyvista as pv

import pandas as pd

df = pd.read_excel(
    "data.xlsx",
    skiprows=2
)

span = df["Span"].values
auw = df["AUW"].values / 1e3
mass_wing = df["Wing"].values / 1e3

# fit = asb.FittedModel(
#     model=lambda x, p: (
#         p["c"] *
#         x["m"] ** p["m"] *
#         x["s"] ** p["s"]
#     ),
#     x_data={
#         "s": span,
#         "m": auw - mass_wing,
#     },
#     y_data=mass_wing,
#     parameter_guesses={
#         "c": 1,
#         "m": 1,
#         "s": 1
#     },
#     put_residuals_in_logspace=True
# )
fit = asb.FittedModel(
    model=lambda x, p: (
        p["c"] *
        x ** p["s"]
    ),
    x_data=span,
    y_data=mass_wing,
    parameter_guesses={
        "c": 1,
        # "m": 1,
        "s": 1
    },
    put_residuals_in_logspace=True
)

# lambda