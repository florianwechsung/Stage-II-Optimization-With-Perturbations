#!/usr/bin/env python
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux, FOCUSObjective
from simsopt.geo.curve import curves_to_vtk
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.curveobjectives import CurveLength, CoshCurveCurvature
from simsopt.geo.curveobjectives import MinimumDistance
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.surfaceobjectives import boozer_surface_residual, ToroidalFlux, Area
from simsopt.geo.qfmsurface import QfmSurface
from simsopt.geo.surfaceobjectives import QfmResidual, ToroidalFlux, Area, Volume
from objective import create_curves
from scipy.optimize import minimize
import argparse
import numpy as np

sigma = 1e-3

base_curves, base_currents, coils_fil, coils_fil_pert = create_curves(
    fil=0, ig=0, nsamples=1, stoch_seed=1, sigma=sigma)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 3)

errs = [2*np.pi*np.asarray(coils_fil[0].curve.quadpoints)]
titles = ["phi"]
coils = coils_fil_pert[0]
for j in range(4):
    shift = j
    for i in range(4):
        xyz = coils_fil[4*i + shift].curve.gamma()
        xyz_ = coils[4*i + shift].curve.gamma()
        rphiz = np.zeros_like(xyz)
        rphiz_ = np.zeros_like(xyz_)
        rphiz[:, 0] = np.linalg.norm(xyz[:, :2], axis=1)
        rphiz_[:, 0] = np.linalg.norm(xyz_[:, :2], axis=1)
        rphiz[:, 1] = np.arctan2(xyz[:, 1], xyz[:, 0])
        rphiz_[:, 1] = np.arctan2(xyz_[:, 1], xyz_[:, 0])
        rphiz[:, 2] = xyz[:, 2]
        rphiz_[:, 2] = xyz_[:, 2]
        sign = -1 if i%2 == 0 else 1
        axs[0].plot(rphiz[:, 0] - rphiz_[:, 0])
        axs[1].plot(sign*(rphiz[:, 1] - rphiz_[:, 1]))
        axs[2].plot(sign*(rphiz[:, 2] - rphiz_[:, 2]))
        errs.append(1000*(rphiz[:, 0] - rphiz_[:, 0]))
        errs.append(rphiz[:, 0]*1000*sign*(rphiz[:, 1] - rphiz_[:, 1])) # angular error must be multiplied with radius to obtain error
        errs.append(1000*sign*(rphiz[:, 2] - rphiz_[:, 2]))
        titles.append(f"err_r_{i}_{j}")
        titles.append(f"err_phi_{i}_{j}")
        titles.append(f"err_z_{i}_{j}")

np.savetxt("err.txt", np.asarray(errs).T, delimiter=";", header=";".join(titles), comments="")
# axs[0].set_ylim((-0.005, 0.005))
# axs[1].set_ylim((-0.005, 0.005))
# axs[2].set_ylim((-0.005, 0.005))
# plt.show()
