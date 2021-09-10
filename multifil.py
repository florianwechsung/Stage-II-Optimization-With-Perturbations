#!/usr/bin/env python
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux, FOCUSObjective
from simsopt.geo.curve import curves_to_vtk
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.curveobjectives import CurveLength, CoshCurveCurvature
from simsopt.geo.curveobjectives import MinimumDistance
from objective import create_curves
from scipy.optimize import minimize
import argparse
import numpy as np
from pathlib import Path
TEST_DIR = (Path(__file__).parent / ".." / "simsopt" / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'
from mpi4py import MPI
comm = MPI.COMM_WORLD
import logging
logger = logging.getLogger("SIMSOPT-STAGE2")
handler = logging.StreamHandler()
formatter = logging.Formatter(fmt="%(levelname)s %(message)s")
handler.setFormatter(formatter)
if comm is not None and comm.rank != 0:
    handler = logging.NullHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False



parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.)
parser.add_argument("--fil", type=int, default=0)
parser.add_argument("--ig", type=int, default=0)
parser.add_argument("--nsamples", type=int, default=0)
parser.add_argument("--sigma", type=float, default=0.001)
args = parser.parse_args()


import os
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']

"""
In this example we solve a FOCUS like Stage II coil optimisation problem: the
goal is to find coils that generate a specific target normal field on a given
surface.  In this particular case we consider a vacuum field, so the target is
just zero.

The objective is given by

    J = \int |Bn| ds + alpha * (sum CurveLength) + beta * MininumDistancePenalty

if alpha or beta are increased, the coils are more regular and better
separated, but the target normal field may not be achieved as well.

The target equilibrium is the QA configuration of arXiv:2108.03711.
"""

nfp = 2
nphi = 64
ntheta = 64
phis = np.linspace(0, 1./(2*nfp), nphi, endpoint=False)
thetas = np.linspace(0, 1., ntheta, endpoint=False)
s = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phis, quadpoints_theta=thetas)


ALPHA = args.alpha
MIN_DIST = 0.1
DIST_ALPHA = 10.
BETA = 10
MAXITER = 50 if ci else 1000
KAPPA_MAX = 10.
KAPPA_ALPHA = 1.
KAPPA_WEIGHT = .1

outdir = f"output_alpha_{ALPHA}_fil_{args.fil}_ig_{args.ig}_samples_{args.nsamples}/"
os.makedirs(outdir, exist_ok=True)

base_curves, base_currents, coils_fil, coils_fil_pert = create_curves(fil=args.fil, ig=args.ig, nsamples=args.nsamples, stoch_seed=0, sigma=args.sigma)

bs = BiotSavart(coils_fil)
bs.set_points(s.gamma().reshape((-1, 3)))

pointData = {"B_N": np.sum(bs.B().reshape(s.gamma().shape) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(outdir + "surf_init", extra_data=pointData)
curves_rep = [c.curve for c in coils_fil]

curves_to_vtk(curves_rep, outdir + "curves_init")

Jls = [CurveLength(c) for c in base_curves]
Jdist = MinimumDistance(base_curves, MIN_DIST, penalty_type="cosh", alpha=DIST_ALPHA)
Jf = SquaredFlux(s, bs)

Jfs = [SquaredFlux(s, BiotSavart(cs)) for cs in coils_fil_pert]

if args.nsamples > 0:
    from objective import MPIObjective
    Jmpi = MPIObjective(Jfs, comm)
    Jmpi.J()
    JF = FOCUSObjective([Jmpi], Jls, ALPHA, Jdist, BETA)
else:
    JF = FOCUSObjective(Jf, Jls, ALPHA, Jdist, BETA)

Jkappas = [CoshCurveCurvature(c, kappa_max=KAPPA_MAX, alpha=KAPPA_ALPHA) for c in base_curves]


history = []
ctr = [0]


def cb(x):
    ctr[0] += 1


# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize
def fun(dofs):
    Jf.x = dofs
    JF.x = dofs
    J = JF.J() + KAPPA_WEIGHT*sum(Jk.J() for Jk in Jkappas)
    dJ = JF.dJ()
    for Jk in Jkappas:
        dJ += KAPPA_WEIGHT * Jk.dJ()
    grad = dJ(JF)
    cl_string = ", ".join([f"{J.J():.3f}" for J in Jls])
    mean_AbsB = np.mean(bs.AbsB())
    jf = Jf.J()
    s = f"{ctr[0]}, J={J:.3e}, Jflux={jf:.3e}, sqrt(Jflux)/Mean(|B|)={np.sqrt(jf)/mean_AbsB:.3e}, CoilLengths=[{cl_string}], ||∇J||={np.linalg.norm(grad):.3e}"
    if args.nsamples > 0:
        s += f", {Jmpi.J():.3e}"
    logger.info(s)
    history.append((J, jf, np.linalg.norm(grad)))
    return J, grad


logger.info(f"Curvatures {[np.max(c.kappa()) for c in base_curves]}")
logger.info(f"Shortest distance {Jdist.shortest_distance()}")
logger.info("""
################################################################################
### Perform a Taylor test ######################################################
################################################################################
""")
f = fun
dofs = JF.x
np.random.seed(1)
h = np.random.uniform(size=dofs.shape)
J0, dJ0 = f(dofs)
dJh = sum(dJ0 * h)
for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    J1, _ = f(dofs + eps*h)
    J2, _ = f(dofs - eps*h)
    logger.info(f"err {(J1-J2)/(2*eps) - dJh}")

logger.info("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")
curiter = 0
tries = 0
while MAXITER-curiter > 0 and tries < 10:
    # res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER-curiter, 'maxcor': 400}, tol=1e-15, callback=cb)
    res = minimize(fun, dofs, jac=True, method='BFGS', options={'maxiter': MAXITER-curiter}, tol=1e-15, callback=cb)
    dofs = res.x
    curiter += res.nit
    tries += 1
meanB = np.mean(bs.AbsB())

curves_to_vtk(curves_rep, outdir + "curves_opt")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(outdir + "surf_opt", extra_data=pointData)
kappas = [np.max(c.kappa()) for c in base_curves]
arclengths = [np.min(c.incremental_arclength()) for c in base_curves]
dist = Jdist.shortest_distance()
logger.info(f"Curvatures {kappas}")
logger.info(f"Arclengths {arclengths}")
logger.info(f"Shortest distance {dist}")
np.savetxt(outdir + "kappas.txt", kappas)
np.savetxt(outdir + "arclengths.txt", arclengths)
np.savetxt(outdir + "dist.txt", [dist])
np.savetxt(outdir + "length.txt", [J.J() for J in Jls])
np.savetxt(outdir + "xmin.txt", JF.x)
for i in range(len(base_curves)):
    np.savetxt(outdir + f"curve_{i}.txt", base_curves[i].x)
np.savetxt(outdir + "history.txt", history)
logger.info(outdir)


logger.info("""
################################################################################
### Perform a Taylor test ######################################################
################################################################################
""")
f = fun
dofs = JF.x
np.random.seed(1)
h = np.random.uniform(size=dofs.shape)
J0, dJ0 = f(dofs)
dJh = sum(dJ0 * h)
for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    J1, _ = f(dofs + eps*h)
    J2, _ = f(dofs - eps*h)
    logger.info(f"err {(J1-J2)/(2*eps) - dJh}")
