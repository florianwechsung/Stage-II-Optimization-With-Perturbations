#!/usr/bin/env python
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux, FOCUSObjective
from simsopt.geo.curve import curves_to_vtk
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.curveobjectives import CurveLength, CoshCurveCurvature
from simsopt.geo.curveobjectives import MinimumDistance
from objective import create_curves, CoshCurveLength
from scipy.optimize import minimize
import argparse
import os
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
parser.add_argument("--lengthbound", type=float, default=0.)
args = parser.parse_args()
if args.nsamples == 0:
    args.sigma = 0.


def set_file_logger(path):
    from math import log10, ceil
    digits = ceil(log10(comm.size))
    filename, file_extension = os.path.splitext(path)
    fileHandler = logging.FileHandler(filename + "-rank" + ("%i" % comm.rank).zfill(digits) + file_extension, mode='a')
    formatter = logging.Formatter(fmt="%(asctime)s:%(name)s:%(levelname)s %(message)s")
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)


"""
The target equilibrium is the QA configuration of arXiv:2108.03711.
"""

nfp = 2
nphi = 64
ntheta = 64
phis = np.linspace(0, 1./(2*nfp), nphi, endpoint=False)
thetas = np.linspace(0, 1., ntheta, endpoint=False)
s = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phis, quadpoints_theta=thetas)


MAXITER = 5000
ALPHA = args.alpha

MIN_DIST = 0.1
DIST_ALPHA = 10.
DIST_WEIGHT = 1

KAPPA_MAX = 10.
KAPPA_ALPHA = 1.
KAPPA_WEIGHT = .1

LENGTH_CON_ALPHA = 0.1
LENGTH_CON_WEIGHT = 1

outdir = f"output_temp/lengthbound_{args.lengthbound}_alpha_{ALPHA}_fil_{args.fil}_ig_{args.ig}_samples_{args.nsamples}_sigma_{args.sigma}/"
os.makedirs(outdir, exist_ok=True)
set_file_logger(outdir + "log.txt")

base_curves, base_currents, coils_fil, coils_fil_pert = create_curves(fil=args.fil, ig=args.ig, nsamples=args.nsamples, stoch_seed=0, sigma=args.sigma)

bs = BiotSavart(coils_fil)
bs.set_points(s.gamma().reshape((-1, 3)))

pointData = {"B_N": np.sum(bs.B().reshape(s.gamma().shape) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(outdir + "surf_init", extra_data=pointData)
curves_rep = [c.curve for c in coils_fil]
NFIL = (2*args.fil + 1)**2
curves_rep_no_fil = [curves_rep[NFIL//2 + i*NFIL] for i in range(len(curves_rep)//NFIL)]

curves_to_vtk(curves_rep, outdir + "curves_init")

Jls = [CurveLength(c) for c in base_curves]
Jlconstraint = CoshCurveLength(Jls, args.lengthbound, LENGTH_CON_ALPHA)
Jdist = MinimumDistance(curves_rep_no_fil, MIN_DIST, penalty_type="cosh", alpha=DIST_ALPHA)
Jf = SquaredFlux(s, bs)

Jfs = [SquaredFlux(s, BiotSavart(cs)) for cs in coils_fil_pert]

if args.nsamples > 0:
    from objective import MPIObjective
    Jmpi = MPIObjective(Jfs, comm)
    Jmpi.J()
    JF = FOCUSObjective([Jmpi], Jls, ALPHA, Jdist, DIST_WEIGHT)
else:
    JF = FOCUSObjective(Jf, Jls, ALPHA, Jdist, DIST_WEIGHT)

Jkappas = [CoshCurveCurvature(c, kappa_max=KAPPA_MAX, alpha=KAPPA_ALPHA) for c in base_curves]


history = []
ctr = [0]


def cb(*args):
    ctr[0] += 1


# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize
def fun(dofs, silent=False):
    Jf.x = dofs
    JF.x = dofs
    J = JF.J() + KAPPA_WEIGHT*sum(Jk.J() for Jk in Jkappas)
    dJ = JF.dJ(partials=True)
    for Jk in Jkappas:
        dJ += KAPPA_WEIGHT * Jk.dJ(partials=True)
    if args.lengthbound > 0:
        J += LENGTH_CON_WEIGHT * Jlconstraint.J()
        dJ += LENGTH_CON_WEIGHT * Jlconstraint.dJ()
    grad = dJ(JF)
    cl_string = ", ".join([f"{J.J():.3f}" for J in Jls])
    totalcl = sum([J.J() for J in Jls])
    mean_AbsB = np.mean(bs.AbsB())
    jf = Jf.J()
    s = f"{ctr[0]}, J={J:.3e}, Jflux={jf:.3e}, sqrt(Jflux)/Mean(|B|)={np.sqrt(jf)/mean_AbsB:.3e}, CoilLengths=[{cl_string}]={totalcl:.3e}, ||∇J||={np.linalg.norm(grad):.3e}"
    if args.nsamples > 0:
        s += f", {Jmpi.J():.3e}"
    if not silent:
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
outeriter = 0
MAXLOCALITER = MAXITER//4
while MAXITER-curiter > 0 and outeriter < 10:
    if outeriter > 0:
        LENGTH_CON_WEIGHT *= 10
        KAPPA_WEIGHT *= 10
        JF.beta *= 10
    # res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER-curiter, 'maxcor': 400}, tol=1e-15, callback=cb)
    res = minimize(fun, dofs, jac=True, method='BFGS', options={'maxiter': min(MAXLOCALITER, MAXITER-curiter)}, tol=1e-15, callback=cb)

    dofs = res.x
    curiter += res.nit
    outeriter += 1


def approx_H(x):
    n = x.size
    H = np.zeros((n, n))
    x0 = x
    eps = 1e-4
    for i in range(n):
        x = x0.copy()
        x[i] += eps
        d1 = fun(x, silent=True)[1]
        x[i] -= 2*eps
        d0 = fun(x, silent=True)[1]
        H[i, :] = (d1-d0)/(2*eps)
    H = 0.5 * (H+H.T)
    return H


from scipy.linalg import eigh
x = dofs
f, d = fun(x)
for i in range(10):
    H = approx_H(x)
    D, E = eigh(H)
    bestd = np.inf
    bestx = None
    # Computing the Hessian is the most expensive thing, so we can be pretty
    # naive with the next step and just try a whole bunch of damping parameters
    # and step sizes and then take the one with smallest gradient norm that
    # still decreases the objective
    for lam in [1e-5, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]:
        Dm = np.abs(D) + lam
        dx = E @ np.diag(1./Dm) @ E.T @ d
        alpha = 1.
        for j in range(5):
            xnew = x - alpha * dx
            fnew, dnew = fun(xnew, silent=True)
            dnormnew = np.linalg.norm(dnew)
            foundnewbest = ""
            if fnew < f and dnormnew < bestd:
                bestd = dnormnew
                bestx = xnew
                foundnewbest = "x"
            logger.info(f'Linesearch: lam={lam:.5f}, alpha={alpha:.4f}, J(xnew)={fnew:.15f}, |dJ(xnew)|={dnormnew:.3e}, {foundnewbest}')
            alpha *= 0.5
    if bestx is None:
        logger.info(f"Stop Newton because no point with smaller function value could be found.")
        break
    fnew, dnew = fun(bestx)
    dnormnew = np.linalg.norm(dnew)
    if dnormnew >= np.linalg.norm(d):
        logger.info(f"Stop Newton because |{dnormnew}| >= |{np.linalg.norm(d)}|.")
        break
    x = bestx
    d = dnew
    f = fnew
    logger.info(f"J(x)={f:.15f}, |dJ(x)|={np.linalg.norm(d):.3e}")

curves_to_vtk(curves_rep, outdir + "curves_opt")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(outdir + "surf_opt", extra_data=pointData)
kappas = [np.max(c.kappa()) for c in base_curves]
arclengths = [np.min(c.incremental_arclength()) for c in base_curves]
dist = Jdist.shortest_distance()
logger.info(f"Curvatures {kappas}")
logger.info(f"Arclengths {arclengths}")
logger.info(f"Shortest distance {dist}")
logger.info(f"Lengths sum({[J.J() for J in Jls]})={sum([J.J() for J in Jls])}")
logger.info("Currents %s" % [c.current.get_value() for c in coils_fil])
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
