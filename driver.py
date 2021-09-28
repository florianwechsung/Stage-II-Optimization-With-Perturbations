#!/usr/bin/env python
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux, CoilOptObjective
from simsopt.geo.curve import curves_to_vtk
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.curveobjectives import CurveLength, CoshCurveCurvature
from simsopt.geo.curveobjectives import MinimumDistance, LpCurveCurvature
from objective import create_curves, CoshCurveLength, QuadraticCurveLength
from scipy.optimize import minimize
import argparse
import os
import numpy as np
from mpi4py import MPI
import logging
comm = MPI.COMM_WORLD
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
parser.add_argument("--lengthbound", type=float, default=18.)
parser.add_argument("--mindist", type=float, default=0.10)
parser.add_argument("--maxkappa", type=float, default=7.0)
parser.add_argument("--well", dest="well", default=False, action="store_true")
parser.add_argument("--zeromean", dest="zeromean", default=False, action="store_true")
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
if args.well:
    filename = "input.20210728-01-010_QA_nfp2_A6_magwell_weight_1.00e+01_rel_step_3.00e-06_centered"
else:
    filename = "input.LandremanPaul2021_QA"
s = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phis, quadpoints_theta=thetas)


MAXITER = 12500
ALPHA = args.alpha

MIN_DIST = args.mindist
DIST_ALPHA = 10.
DIST_WEIGHT = 1

KAPPA_MAX = args.maxkappa
KAPPA_ALPHA = 0.1
KAPPA_WEIGHT = .1

LENGTH_CON_ALPHA = 0.1
LENGTH_CON_WEIGHT = 1

outdir = f"output-quad/well_{args.well}_lengthbound_{args.lengthbound}_kap_{args.maxkappa}_dist_{args.mindist}_fil_{args.fil}_ig_{args.ig}_samples_{args.nsamples}_sigma_{args.sigma}_zeromean_{args.zeromean}/"
os.makedirs(outdir, exist_ok=True)
set_file_logger(outdir + "log.txt")

base_curves, base_currents, coils_fil, coils_fil_pert = create_curves(fil=args.fil, ig=args.ig, nsamples=args.nsamples, stoch_seed=0, sigma=args.sigma, zero_mean=args.zeromean)

bs = BiotSavart(coils_fil)
bs.set_points(s.gamma().reshape((-1, 3)))

pointData = {"B_N/|B|": np.sum(bs.B().reshape(s.gamma().shape) * s.unitnormal(), axis=2)[:, :, None]/bs.AbsB().reshape((nphi, ntheta, 1))}
s.to_vtk(outdir + "surf_init", extra_data=pointData)
curves_rep = [c.curve for c in coils_fil]
NFIL = (2*args.fil + 1)**2
curves_rep_no_fil = [curves_rep[NFIL//2 + i*NFIL] for i in range(len(curves_rep)//NFIL)]

curves_to_vtk(curves_rep, outdir + "curves_init")

Jls = [CurveLength(c) for c in base_curves]


# Jlconstraint = CoshCurveLength(Jls, args.lengthbound, LENGTH_CON_ALPHA)
# Jdist = MinimumDistance(curves_rep_no_fil, MIN_DIST, penalty_type="cosh", alpha=DIST_ALPHA)
# Jkappas = [CoshCurveCurvature(c, kappa_max=KAPPA_MAX, alpha=KAPPA_ALPHA) for c in base_curves]

Jlconstraint = QuadraticCurveLength(Jls, args.lengthbound, 0.1*LENGTH_CON_ALPHA)
Jdist = MinimumDistance(curves_rep_no_fil, MIN_DIST, penalty_type="quadratic", alpha=1.)
KAPPA_WEIGHT = 1e-7
DIST_WEIGHT = 0.1
LENGTH_CON_WEIGHT = 0.01
Jkappas = [LpCurveCurvature(c, 2, desired_length=2*np.pi/KAPPA_MAX) for c in base_curves]

Jf = SquaredFlux(s, bs)

Jfs = [SquaredFlux(s, BiotSavart(cs)) for cs in coils_fil_pert]

if args.nsamples > 0:
    from objective import MPIObjective
    Jmpi = MPIObjective(Jfs, comm)
    Jmpi.J()
    JF = CoilOptObjective([Jmpi], Jls, ALPHA, Jdist, DIST_WEIGHT)
else:
    JF = CoilOptObjective(Jf, Jls, ALPHA, Jdist, DIST_WEIGHT)



history = []
ctr = [0]


def cb(*args):
    ctr[0] += 1


lastgrad = [None]
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
        dJ += LENGTH_CON_WEIGHT * Jlconstraint.dJ(partials=True)
    grad = dJ(JF)
    cl_string = ", ".join([f"{J.J():.3f}" for J in Jls])
    totalcl = sum([J.J() for J in Jls])
    mean_AbsB = np.mean(bs.AbsB())
    jf = Jf.J()
    kappas = [np.max(c.kappa()) for c in base_curves]
    kappa_string = ", ".join([f"{k:.3f}" for k in kappas])
    dist = Jdist.shortest_distance()
    s = f"{ctr[0]}, J={J:.3e}, Jflux={jf:.3e}, sqrt(Jflux)/Mean(|B|)={np.sqrt(jf)/mean_AbsB:.3e}, CoilLengths=[{cl_string}]={totalcl:.3e}, kappas=[{kappa_string}], dist={dist:.3e}, ||∇J||={np.linalg.norm(grad):.3e}"
    if args.nsamples > 0:
        s += f", {Jmpi.J():.3e}"
    if not silent:
        logger.info(s)
    history.append((J, jf, np.linalg.norm(grad)))
    if J > 1.:
        J = 1.
        grad = -lastgrad[0]
    else:
        lastgrad[0] = grad
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
f(dofs)
# import time
# import cProfile
# pr = cProfile.Profile()
# pr.enable()
# t1 = time.time()

np.random.seed(1)
h = np.random.uniform(size=dofs.shape)
J0, dJ0 = f(dofs)
dJh = sum(dJ0 * h)
for eps in [1e-3, 1e-4, 1e-5]:
    Jpp, _ = f(dofs + 2*eps*h)
    Jp, _ = f(dofs + eps*h)
    Jm, _ = f(dofs - eps*h)
    Jmm, _ = f(dofs - 2*eps*h)
    # logger.info(f"err {, (Jp-Jm)/(2*eps) - dJh}")
    logger.info(f"err {((1/12)*Jmm - (2/3)*Jm + (2/3)*Jp - (1/12)*Jpp)/(eps) - dJh}")
# t2 = time.time()
# print("Time", t2-t1)
# pr.disable()
# pr.dump_stats('profile.stat')
# import sys; sys.exit()
logger.info("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")
curiter = 0
outeriter = 0
PENINCREASES = 5
MAXLOCALITER = MAXITER//PENINCREASES
while MAXITER-curiter > 0 and outeriter < 10:
    if outeriter > 0 and outeriter < PENINCREASES:
        if max([np.max(c.kappa()) for c in base_curves]) > (1+1e-3)*KAPPA_MAX:
            logger.info("Increase weight for kappa")
            KAPPA_WEIGHT *= 10.
        if sum([J.J() for J in Jls]) > (1+1e-3)*args.lengthbound:
            logger.info("Increase weight for length")
            LENGTH_CON_WEIGHT *= 10.
        if Jdist.shortest_distance() < (1-1e-3)*MIN_DIST:
            logger.info("Increase weight for distance")
            JF.beta *= 10.
    # res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': min(MAXLOCALITER, MAXITER-curiter), 'maxcor': 400}, tol=0., callback=cb)
    # res = minimize(fun, dofs, jac=True, method='BFGS', options={'maxiter': min(MAXLOCALITER, MAXITER-curiter)}, tol=1e-15, callback=cb)

    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxfun': min(MAXLOCALITER, MAXITER-curiter), 'maxcor': 400}, tol=0., callback=cb)
    logger.info("%s" % res)
    dofs = res.x
    curiter += res.nfev
    outeriter += 1
    curves_to_vtk(curves_rep, outdir + f"curves_iter_{curiter}")


def approx_H(x, eps=1e-4):
    n = x.size
    H = np.zeros((n, n))
    x0 = x
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
eps = 1e-4
for i in range(5):
    try:
        H = approx_H(x, eps=eps)
        D, E = eigh(H)
    except:
        logger.info(f"Newton iteration {i} failed, decrease eps")
        eps *= 0.1
        continue
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

fun(x)

curves_to_vtk(curves_rep, outdir + "curves_opt")
pointData = {"B_N/|B|": np.sum(bs.B().reshape(s.gamma().shape) * s.unitnormal(), axis=2)[:, :, None]/bs.AbsB().reshape((nphi, ntheta, 1))}
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
