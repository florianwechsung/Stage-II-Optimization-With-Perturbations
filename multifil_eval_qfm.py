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
parser.add_argument("--noutsamples", type=int, default=0)
args = parser.parse_args()

if args.nsamples == 0:
    args.sigma = 0.

import os
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']

nfp = 2
nphi = 128
ntheta = 128
phis = np.linspace(0, 1./(2*nfp), nphi, endpoint=False)
thetas = np.linspace(0, 1., ntheta, endpoint=False)
s = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phis, quadpoints_theta=thetas)

outdir = f"output/alpha_{args.alpha}_fil_{args.fil}_ig_{args.ig}_samples_{args.nsamples}_sigma_{args.sigma}/"
os.makedirs(outdir, exist_ok=True)
x = np.loadtxt(outdir + "xmin.txt")

base_curves, base_currents, coils_fil, coils_fil_pert = create_curves(
    fil=args.fil, ig=args.ig, nsamples=args.nsamples, stoch_seed=0, sigma=args.sigma)
if args.nsamples > 0:
    print(np.max(np.linalg.norm(coils_fil[0].curve.gamma()-coils_fil_pert[0][0].curve.gamma(), axis=1)))
bs = BiotSavart(coils_fil)
bs.x = x
bs.set_points(s.gamma().reshape((-1, 3)))
print("Det", SquaredFlux(s, bs).J())



coils_qfm = coils_fil


mpol = s.mpol
ntor = s.ntor
from simsopt.geo.qfmsurface import QfmSurface
from simsopt.geo.surfaceobjectives import QfmResidual, ToroidalFlux, Area, Volume
sq = SurfaceRZFourier(mpol=mpol+13, ntor=ntor+13, nfp=nfp, stellsym=True, quadpoints_phi=phis, quadpoints_theta=thetas)
print(mpol+13)
print(ntor+13)
# sq.set_dofs(s.get_dofs())
sq.least_squares_fit(s.gamma())
bs = BiotSavart(coils_qfm)
bs_tf = BiotSavart(coils_qfm)

ar = Area(sq)
ar_target = ar.J()

tf = ToroidalFlux(sq, bs_tf)
tf_target = tf.J()

qfm = QfmResidual(sq, bs)
# qfm_surface = QfmSurface(bs, sq, tf, tf_target)
qfm_surface = QfmSurface(bs, sq, ar, ar_target)

constraint_weight = 1e0
print("intial qfm value", qfm.J())
res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-12, maxiter=400,
                                                         constraint_weight=constraint_weight)
print(f"||tf constraint||={0.5*(tf.J()-tf_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-12, maxiter=400,
                                                         constraint_weight=constraint_weight)
print(f"||tf constraint||={0.5*(tf.J()-tf_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=1e-12, maxiter=10)
print(f"||tf constraint||={0.5*(tf.J()-tf_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
np.save(outdir + "qfm", sq.get_dofs())
