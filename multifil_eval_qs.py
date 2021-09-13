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
nphi = 64
ntheta = 64
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

coils_boozer = coils_fil
bs = BiotSavart(coils_boozer)
bs.x = x
bs_tf = BiotSavart(coils_boozer)
bs_tf.x = x
current_sum = sum(abs(c.current.get_value()) for c in coils_boozer)
G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

mpol = 8  # try increasing this to 8 or 10 for smoother surfaces
ntor = 8  # try increasing this to 8 or 10 for smoother surfaces
stellsym = True
iota = 0.4
phis = np.linspace(0, 1/(2*nfp), ntor+1, endpoint=False)
thetas = np.linspace(0, 1., 2*mpol+1, endpoint=False)
NFP = nfp

# mpol = 8
# ntor = nfp*8
# NFP = 1
# stellsym = False
# iota = 0.4
# phis = np.linspace(0, 1, 2*ntor+1, endpoint=False)
# thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)

s = SurfaceXYZTensorFourier(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=NFP, quadpoints_phi=phis, quadpoints_theta=thetas)
s.least_squares_fit(SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phis, quadpoints_theta=thetas).gamma())
s.to_vtk("/tmp/surf")

tf = ToroidalFlux(s, bs_tf)
ar = Area(s)
ar_target = ar.J()


def magnetic_field_on_surface(s, bs):
    x = s.gamma()
    B = bs.set_points(x.reshape((-1, 3))).B().reshape(x.shape)
    mod_B = np.linalg.norm(B, axis=2)
    return mod_B


def compute_non_quasisymmetry_L2(s, bs):
    x = s.gamma()
    B = bs.set_points(x.reshape((-1, 3))).B().reshape(x.shape)
    mod_B = np.linalg.norm(B, axis=2)
    n = np.linalg.norm(s.normal(), axis=2)
    mean_phi_mod_B = np.mean(mod_B*n, axis=0)/np.mean(n, axis=0)
    mod_B_QS = mean_phi_mod_B[None, :]
    mod_B_non_QS = mod_B - mod_B_QS
    non_qs = np.mean(mod_B_non_QS**2 * n)**0.5
    qs = np.mean(mod_B_QS**2 * n)**0.5
    return non_qs, qs


boozer_surface = BoozerSurface(bs, s, ar, ar_target)
res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(
    tol=1e-10, maxiter=300, constraint_weight=100., iota=iota, G=G0)
print(f"After LBFGS:   iota={res['iota']:.3f}, tf={tf.J():.3f}, area={s.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")

# for ar_target in [ar.J()]:
for ar_target in np.flipud(np.linspace(2, ar.J(), 10, endpoint=True)):
    boozer_surface = BoozerSurface(bs, s, ar, ar_target)
    res = boozer_surface.minimize_boozer_penalty_constraints_ls(
        tol=1e-11, maxiter=100, constraint_weight=100., iota=res['iota'], G=res['G'], method='manual')
    print(f"After Lev-Mar: iota={res['iota']:.3f}, tf={tf.J():.3f}, area={s.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")
    # res = boozer_surface.solve_residual_equation_exactly_newton(
    #     tol=1e-10, maxiter=100, iota=res['iota'], G=res['G'])
    # print(f"After Exact : iota={res['iota']:.3f}, tf={tf.J():.3f}, area={s.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")
    non_qs, qs = compute_non_quasisymmetry_L2(s, bs)
    print(ar_target, ";", non_qs/qs)
    s.to_vtk("/tmp/surf", extra_data={"BN": magnetic_field_on_surface(s, bs)[:, :, None]})

import sys; sys.exit()
import matplotlib.pyplot as plt
fig = plt.figure()
im = plt.contourf(2*np.pi*phis, 2*np.pi*thetas, magnetic_field_on_surface(s, bs).T, cmap='viridis')
plt.xlabel(r'$\phi$')
plt.ylabel(r'$\theta$')
plt.title('Magnetic field strength on surface')
plt.colorbar()
fig.tight_layout()
plt.show()
# plt.savefig(f'surfaces/surfaces.pdf')
# plt.close()
