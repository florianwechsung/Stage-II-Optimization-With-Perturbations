#!/usr/bin/env python
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.curve import curves_to_vtk
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.curveobjectives import CurveLength, CoshCurveCurvature
from simsopt.geo.curveobjectives import MinimumDistance
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.surfaceobjectives import boozer_surface_residual, ToroidalFlux, Area
from simsopt.geo.qfmsurface import QfmSurface
from simsopt.geo.surfaceobjectives import QfmResidual, ToroidalFlux, Area, Volume
from objective import create_curves, get_outdir, add_correction_to_coils
from scipy.optimize import minimize
import argparse
import numpy as np

import os
os.makedirs("qfmsurfacesdet", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--outdiridx", type=int, default=0)
parser.add_argument("--well", dest="well", default=False, action="store_true")
args = parser.parse_args()

if not args.well:
    filename = 'input.LandremanPaul2021_QA'
else:
    filename = "input.20210728-01-010_QA_nfp2_A6_magwell_weight_1.00e+01_rel_step_3.00e-06_centered"

outdir = get_outdir(args.well, args.outdiridx)


fil = 0
nfp = 2

nphi = 25
ntheta = 25
phis = np.linspace(0, 1./(2*nfp), nphi, endpoint=False)
phis += phis[0]/2
thetas = np.linspace(0, 1., ntheta, endpoint=False)
s = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phis, quadpoints_theta=thetas)
x = np.loadtxt(outdir + "xmin.txt")

base_curves, base_currents, coils_fil, coils_fil_pert = create_curves(
    fil=fil, ig=0, nsamples=0, stoch_seed=1, sigma=0, order=16)

coils_qfm = coils_fil

bs = BiotSavart(coils_qfm)
bs.x = x

mpol = s.mpol
ntor = s.ntor

sq = SurfaceRZFourier(mpol=8, ntor=8, nfp=2, stellsym=True, quadpoints_phi=phis, quadpoints_theta=thetas)
for m in range(0, 6):
    for n in range(-5, 6):
        sq.set_rc(m, n, s.get_rc(m, n))
        sq.set_zs(m, n, s.get_zs(m, n))

phisfine = np.linspace(0, 1./(2*nfp), 2*nphi, endpoint=False)
phisfine += phisfine[0]/2
thetasfine = np.linspace(0, 1., 2*ntheta, endpoint=False)
sqfine = SurfaceRZFourier(mpol=16, ntor=16, nfp=2, stellsym=True, quadpoints_phi=phisfine, quadpoints_theta=thetasfine)
def transfer(sq, sqfine):
    for m in range(0, 9):
        for n in range(-8, 9):
            sqfine.set_rc(m, n, sq.get_rc(m, n))
            sqfine.set_zs(m, n, sq.get_zs(m, n))


print(len(sq.get_dofs()), "vs", nphi*ntheta)
from find_magnetic_axis import find_magnetic_axis
axis = find_magnetic_axis(bs, 200, 1.0, output='cartesian')
from simsopt.geo.curverzfourier import CurveRZFourier 
ma = CurveRZFourier(200, 10, 1, False)
ma.least_squares_fit(axis)
curves_to_vtk([ma], "/tmp/axis")

outname = outdir + f"qfm"
bs_tf = BiotSavart(coils_qfm)
# bs_tf.x = x
tf = ToroidalFlux(sq, bs_tf)
tf_init = tf.J()
bs_tffine = BiotSavart(coils_qfm)
# bs_tffine.x = x
tffine = ToroidalFlux(sqfine, bs_tffine)
# tf_init = tf.J()
# print(tf.J())
# sq.extend_via_normal(-0.04)
# print(tf.J())
# sq.to_vtk("/tmp/shrunk")

faks = [1.0, 0.25]
for i, f in enumerate(faks):
    sq.to_vtk(f"/tmp/pre_{f}")
    tf_target = tf_init * f
    qfm = QfmResidual(sq, bs)
    qfm_surface = QfmSurface(bs, sq, tf, tf_target)

    constraint_weight = 1
    print("intial qfm value", qfm.J())

    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-14, maxiter=1600,
                                                             constraint_weight=constraint_weight)
    print(f"||ar constraint||={0.5*(tf.J()-tf_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
    transfer(sq, sqfine)
    qfm = QfmResidual(sqfine, bs)
    qfm_surface = QfmSurface(bs, sqfine, tffine, tf_target)

    # constraint_weight = 1
    print("intial qfm value", qfm.J())
    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-18, maxiter=1600,
                                                             constraint_weight=constraint_weight)
    print(f"||ar constraint||={0.5*(tf.J()-tf_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-20, maxiter=1600,
                                                             constraint_weight=constraint_weight)
    print(f"||ar constraint||={0.5*(tf.J()-tf_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

    print("Volume", sqfine.volume())
    print("Area", sqfine.area())
    np.save(outname + f"_flux_{f}", sqfine.get_dofs())
    np.save("qfmsurfacesdet/" + outname.replace("/", "_") + f"_flux_{f}", sqfine.get_dofs())


    sq.to_vtk(f"/tmp/opt_{f}")
    if i < len(faks)-1:
        print("Before scale", tf.J())
        sq.scale_around_curve(ma, (faks[i+1]/faks[i])**0.5)
        print("After scale", tf.J())
