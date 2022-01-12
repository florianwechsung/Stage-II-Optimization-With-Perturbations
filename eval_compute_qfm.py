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
from objective import create_curves
from scipy.optimize import minimize
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--sigma", type=float, default=1e-3)
parser.add_argument("--sym", dest="sym", default=False, action="store_true")
parser.add_argument("--sampleidx", type=int, default=-1)
parser.add_argument("--outdiridx", type=int, default=0)
parser.add_argument("--well", dest="well", default=False, action="store_true")
parser.add_argument("--flux", type=float, default=1.0)
parser.add_argument("--zeromean", dest="zeromean", default=False, action="store_true")
args = parser.parse_args()

if args.sampleidx == -1:
    sampleidx = None
else:
    sampleidx = args.sampleidx
if not args.well:
    filename = 'input.LandremanPaul2021_QA'
    outdirs = [
        "output/well_False_lengthbound_18.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_expquad/",
        "output/well_False_lengthbound_20.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_expquad/",
        "output/well_False_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_4_order_16_expquad/",
        "output/well_False_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_2_order_16_expquad/",
    ]
else:
    filename = "input.20210728-01-010_QA_nfp2_A6_magwell_weight_1.00e+01_rel_step_3.00e-06_centered"
    outdirs = [
        "output/well_True_lengthbound_18.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_expquad/",
        "output/well_True_lengthbound_20.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_4_order_16_expquad/",
        "output/well_True_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_expquad/",
        "output/well_True_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_5_order_16_expquad/",
        "output/well_True_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_1_order_16_expquad_samples_4096_sigma_0.0005_usedetig_dashfix/",
        "output/well_True_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_expquad_samples_4096_sigma_0.001_usedetig_dashfix/",
        "output/well_True_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_expquad_samples_4096_sigma_0.002_dashfix/",
        "output/well_True_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_expquad_samples_4096_sigma_0.001_zeromean_True_usedetig_dashfix/",
    ]


initial_guess = None
# initial_guess = "output/well_True_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_5_order_16_expquad/qfm_None_flux_1.0.npy"
initial_guess = f"output/well_True_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_expquad/qfm_None_flux_{args.flux}.npy"
label_appendix = f"_{args.flux}"

fil = 0
nfp = 2

nphi = 100
ntheta = 100
sigma = args.sigma

outdir = outdirs[args.outdiridx]
x = np.loadtxt(outdir + "xmin.txt")

nsamples = 0 if sampleidx is None else sampleidx + 1
base_curves, base_currents, coils_fil, coils_fil_pert = create_curves(
    fil=fil, ig=0, nsamples=nsamples, stoch_seed=1, sigma=sigma, order=16, sym=args.sym,
    zero_mean=args.zeromean)
if sampleidx is None:
    coils_qfm = coils_fil
else:
    coils_qfm = coils_fil_pert[sampleidx]

# sq = SurfaceRZFourier(mpol=mpol+13, ntor=ntor+13, nfp=nfp, stellsym=True, quadpoints_phi=phis, quadpoints_theta=thetas)
if not args.sym:
    phis = np.linspace(0, 1., nphi, endpoint=False)
    thetas = np.linspace(0, 1., ntheta, endpoint=False)
    sq = SurfaceRZFourier(mpol=32, ntor=32, nfp=1, stellsym=False, quadpoints_phi=phis, quadpoints_theta=thetas)
    if initial_guess is None:
        s = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phis, quadpoints_theta=thetas)
        for m in range(0, 6):
            for n in range(-5, 6):
                sq.set_rc(m, 2*n, s.get_rc(m, n))
                # sq.set_rc(m, n, s.get_rc(m, n))
                # sq.set_rs(m, n, s.get_rs(m, n))
                # sq.set_zc(m, n, s.get_zc(m, n))
                sq.set_zs(m, 2*n, s.get_zs(m, n))
                # sq.set_zs(m, n, s.get_zs(m, n))
    else:
        s = SurfaceRZFourier(mpol=16, ntor=16, nfp=2, stellsym=True)
        s.x = np.load(initial_guess)
        for m in range(0, 17):
            for n in range(-16, 17):
                sq.set_rc(m, 2*n, s.get_rc(m, n))
                sq.set_zs(m, 2*n, s.get_zs(m, n))
else:
    phis = np.linspace(0, 1./4, nphi, endpoint=False)
    phis += phis[0]/2
    thetas = np.linspace(0, 1., ntheta, endpoint=False)
    sq = SurfaceRZFourier(mpol=32, ntor=32, nfp=2, stellsym=True, quadpoints_phi=phis, quadpoints_theta=thetas)
    if initial_guess is None:
        s = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phis, quadpoints_theta=thetas)
        for m in range(0, 6):
            for n in range(-5, 6):
                sq.set_rc(m, n, s.get_rc(m, n))
                # sq.set_rc(m, n, s.get_rc(m, n))
                # sq.set_rs(m, n, s.get_rs(m, n))
                # sq.set_zc(m, n, s.get_zc(m, n))
                sq.set_zs(m, n, s.get_zs(m, n))
                # sq.set_zs(m, n, s.get_zs(m, n))
    else:
        s = SurfaceRZFourier(mpol=16, ntor=16, nfp=2, stellsym=True)
        s.x = np.load(initial_guess)
        for m in range(0, 17):
            for n in range(-16, 17):
                sq.set_rc(m, n, s.get_rc(m, n))
                sq.set_zs(m, n, s.get_zs(m, n))




print(len(sq.get_dofs()), "vs", nphi*ntheta)
bs = BiotSavart(coils_qfm)
bs.x = x

if sampleidx is None:
    outname = outdir + f"qfm_{sampleidx}"
else:
    outname = outdir + f"qfm_seed_{sampleidx}"
    if args.zeromean:
        outname += "_zeromean"
    if args.sym:
        outname += "_sym"
    outname += f"_sigma_{sigma}"

outname += "_32_32" + label_appendix
# for i in range(16):
#     # np.savetxt("outputformatt/" + outname.replace("/", "_") + f"_curve_{i}.txt", coils_qfm[i].curve.full_x)
#     np.savetxt("outputformatt/" + outname.replace("/", "_") + f"_curve_{i}_xyz.txt", coils_qfm[i].curve.gamma())
#     np.savetxt("outputformatt/" + outname.replace("/", "_") + f"_current_{i}.txt", coils_qfm[i].current.full_x)

ar = Area(sq)
ar_target = ar.J()

# bs_tf = BiotSavart(coils_qfm)
# bs_tf.x = x
# tf = ToroidalFlux(sq, bs_tf)
# tf_target = tf.J()

qfm = QfmResidual(sq, bs)
# qfm_surface = QfmSurface(bs, sq, tf, tf_target)
qfm_surface = QfmSurface(bs, sq, ar, ar_target)

constraint_weight = 1e-3
print("intial qfm value", qfm.J())
from simsopt.objectives.fluxobjective import SquaredFlux
print("intial qfm value normalised", SquaredFlux(s, bs).J())

# import time
# t1 = time.time()
# res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-14, maxiter=10,
#                                                          constraint_weight=constraint_weight)
# t2 = time.time()
# print(t2-t1)
res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-14, maxiter=1600,
                                                         constraint_weight=constraint_weight)
print(f"||ar constraint||={0.5*(ar.J()-ar_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-16, maxiter=1600,
                                                         constraint_weight=constraint_weight)
print(f"||ar constraint||={0.5*(ar.J()-ar_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-18, maxiter=1600,
                                                         constraint_weight=constraint_weight)
print(f"||ar constraint||={0.5*(ar.J()-ar_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
# res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=1e-12, maxiter=10)
# print(f"||tf constraint||={0.5*(tf.J()-tf_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

# np.save(outname, sq.get_dofs())
np.save("qfmsurfaces/" + outname.replace("/", "_"), sq.get_dofs())
B = bs.set_points(sq.gamma().reshape((-1, 3))).B().reshape(sq.gamma().shape)
np.save("qfmsurfaces/" + outname.replace("/", "_") + "_B", B)
pointData = {"B_N/|B|": np.sum(bs.B().reshape(sq.gamma().shape) * sq.unitnormal(), axis=2)[:, :, None]/bs.AbsB().reshape((nphi, ntheta, 1))}
print(outdir)
bs.set_points([[0., 0., 0.]]).dB_by_dX() # clear memory usage of biot savart object
