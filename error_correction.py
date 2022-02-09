#!/usr/bin/env python
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.curve import curves_to_vtk
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.curveobjectives import CurveLength, CoshCurveCurvature
from simsopt.geo.curveobjectives import MinimumDistance
from simsopt.geo.curvecorrected import CurveCorrected
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.surfaceobjectives import boozer_surface_residual, ToroidalFlux, Area
from simsopt.geo.qfmsurface import QfmSurface
from simsopt.geo.surfaceobjectives import QfmResidual, ToroidalFlux, Area, Volume
from simsopt.objectives.fluxobjective import SquaredFlux, CoilOptObjective
from simsopt.field.coil import Current, Coil, ScaledCurrent, coils_via_symmetries
from objective import create_curves, add_correction_to_coils
from scipy.optimize import minimize
import argparse
import os
import numpy as np
import logging
logger = logging.getLogger("SIMSOPT-STAGE2")
handler = logging.StreamHandler()
formatter = logging.Formatter(fmt="%(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

parser = argparse.ArgumentParser()
parser.add_argument("--sigma", type=float, default=1e-3)
parser.add_argument("--sym", dest="sym", default=False, action="store_true")
parser.add_argument("--sampleidx", type=int, default=-1)
parser.add_argument("--outdiridx", type=int, default=0)
parser.add_argument("--well", dest="well", default=False, action="store_true")
parser.add_argument("--correctionlevel", type=int, default=1)
args = parser.parse_args()
if args.correctionlevel == 0:
    quit()

print(args)
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
    ]


fil = 0
nfp = 2

sigma = args.sigma

outdir = outdirs[args.outdiridx]
x = np.loadtxt(outdir + "xmin.txt")

nsamples = 0 if sampleidx is None else sampleidx + 1
base_curves, base_currents, coils_fil, coils_fil_pert = create_curves(
    fil=fil, ig=0, nsamples=nsamples, stoch_seed=1, sigma=sigma, order=16, sym=args.sym)
if sampleidx is None:
    coils = coils_fil
else:
    coils = coils_fil_pert[sampleidx]

bs = BiotSavart(coils)
bs.x = x
coils_corrected = add_correction_to_coils(coils, args.correctionlevel)
curves_corrected = [c.curve for c in coils_corrected]

nphi = 128
phis = np.linspace(0, 1., nphi, endpoint=False)
phis += phis[1]/2
ntheta = 32
thetas = np.linspace(0, 1., ntheta, endpoint=False)
s = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phis, quadpoints_theta=thetas)
MIN_DIST = 0.1
DIST_WEIGHT = 10

bs = BiotSavart(coils_corrected)
bs.set_points(s.gamma().reshape((-1, 3)))
Jf = SquaredFlux(s, bs)
Jdist = MinimumDistance(curves_corrected, MIN_DIST, penalty_type="quadratic", alpha=1.)
Jls = [CurveLength(c) for c in base_curves]
JF = CoilOptObjective(Jf, [], 0, Jdist, DIST_WEIGHT)

lastgrad = [None]
ctr = [0]
def fun(dofs, silent=False):
    Jf.x = dofs
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    # J = Jf.J()
    # grad = Jf.dJ()

    cl_string = ", ".join([f"{J.J():.3f}" for J in Jls])
    totalcl = sum([J.J() for J in Jls])
    mean_AbsB = np.mean(bs.AbsB())
    kappas = [np.max(c.kappa()) for c in base_curves]
    kappa_string = ", ".join([f"{k:.3f}" for k in kappas])
    dist = Jdist.shortest_distance()
    s = f"{ctr[0]}, J={J:.3e}, Jflux={jf:.3e}, sqrt(Jflux)/Mean(|B|)={np.sqrt(jf)/mean_AbsB:.3e}, CoilLengths=[{cl_string}]={totalcl:.3e}, kappas=[{kappa_string}], dist={dist:.3e}, ||∇J||={np.linalg.norm(grad):.3e}"
    if not silent:
        logger.info(s)
    # if J > 1.:
    #     J = 1.
    #     grad = -lastgrad[0]
    # else:
    #     lastgrad[0] = grad
    ctr[0] += 1
    return 1e-4 * J, 1e-4 * grad


logger.info("""
################################################################################
### Perform a Taylor test ######################################################
################################################################################
""")
f = fun
dofs = JF.x
print(dofs.shape)
# dofs = x
f(dofs)
# import time
# import cProfile
# pr = cProfile.Profile()
# pr.enable()
# t1 = time.time()

np.random.seed(1)
# h = np.random.uniform(size=dofs.shape)
h = np.random.standard_normal(size=dofs.shape)
J0, dJ0 = f(dofs)
print("J0", J0)
# J1, dJ1 = f(dofs+0.01*h)
# print("J1", J1)
# import sys; sys.exit()
dJh = sum(dJ0 * h)
for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    Jpp, _ = f(dofs + 2*eps*h, silent=True)
    Jp, _ = f(dofs + eps*h, silent=True)
    Jm, _ = f(dofs - eps*h, silent=True)
    Jmm, _ = f(dofs - 2*eps*h, silent=True)
    # logger.info(f"err {, (Jp-Jm)/(2*eps) - dJh}")
    logger.info(f"err {((1/12)*Jmm - (2/3)*Jm + (2/3)*Jp - (1/12)*Jpp)/(eps) - dJh}")
# t2 = time.time()
# print("Time", t2-t1)
# pr.disable()
# pr.dump_stats('profile.stat')
# import sys; sys.exit()
f(dofs)
print(len(dofs))

dofs = JF.x
print(len(dofs))
logger.info("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")

cb = None
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxfun': 200, 'maxcor': 200, 'maxls': 40}, tol=0., callback=cb)

curves_to_vtk([c.curve for c in coils_corrected], "/tmp/curves_corrected")
print(JF.x.shape)

# import matplotlib.pyplot as plt
# idx = 2
# plt.plot(coils[idx].curve.gamma() - base_curves[idx].gamma(), "-", label="Coil error before correction")
# plt.gca().set_prop_cycle(None)
# plt.plot(coils_corrected[idx].curve.gamma() - base_curves[idx].gamma(), ":", label="Coil error after correction")
# plt.show()
# import IPython; IPython.embed()
# import sys; sys.exit()
outname = outdir.replace("/", "_")[:-1] + f"_correction_sigma_{args.sigma}_sampleidx_{sampleidx}_correctionlevel_{args.correctionlevel}"
if args.sym:
    outname += "_sym"
np.savetxt("corrections/" + outname + ".txt", JF.x)
print(outname)

