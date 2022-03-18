from objective import create_curves, MeanSquareCurvature, get_outdir
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.curveobjectives import MinimumDistance
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from scipy.spatial.distance import cdist
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--well", dest="well", default=False, action="store_true")
args = parser.parse_args()


if not args.well:
    filename = 'input.LandremanPaul2021_QA'
else:
    filename = "input.20210728-01-010_QA_nfp2_A6_magwell_weight_1.00e+01_rel_step_3.00e-06_centered"


base_curves, base_currents, coils_fil, coils_fil_pert = create_curves(
    fil=0, ig=0, nsamples=0,
    stoch_seed=0, sigma=0,
    zero_mean=False, order=16)

s = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=np.linspace(0, 1, 100, endpoint=False), quadpoints_theta=np.linspace(0, 1, 100, endpoint=False))
sg = s.gamma().reshape((-1, 3))

bs = BiotSavart(coils_fil)


os.makedirs("minimizers", exist_ok=True)

for idx in range(8):
    outdir = get_outdir(args.well, idx)
    bs.x = np.loadtxt(outdir + "xmin.txt")
    bs.set_points(sg)
    BnbyB = np.sum(bs.B().reshape(s.gamma().shape) * s.unitnormal(), axis=2)[:, :, None]/bs.AbsB().reshape(s.gamma().shape[:2] + (1,))
    bmax = np.max(np.abs(BnbyB))
    maxkappa = max([np.max(c.kappa()) for c in base_curves])
    kappa_string = ", ".join([f"{np.max(c.kappa()):.1f}" for c in base_curves])
    msc_string = ", ".join([f"{MeanSquareCurvature(c, None).msc():.1f}" for c in base_curves])
    maxmsc = max(MeanSquareCurvature(c, None).msc() for c in base_curves)
    mindist_c = MinimumDistance([c.curve for c in coils_fil], 0.1).shortest_distance()
    mindist_s = min(np.min(cdist(sg, c.gamma())) for c in base_curves)
    print(f"[{kappa_string}]  & [{msc_string}] & {mindist_c:.3f} & {mindist_s:.3f} & {bmax:.3e}\\\\")
    for i in range(16):
        np.savetxt("minimizers/" + outdir.replace("/", "_") + f"_curve_{i}.txt", coils_fil[i].curve.full_x)
        np.savetxt("minimizers/" + outdir.replace("/", "_") + f"_curve_{i}_xyz.txt", coils_fil[i].curve.gamma())
        np.savetxt("minimizers/" + outdir.replace("/", "_") + f"_current_{i}.txt", coils_fil[i].current.full_x)
