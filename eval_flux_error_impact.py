#!/usr/bin/env python
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.curve import curves_to_vtk
from simsopt.field.coil import Current, Coil, ScaledCurrent, coils_via_symmetries
from simsopt.geo.curvecorrected import CurveCorrected
from objective import create_curves, add_correction_to_coils, get_outdir
import numpy as np
import os
os.makedirs("analysisresults", exist_ok=True)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--well", dest="well", default=False, action="store_true")
parser.add_argument("--stoch", dest="stoch", default=False, action="store_true")
parser.add_argument("--correctionlevel", type=int, default=0)
args = parser.parse_args()


well = args.well
if not well:
    filename = 'input.LandremanPaul2021_QA'
else:
    filename = "input.20210728-01-010_QA_nfp2_A6_magwell_weight_1.00e+01_rel_step_3.00e-06_centered"
correctionlevel = args.correctionlevel

seed = 1

fil = 0

nfp = 2
nphi = 128
ntheta = 32
phis = np.linspace(0, 1., nphi, endpoint=False)
thetas = np.linspace(0, 1., ntheta, endpoint=False)
s = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phis, quadpoints_theta=thetas)

fluxs = []
sigmas = np.asarray([i*1e-4 for i in range(1, 11)])

noutsamples = 128
idxs = range(8) if args.stoch else range(4)
for idx in idxs:
    fluxsthis = []
    outdir = get_outdir(well, idx)
    try:
        x = np.loadtxt(outdir + "xmin.txt")
    except:
        print("skipping", outdir)
        continue

    base_curves, base_currents, coils_fil, coils_fil_pert = create_curves(
        fil=fil, ig=0, nsamples=0, stoch_seed=0, sigma=0, zero_mean=False,
        order=16)

    bs = BiotSavart(coils_fil)
    bs.x = x
    fluxsthis.append(SquaredFlux(s, bs).J())
    for sigma in sigmas:
        base_curves, base_currents, coils_fil, coils_fil_pert = create_curves(
            fil=fil, ig=0, nsamples=noutsamples, stoch_seed=seed, sigma=sigma,
            zero_mean=False, order=16)


        outsamples = []
        outerrs = []
        bs = BiotSavart(coils_fil)
        bs.x = x
        # fix_all_dofs(coils_fil)

        for k in range(noutsamples):
            cs = coils_fil_pert[k]
            if correctionlevel > 0:
                cs = add_correction_to_coils(cs, correctionlevel, already_fixed=k>0)
                bs = BiotSavart(cs)
                corrname = "corrections/" \
                    + outdir.replace("/", "_")[:-1] \
                    + f"_correction_sigma_{sigma:.4g}_sampleidx_{k}_correctionlevel_{correctionlevel}"
                y = np.loadtxt(corrname + ".txt")
                bs.x = y
            else:
                bs = BiotSavart(cs)
            val = SquaredFlux(s, bs).J()
            bs.set_points([[0., 0., 0.]]).B()
            outsamples.append(val)
        print(f"Outsample sigma={sigma:.1e}, mean={np.mean(outsamples)}, stddev={np.std(outsamples)}, max={np.max(outsamples)}")
        fluxsthis.append(np.mean(outsamples))
    fluxs.append(fluxsthis)

dat = np.asarray([[0] + [1000*s for s in sigmas]]+fluxs).T
print(np.array2string(dat, m3.62666e-06ax_line_width=200, precision=5, separator=";", ))
colnames = [
    "Det18","Det20","Det22","Det24",
    "Stoch18","Stoch20",
    "Stoch22","Stoch24"]
np.savetxt(f"analysisresults/flux_well_{well}_correctionlevel_{correctionlevel}.txt", dat, delimiter=',', newline='\n', header='sigma,' + ','.join([colnames[i] for i in idxs]), comments='')
print("\n\n\n")
# import IPython; IPython.embed()
# import sys; sys.exit()
