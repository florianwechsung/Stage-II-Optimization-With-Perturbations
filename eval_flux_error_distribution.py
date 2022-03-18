#!/usr/bin/env python
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.field.biotsavart import BiotSavart
from objective import create_curves
import numpy as np


well = True
if not well:
    filename = 'input.LandremanPaul2021_QA'
else:
    filename = "input.20210728-01-010_QA_nfp2_A6_magwell_weight_1.00e+01_rel_step_3.00e-06_centered"


labels = [
    "Det", "Stoch"
]

# zero_mean = True
zero_mean = False
sigma = 1e-3
# sigma = 5e-4
fil = 0

nfp = 2
noutsamples = 4024
# nphi = 64
# ntheta = 64
# phis = np.linspace(0, 1./(2*nfp), nphi, endpoint=False)
# thetas = np.linspace(0, 1., ntheta, endpoint=False)
nphi = 64
ntheta = 128
phis = np.linspace(0, 1., nphi, endpoint=False)
thetas = np.linspace(0, 1., ntheta, endpoint=False)
s = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phis, quadpoints_theta=thetas)

outsamples = []
for idx in [3, 5]:
    fluxsthis = []
    outdir = get_outdir(well, idx)
    print("outdir", outdir)
    try:
        x = np.loadtxt(outdir + "xmin.txt")
    except:
        print("skipping", outdir)
        continue

    # base_curves, base_currents, coils_fil, coils_fil_pert = create_curves(
    #     fil=fil, ig=0, nsamples=noutsamples, stoch_seed=1, sigma=sigma, zero_mean=zero_mean, order=16)
    # outsamplesthis = []
    # for coils in coils_fil_pert:
    #     bs = BiotSavart(coils)
    #     bs.x = x
    #     bs.set_points(s.gamma().reshape((-1, 3)))
    #     val = SquaredFlux(s, bs).J()
    #     outsamplesthis.append(val)

    base_curves, base_currents, coils_fil, coils_fil_pert = create_curves(
        fil=fil, ig=0, nsamples=1, stoch_seed=1, sigma=sigma, zero_mean=zero_mean,
        order=16)

    cs = coils_fil_pert[0]
    outsamples = []
    outerrs = []
    bs = BiotSavart(cs)
    bs.x = x
    outsamplesthis = []
    for k in range(noutsamples):
        for i in range(16):
            cs[i].curve.resample()
        for i in range(4):
            cs[i].curve.curve.resample()
        for i in range(4, 16):
            cs[i].curve.curve.invalidate_cache()
        val = SquaredFlux(s, bs).J()
        outsamplesthis.append(val)
    print(f"Outsample sigma={sigma:.1e}, mean={np.mean(outsamplesthis)}, 5%={np.quantile(outsamplesthis, 0.05)}, 95%={np.quantile(outsamplesthis, 0.95)}, stddev={np.std(outsamplesthis)}, max={np.max(outsamplesthis)}")
    outsamples.append(outsamplesthis)

import sys; sys.exit()
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_style('whitegrid')
data = []
for i in range(2):
    try:
        p = sns.kdeplot(outsamples[i], label=labels[i])
        d = p.get_lines()[-1].get_data()
        data += d
    except:
        pass
# plt.xlabel('$f_B$')
# plt.ylabel('pdf')
# plt.title('Probability density at optimal configuration (L_max = 24)')
# plt.legend()
# plt.show()

print(np.array2string(np.asarray(data).T[::3, :], max_line_width=200, precision=5, separator=";", ))
import IPython; IPython.embed()
import sys; sys.exit()
