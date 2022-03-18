#!/usr/bin/env python
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.biotsavart import BiotSavart
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.surfaceobjectives import boozer_surface_residual, ToroidalFlux, Area
from simsopt.geo.curve import curves_to_vtk
from objective import create_curves, get_outdir, add_correction_to_coils
import numpy as np


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--well", dest="well", default=False, action="store_true")
parser.add_argument("--sampleidx", type=int, default=-1)
parser.add_argument("--correctionlevel", type=int, default=0)
args = parser.parse_args()

if not args.well:
    filename = 'input.LandremanPaul2021_QA'
else:
    filename = "input.20210728-01-010_QA_nfp2_A6_magwell_weight_1.00e+01_rel_step_3.00e-06_centered"

fil = 0

nfp = 2
nphi = 64
ntheta = 64
phis = np.linspace(0, 1./(2*nfp), nphi, endpoint=False)
thetas = np.linspace(0, 1., ntheta, endpoint=False)
starget = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phis, quadpoints_theta=thetas)
area_targets = np.flipud(np.linspace(2, starget.area(), 10, endpoint=True))

phisviz = np.linspace(0, 1./(2*nfp), 256, endpoint=True)
thetasviz = np.linspace(0, 1., 128, endpoint=True)
sviz = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phisviz, quadpoints_theta=thetasviz)

phisvizfull = np.linspace(0, 1., 1024, endpoint=True)
thetasvizfull = np.linspace(0, 1., 128, endpoint=True)
svizfull = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phisvizfull, quadpoints_theta=thetasvizfull)

nfp = 2
ntheta = 64
nphi = 64
phis = np.linspace(0, 1./(2*nfp), nphi, endpoint=False)
phis += phis[1]/2
thetas = np.linspace(0, 1., ntheta, endpoint=False)
s = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phis, quadpoints_theta=thetas)


def coilpy_plot(curves, filename, height=0.1, width=0.1):
    def wrap(data):
        return np.concatenate([data, [data[0]]])
    xx = [wrap(c.gamma()[:, 0]) for c in curves]
    yy = [wrap(c.gamma()[:, 1]) for c in curves]
    zz = [wrap(c.gamma()[:, 2]) for c in curves]
    II = [1. for _ in curves]
    names = [i for i in range(len(curves))]
    from coilpy import Coil
    coils = Coil(xx, yy, zz, II, names, names)
    coils.toVTK(filename, line=False, height=height, width=width)


for idx in range(8 if args.correctionlevel == 0 else 4):
    outdir = get_outdir(args.well, idx)
    base_curves, base_currents, coils_fil, coils_fil_pert = create_curves(
        fil=fil, ig=0, nsamples=args.sampleidx+1, stoch_seed=1, sigma=2e-3, order=16)
    coils = coils_fil if args.sampleidx == -1 else coils_fil_pert[args.sampleidx]
    bs = BiotSavart(coils)
    bs.x = np.loadtxt(outdir + "xmin.txt")
    if args.correctionlevel > 0:
        coils = add_correction_to_coils(coils, args.correctionlevel)
        bs = BiotSavart(coils)
        corrname = "corrections/" \
            + outdir.replace("/", "_")[:-1] \
            + f"_correction_sigma_{2e-3:.4g}_sampleidx_{args.sampleidx}_correctionlevel_{args.correctionlevel}"
        y = np.loadtxt(corrname + ".txt")
        bs.x = y

    B_on_surface = bs.set_points(s.gamma().reshape((-1, 3))).AbsB()
    norm = np.linalg.norm(s.normal().reshape((-1, 3)), axis=1)
    meanb = np.mean(B_on_surface * norm)/np.mean(norm)

    appendix = ""
    if args.sampleidx >= 0:
        appendix += f"_sample_{args.sampleidx}_cl_{args.correctionlevel}"
    coilpy_plot([c.curve for c in coils], outdir + f"coils_fb_full{appendix}.vtu", height=0.05, width=0.05)
    coilpy_plot([c.curve for c in coils[:4]], outdir + f"coils_fb_partial{appendix}.vtu", height=0.05, width=0.05)
    pointData = {
        "B·n/|B|": np.sum(bs.set_points(svizfull.gamma().reshape((-1, 3))).B().reshape(svizfull.gamma().shape) * svizfull.unitnormal(), axis=2)[:, :, None]/bs.AbsB().reshape(svizfull.gamma().shape[:2] + (1,)),
        "|B|": bs.AbsB().reshape(svizfull.gamma().shape[:2] + (1,))/meanb
    }
    svizfull.to_vtk(outdir + "surf_opt_vis_full" + appendix, extra_data=pointData)

    pointData = {
        "B·n/|B|": np.sum(bs.set_points(sviz.gamma().reshape((-1, 3))).B().reshape(sviz.gamma().shape) * sviz.unitnormal(), axis=2)[:, :, None]/bs.AbsB().reshape(sviz.gamma().shape[:2] + (1,)),
        "|B|": bs.AbsB().reshape(sviz.gamma().shape[:2] + (1,))/meanb
    }
    sviz.to_vtk(outdir + "surf_opt_vis_partial" + appendix, extra_data=pointData)
    bnbyb = np.sum(bs.set_points(sviz.gamma().reshape((-1, 3))).B().reshape(sviz.gamma().shape) * sviz.unitnormal(), axis=2)[:, :, None]/bs.AbsB().reshape(sviz.gamma().shape[:2] + (1,))
    print(np.linalg.norm(bnbyb))
    print(f"Created paraview files in directory {outdir}")

