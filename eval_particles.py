#!/usr/bin/env python
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.magneticfieldclasses import InterpolatedField, UniformInterpolationRule
from simsopt.field.biotsavart import BiotSavart
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.surfaceobjectives import boozer_surface_residual, ToroidalFlux, Area
from simsopt.geo.curvecorrected import CurveCorrected
from simsopt.field.coil import Current, Coil, ScaledCurrent
from objective import create_curves, add_correction_to_coils, get_outdir

import os
os.makedirs("losses", exist_ok=True)

from simsopt.field.tracing import trace_particles_starting_on_surface, SurfaceClassifier, \
    particles_to_vtk, LevelsetStoppingCriterion, plot_poincare_data
from simsopt.util.constants import FUSION_ALPHA_PARTICLE_ENERGY, ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE

import logging
logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')
logger.setLevel(1)

from objective import create_curves
import argparse
import numpy as np
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD

parser = argparse.ArgumentParser()
parser.add_argument("--sigma", type=float, default=1e-3)
parser.add_argument("--sampleidx", type=int, default=-1)
parser.add_argument("--spawnidx", type=int, default=1)
parser.add_argument("--outdiridx", type=int, default=0)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--resolution", type=int, default=75)
parser.add_argument("--well", dest="well", default=False, action="store_true")
parser.add_argument("--sym", dest="sym", default=False, action="store_true")
parser.add_argument("--correctionlevel", type=int, default=0)
parser.add_argument("--nparticles", type=int, default=2000)
args = parser.parse_args()
print(args, flush=True)

if args.sampleidx == -1:
    sampleidx = None
else:
    sampleidx = args.sampleidx
filename = 'input.LandremanPaul2021_QA'

outdir = get_outdir(args.well, args.outdiridx)


LENGTH_SCALE = 10.1515
B_SCALE = 5.78857

fil = 0

nphi = 100
ntheta = 100
if sampleidx is None or args.sym:
    mpol = 16
    ntor = 16
    mpol = 32
    ntor = 32
    stellsym = True
    nfp = 2
else:
    mpol = 10
    ntor = 20
    stellsym = False
    nfp = 1
phis = np.linspace(0, 1., nphi, endpoint=False)
thetas = np.linspace(0, 1., ntheta, endpoint=False)


sigma = args.sigma
nsamples = 0 if sampleidx is None else sampleidx + 1
base_curves, base_currents, coils_fil, coils_fil_pert = create_curves(
    fil=fil, ig=0, nsamples=nsamples, stoch_seed=1, sigma=sigma, order=16, comm=MPI.COMM_SELF, sym=False,
    zero_mean=False)
# for i in list(range(nsamples)):
# for i in [None] + list(range(nsamples)):
if sampleidx is None:
    coils_boozer = coils_fil
else:
    coils_boozer = coils_fil_pert[sampleidx]

x = np.loadtxt(outdir + "xmin.txt")

bs = BiotSavart(coils_boozer)
bs.x = x


for i in range(4):
    coils_boozer[i].curve.x = coils_boozer[i].curve.x * LENGTH_SCALE
if sampleidx is not None:
    if not args.sym:
        for i in range(4):
            coils_boozer[i].curve.curve.sample *= LENGTH_SCALE
        for i in range(16):
            coils_boozer[i].curve.sample *= LENGTH_SCALE
    else:
        for i in range(4):
            coils_boozer[i].curve.sample *= LENGTH_SCALE

if (sampleidx is not None) and args.correctionlevel > 0:
    coils_boozer = add_correction_to_coils(coils_boozer, args.correctionlevel)
    bs = BiotSavart(coils_boozer)
    corrname = "corrections/" \
        + outdir.replace("/", "_")[:-1] \
        + f"_correction_sigma_{args.sigma}_sampleidx_{sampleidx}_correctionlevel_{args.correctionlevel}"
    y = np.loadtxt(corrname + ".txt")
    bs.x = y
    for i in range(16):
       cx = coils_boozer[i].curve.x
       cx[:3] *= LENGTH_SCALE
       coils_boozer[i].curve.x = cx

qfmfilename = outdir.replace("/", "_") + f"qfm"
if sampleidx is not None:
    qfmfilename += f"_sampleidx_{sampleidx}"
    qfmfilename += f"_sigma_{sigma}"
    qfmfilename += f"_correctionlevel_{args.correctionlevel}"

#souter = SurfaceXYZTensorFourier(
#    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
souter = SurfaceRZFourier(
    mpol=32, ntor=32, stellsym=False, nfp=1, quadpoints_phi=phis, quadpoints_theta=thetas)
#souter = SurfaceRZFourier(
#    mpol=32, ntor=32, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
souter.x = np.load("qfmsurfaces/" + qfmfilename + f"_flux_1.0.npy") * LENGTH_SCALE
#sinner = SurfaceXYZTensorFourier(
#    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
sinner = SurfaceRZFourier(
    mpol=32, ntor=32, stellsym=False, nfp=1, quadpoints_phi=phis, quadpoints_theta=thetas)
#sinner = SurfaceRZFourier(
#    mpol=32, ntor=32, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
sinner.x = np.load("qfmsurfaces/" + qfmfilename + f"_flux_0.{args.spawnidx}.npy") * LENGTH_SCALE

B = bs.set_points(souter.gamma().reshape((-1, 3))).B().reshape(souter.gamma().shape)
print("Bn", np.mean(np.sum(B * souter.normal(), axis=2)**2))

B_on_surface = bs.set_points(souter.gamma().reshape((-1, 3))).AbsB()
norm = np.linalg.norm(souter.normal().reshape((-1, 3)), axis=1)
meanb = np.mean(B_on_surface * norm)/np.mean(norm)

if args.correctionlevel > 0:
    for i in range(4):
        c = coils_boozer[i].current._CurrentSum__current_A._ScaledCurrent__basecurrent
        c.unfix_all()
        c.x = c.x * 5.78857 / meanb
    if args.correctionlevel == 2:
        for i in range(16):
            cur = coils_boozer[i].current._CurrentSum__current_B._ScaledCurrent__basecurrent
            cur.x = cur.x * 5.78857 / meanb
else:
    for i in range(4):
        c = coils_boozer[i].current._ScaledCurrent__basecurrent
        c.unfix_all()
        c.x = c.x * 5.78857 / meanb

print("Intial qfm value normalised", SquaredFlux(souter, bs).J())

sc_particle = SurfaceClassifier(souter, h=0.1, p=2)
#n = 100 if sampleidx is None else 75
n = args.resolution

def skip(rs, phis, zs):
    rphiz = np.asarray([rs, phis, zs]).T.copy()
    dists = sc_particle.evaluate_rphiz(rphiz)
    skip = list((dists < -0.04).flatten())
    print("sum(skip) =", sum(skip), "out of ", len(skip), flush=True)
    # skip = [p < 0.5 for p in phis]
    return skip

rs = np.linalg.norm(souter.gamma()[:, :, 0:2], axis=2)
zs = souter.gamma()[:, :, 2]

nparticles = args.nparticles

degree = 5
print("n =", n, ", degree =", degree)
rrange = (np.min(rs), np.max(rs), n)
phirange = (0, 2*np.pi/nfp, 8*n//nfp)
zrange = (0, np.max(zs), (n//2)) if stellsym else (np.min(zs), np.max(zs), n)
bsh = InterpolatedField(
    bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=stellsym, skip=skip
)
TMAX = 2e-1

seed = args.seed
tol = 1e-11
print("tol", tol)
def trace_particles(bfield, label, mode='gc_vac'):
    t1 = time.time()
    gc_tys, gc_phi_hits = trace_particles_starting_on_surface(
        sinner, bfield, nparticles, tmax=TMAX, seed=seed, mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE,
        Ekin=FUSION_ALPHA_PARTICLE_ENERGY, umin=-1, umax=+1, comm=comm,
        phis=[], tol=tol,
        stopping_criteria=[LevelsetStoppingCriterion(sc_particle.dist)], mode=mode,
        forget_exact_path=True)
    t2 = time.time()
    print(f"Time for particle tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in gc_tys])//nparticles}", flush=True)
    # if comm is None or comm.rank == 0:
        # particles_to_vtk(gc_tys, f'/tmp/particles_{label}_{mode}')
        # plot_poincare_data(gc_phi_hits, phis, f'/tmp/poincare_particle_{label}_loss.png', mark_lost=True)
        # plot_poincare_data(gc_phi_hits, phis, f'/tmp/poincare_particle_{label}.png', mark_lost=False)
    return gc_tys 



def compute_error_on_surface(s):
    bsh.set_points(s.gamma().reshape((-1, 3)))
    dBh = bsh.GradAbsB()
    Bh = bsh.B()
    bs.set_points(s.gamma().reshape((-1, 3)))
    dB = bs.GradAbsB()
    B = bs.B()
    logger.info("Mean(|B|) on surface   %s" % np.mean(bs.AbsB()))
    logger.info("B    errors on surface %s" % np.sort(np.abs(B-Bh).flatten()))
    logger.info("∇|B| errors on surface %s" % np.sort(np.abs(dB-dBh).flatten()))

print("About to compute error")

compute_error_on_surface(sinner)
compute_error_on_surface(souter)
print("", flush=True)

paths_gc_h = trace_particles(bsh, 'bsh', 'gc_vac')
outname = outdir.replace("/", "_")[:-1] + f"_losses_sigma_{sigma}_sampleidx_{sampleidx}_correctionlevel_{args.correctionlevel}_spawnidx_{args.spawnidx}_n_{args.resolution}_seed_{seed}"
np.save("losses/" + outname + ".txt", paths_gc_h)
def get_lost_or_not(paths):
    return np.asarray([p[-1, 0] < TMAX-1e-15 for p in paths]).astype(int)
print(f"{np.mean(get_lost_or_not(paths_gc_h))*100}%")
