#!/usr/bin/env python
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.magneticfieldclasses import InterpolatedField, UniformInterpolationRule
from simsopt.field.biotsavart import BiotSavart
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.surfaceobjectives import boozer_surface_residual, ToroidalFlux, Area

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
parser.add_argument("--well", dest="well", default=False, action="store_true")
args = parser.parse_args()
print(args, flush=True)

if args.sampleidx == -1:
    sampleidx = None
else:
    sampleidx = args.sampleidx
filename = 'input.LandremanPaul2021_QA'
outdirs = [
        "output/well_False_lengthbound_18.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_expquad/",
        "output/well_False_lengthbound_20.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_expquad/",
        "output/well_False_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_4_order_16_expquad/",
        "output/well_False_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_2_order_16_expquad/",
]

filename = "input.20210728-01-010_QA_nfp2_A6_magwell_weight_1.00e+01_rel_step_3.00e-06_centered"
outdirs = [
        "output/well_True_lengthbound_18.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_expquad/",
        "output/well_True_lengthbound_20.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_4_order_16_expquad/",
        "output/well_True_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_expquad/",
        "output/well_True_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_5_order_16_expquad/",
        "output/well_True_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_1_order_16_expquad_samples_4096_sigma_0.0005_usedetig_dashfix/",
        "output/well_True_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_expquad_samples_4096_sigma_0.001_usedetig_dashfix/",
]

LENGTH_SCALE = 10.1515
B_SCALE = 5.78857

fil = 0

nphi = 64
ntheta = 64
if sampleidx is None:
    mpol = 16
    ntor = 16
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
    fil=fil, ig=0, nsamples=nsamples, stoch_seed=1, sigma=sigma, order=16, comm=MPI.COMM_SELF)
# for i in list(range(nsamples)):
# for i in [None] + list(range(nsamples)):
i = sampleidx
if i is None:
    coils_boozer = coils_fil
else:
    coils_boozer = coils_fil_pert[i]

outdir = outdirs[args.outdiridx]
x = np.loadtxt(outdir + "xmin.txt")

bs = BiotSavart(coils_boozer)
bs.x = x


for i in range(4):
    coils_boozer[i].curve.x = coils_boozer[i].curve.x * LENGTH_SCALE
boozeroutdir = outdir.replace("/", "_").replace(".", "p")[:-1] + f"_seed_{sampleidx}"

souter = SurfaceXYZTensorFourier(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
souter.x = np.loadtxt("outputboozer/" + boozeroutdir + f"_1.00.txt") * LENGTH_SCALE
sinner = SurfaceXYZTensorFourier(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
sinner.x = np.loadtxt("outputboozer/" + boozeroutdir + f"_0.{args.spawnidx}0.txt") * LENGTH_SCALE

B_on_surface = bs.set_points(souter.gamma().reshape((-1, 3))).AbsB()
norm = np.linalg.norm(souter.normal().reshape((-1, 3)), axis=1)
meanb = np.mean(B_on_surface * norm)/np.mean(norm)

for i in range(4):
    c = coils_boozer[i].current._ScaledCurrent__basecurrent
    c.unfix_all()
    c.x = c.x * 5.78857 / meanb

sc_particle = SurfaceClassifier(souter, h=0.1, p=2)
n = 100 if sampleidx is None else 75
rs = np.linalg.norm(souter.gamma()[:, :, 0:2], axis=2)
zs = souter.gamma()[:, :, 2]

nparticles = 1000

degree = 5
print("n =", n, ", degree =", degree)
rrange = (np.min(rs), np.max(rs), n)
phirange = (0, 2*np.pi/nfp, n//nfp)
zrange = (0, np.max(zs), n//2) if stellsym else (np.min(zs), np.max(zs), n)
bsh = InterpolatedField(
    bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=stellsym
)
TMAX = 2e-1

seed = args.seed
def trace_particles(bfield, label, mode='gc_vac'):
    t1 = time.time()
    gc_tys, gc_phi_hits = trace_particles_starting_on_surface(
        sinner, bfield, nparticles, tmax=TMAX, seed=seed, mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE,
        Ekin=FUSION_ALPHA_PARTICLE_ENERGY, umin=-1, umax=+1, comm=comm,
        phis=[], tol=1e-11,
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
np.savetxt(f"{outdir}/particles_sampleidx_{args.sampleidx}_spawnidx_{args.spawnidx}_seed_{seed}.txt", paths_gc_h)
def get_lost_or_not(paths):
    return np.asarray([p[-1, 0] < TMAX-1e-15 for p in paths]).astype(int)
print(np.mean(get_lost_or_not(paths_gc_h)))
