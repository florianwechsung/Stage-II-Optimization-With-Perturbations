#!/usr/bin/env python
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.biotsavart import BiotSavart
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.surfaceobjectives import boozer_surface_residual, ToroidalFlux, Area
from objective import create_curves
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--sigma", type=float, default=1e-3)
parser.add_argument("--sampleidx", type=int, default=-1)
parser.add_argument("--outdiridx", type=int, default=0)
parser.add_argument("--well", dest="well", default=False, action="store_true")
args = parser.parse_args()

if args.sampleidx == -1:
    sampleidx = None
else:
    sampleidx = args.sampleidx

if not args.well:
    filename = 'input.LandremanPaul2021_QA'
else:
    filename = "input.20210728-01-010_QA_nfp2_A6_magwell_weight_1.00e+01_rel_step_3.00e-06_centered"

outdir = get_outdir(args.well, args.outdiridx)
fil = 0

nfp = 2
nphi = 64
ntheta = 64
phis = np.linspace(0, 1., nphi, endpoint=False)
thetas = np.linspace(0, 1., ntheta, endpoint=False)
starget = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phis, quadpoints_theta=thetas)
area_targets = np.flipud(np.linspace(2, starget.area(), 20, endpoint=True))
# tf_ratios = [(i/20) for i in reversed(range(1, 21))]
tf_ratios = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1]
print("tf_ratios", tf_ratios)

sigma = args.sigma
nsamples = 0 if sampleidx is None else sampleidx + 1
base_curves, base_currents, coils_fil, coils_fil_pert = create_curves(
    fil=fil, ig=0, nsamples=nsamples, stoch_seed=1, sigma=sigma, order=16)
# for i in list(range(nsamples)):
# for i in [None] + list(range(nsamples)):
i = sampleidx
if i is None:
    coils_boozer = coils_fil
else:
    coils_boozer = coils_fil_pert[i]


x = np.loadtxt(outdir + "xmin.txt")

bs = BiotSavart(coils_boozer)
bs.x = x



bs_tf = BiotSavart(coils_boozer)
bs_tf.x = x
current_sum = sum(abs(c.current.get_value()) for c in coils_boozer)
G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

if sampleidx is None:
    mpol = 16
    ntor = 16
    phis = np.linspace(0, 1/4, ntor+1, endpoint=False)
    thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
    stellsym = True
    NFP = 2
else:
    mpol = 10
    ntor = 20
    # mpol = 16
    # ntor = 16
    phis = np.linspace(0, 1, 2*ntor+1, endpoint=False)
    thetas = np.linspace(0, 1., 2*mpol+1, endpoint=False)
    stellsym = False
    NFP = 1

phisfine = np.linspace(0, 1, 100, endpoint=False)
thetasfine = np.linspace(0, 1, 100, endpoint=False)
iota = 0.416


s = SurfaceXYZTensorFourier(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=NFP, quadpoints_phi=phis, quadpoints_theta=thetas)
s.least_squares_fit(
    SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=-phis, quadpoints_theta=-thetas).gamma()
)

sfine = SurfaceXYZTensorFourier(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=NFP, quadpoints_phi=phisfine, quadpoints_theta=thetasfine)

tf = ToroidalFlux(s, bs_tf)
tf_target = tf.J()
tf_targets = [ratio*tf_target for ratio in tf_ratios]
ar = Area(s)
ar_target = ar.J()

boozer_surface = BoozerSurface(bs, s, tf, tf_target)
res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(
    tol=1e-10, maxiter=200, constraint_weight=100., iota=iota, G=G0)
print(f"After LBFGS:   iota={res['iota']:.3f}, tf={tf.J():.3f}, area={s.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")

boozeroutdir = outdir.replace("/", "_").replace(".", "p")[:-1] + f"_seed_{sampleidx}"

# for ar_target in area_targets:
for i, tf_target in enumerate(tf_targets):
    boozer_surface = BoozerSurface(bs, s, tf, tf_target)
    if i > 0:
        s.scale(tf_target/tf_targets[i-1])
        res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(
            tol=1e-10, maxiter=200, constraint_weight=100., iota=res['iota'], G=res['G'])
        print(f"After LBFGS:   iota={res['iota']:.3f}, tf={tf.J():.3f}, area={s.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")
    res = boozer_surface.minimize_boozer_penalty_constraints_ls(
        tol=1e-10, maxiter=100, constraint_weight=100., iota=res['iota'], G=res['G'], method='manual')
    print(f"After Lev-Mar: iota={res['iota']:.3f}, tf={tf.J():.3f}, area={s.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")
    res = boozer_surface.solve_residual_equation_exactly_newton(
        tol=1e-11, maxiter=100, iota=res['iota'], G=res['G'])
    print(f"After Exact:   iota={res['iota']:.3f}, tf={tf.J():.3f}, area={s.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")
    sfine.x = s.x
    fluxerr = np.mean(np.abs(
        np.sum(
        bs.set_points(sfine.gamma().reshape((-1, 3))).B() * sfine.normal().reshape((-1, 3)),
        axis=1)))
    print(fluxerr)

    np.savetxt("outputboozer/" + boozeroutdir + f"_{tf_ratios[i]:.2f}.txt", s.x)
    s.to_vtk("outputboozer/" + boozeroutdir + f"_{tf_ratios[i]:.2f}")
