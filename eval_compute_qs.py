#!/usr/bin/env python
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.biotsavart import BiotSavart
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.surfaceobjectives import boozer_surface_residual, ToroidalFlux, Area
from objective import create_curves
import numpy as np

well = True
if not well:
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
area_targets = np.flipud(np.linspace(2, starget.area(), 20, endpoint=True))
tf_ratios = [(i/20) for i in reversed(range(1, 21))]

base_curves, base_currents, coils_fil, coils_fil_pert = create_curves(
    fil=fil, ig=0, nsamples=0, stoch_seed=0, sigma=0, order=16)

nonqs = []
fluxs = []
iotas = []
for idx in range(len(outdirs)):
    nonqsthis = []
    iotasthis = []
    outdir = get_outdir(well, idx)
    x = np.loadtxt(outdir + "xmin.txt")
    coils_boozer = coils_fil
    bs = BiotSavart(coils_boozer)
    bs.x = x
    # mindist = 1e10
    # for c in coils_boozer:
    #     dist = np.linalg.norm(
    #         c.curve.gamma()[None, :, :] - starget.gamma().reshape((-1, 3))[:, None, :], axis=2
    #     )
    #     mindist = min(np.min(dist), mindist)
    # print(mindist)
    fluxs.append(SquaredFlux(starget, bs).J())

    bs_tf = BiotSavart(coils_boozer)
    bs_tf.x = x
    current_sum = sum(abs(c.current.get_value()) for c in coils_boozer)
    G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

    mpol = 15
    ntor = 15
    stellsym = True
    iota = 0.416
    phis = np.linspace(0, 1/(2*nfp), ntor+1, endpoint=False)
    thetas = np.linspace(0, 1., 2*mpol+1, endpoint=False)
    NFP = nfp

    s = SurfaceXYZTensorFourier(
        mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=NFP, quadpoints_phi=phis, quadpoints_theta=thetas)
    s.least_squares_fit(
        SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=-phis, quadpoints_theta=-thetas).gamma()
    )

    tf = ToroidalFlux(s, bs_tf)
    tf_target = tf.J()
    tf_targets = [ratio*tf_target for ratio in tf_ratios]
    ar = Area(s)
    ar_target = ar.J()

    def magnetic_field_on_surface(s, bs):
        x = s.gamma()
        B = bs.set_points(x.reshape((-1, 3))).B().reshape(x.shape)
        mod_B = np.linalg.norm(B, axis=2)
        return mod_B

    def compute_non_quasisymmetry_L2(s, bs):
        x = s.gamma()
        B = bs.set_points(x.reshape((-1, 3))).B().reshape(x.shape)
        mod_B = np.linalg.norm(B, axis=2)
        n = np.linalg.norm(s.normal(), axis=2)
        mean_phi_mod_B = np.mean(mod_B*n, axis=0)/np.mean(n, axis=0)
        mod_B_QS = mean_phi_mod_B[None, :]
        mod_B_non_QS = mod_B - mod_B_QS
        non_qs = np.mean(mod_B_non_QS**2 * n)**0.5
        qs = np.mean(mod_B_QS**2 * n)**0.5
        return non_qs, qs

    boozer_surface = BoozerSurface(bs, s, tf, tf_target)
    res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(
        tol=1e-10, maxiter=100, constraint_weight=100., iota=iota, G=G0)
    print(f"After LBFGS:   iota={res['iota']:.3f}, tf={tf.J():.3f}, area={s.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")


    # for ar_target in area_targets:
    for tf_target in tf_targets:
        boozer_surface = BoozerSurface(bs, s, tf, tf_target)
        res = boozer_surface.minimize_boozer_penalty_constraints_ls(
            tol=1e-11, maxiter=100, constraint_weight=100., iota=res['iota'], G=res['G'], method='manual')
        print(f"After Lev-Mar: iota={res['iota']:.3f}, tf={tf.J():.3f}, area={s.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")
        non_qs, qs = compute_non_quasisymmetry_L2(s, bs)
        print(ar_target, ";", non_qs/qs)
        nonqsthis.append(non_qs/qs)
        iotasthis.append(res['iota'])
        s.to_vtk("/tmp/tmp")
    nonqs.append(nonqsthis)
    iotas.append(iotasthis)
print(np.array2string(np.asarray([tf_ratios] + nonqs).T, max_line_width=200, precision=5, separator=";", ))
# print(np.array2string(np.asarray([area_targets] + nonqs).T, max_line_width=200, precision=5, separator=";", ))
print(np.array2string(np.asarray(fluxs).T, max_line_width=200, precision=1, separator=";", ))
# print(np.array2string(np.asarray([area_targets] + iotas).T, max_line_width=200, precision=5, separator=";", ))
print(np.array2string(np.asarray([tf_ratios] + iotas).T, max_line_width=200, precision=5, separator=";", ))
