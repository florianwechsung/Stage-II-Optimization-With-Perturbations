from simsopt._core.graph_optimizable import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec
from simsopt.geo.jit import jit
import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux, CoilOptObjective
from simsopt.geo.curve import RotatedCurve, curves_to_vtk
from simsopt.geo.curvecorrected import CurveCorrected
from simsopt.geo.multifilament import CurveShiftedRotated, FilamentRotation
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Current, Coil, ScaledCurrent, coils_via_symmetries
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.geo.curveperturbed import GaussianSampler, CurvePerturbed, PerturbationSample
from simsopt.field.tracing import parallel_loop_bounds
from randomgen import SeedSequence, PCG64
import jax.numpy as jnp
from jax import grad, vjp
from mpi4py import MPI

def sum_across_comm(derivative, comm):
    newdict = {}
    for k in derivative.data.keys():
        data = derivative.data[k]
        alldata = sum(comm.allgather(data))
        if isinstance(alldata, float):
            alldata = np.asarray([alldata])
        newdict[k] = alldata
    return Derivative(newdict)


@jit
def curve_msc_pure(kappa, gammadash):
    """
    This function is used in a Python+Jax implementation of the curve arclength variation.
    """
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return jnp.mean(kappa**2 * arc_length)/jnp.mean(arc_length)


class MeanSquareCurvature(Optimizable):

    def __init__(self, curve, threshold):
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[curve])
        self.curve = curve
        self.threshold = threshold
        self.thisgrad0 = jit(lambda kappa, gammadash: grad(curve_msc_pure, argnums=0)(kappa, gammadash))
        self.thisgrad1 = jit(lambda kappa, gammadash: grad(curve_msc_pure, argnums=1)(kappa, gammadash))

    def msc(self):
        return float(curve_msc_pure(self.curve.kappa(), self.curve.gammadash()))

    def J(self):
        return 0.5 * max(self.msc()-self.threshold, 0)**2

    @derivative_dec
    def dJ(self):
        grad0 = self.thisgrad0(self.curve.kappa(), self.curve.gammadash())
        grad1 = self.thisgrad1(self.curve.kappa(), self.curve.gammadash())
        deriv = self.curve.dkappa_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1)
        fak = max(self.msc()-self.threshold, 0.)
        return fak * deriv


class QuadraticCurveLength(Optimizable):

    def __init__(self, Jls, threshold, alpha):
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=Jls)
        self.Jls = Jls
        self.threshold = threshold
        self.alpha = alpha

    def J(self):
        sumlen = sum([J.J() for J in self.Jls])
        return (self.alpha*np.maximum(sumlen-self.threshold, 0))**2

    @derivative_dec
    def dJ(self):
        sumlen = sum([J.J() for J in self.Jls])
        dsumlen = sum([J.dJ(partials=True) for J in self.Jls], start=Derivative({}))
        return 2*self.alpha*self.alpha*np.maximum(sumlen-self.threshold, 0)*dsumlen


@jit
def curve_arclengthvariation_pure(l, mat):
    """
    This function is used in a Python+Jax implementation of the curve arclength variation.
    """
    return jnp.var(mat @ l)


class UniformArclength():

    def __init__(self, curve, start=0):
        self.curve = curve
        nquadpoints = len(curve.quadpoints)
        nquadpoints_constraint = curve.full_dof_size//3 - 1
        indices = np.floor(np.linspace(start, nquadpoints, nquadpoints_constraint+1, endpoint=True)).astype(int)
        mat = np.zeros((nquadpoints_constraint, nquadpoints))
        for i in range(nquadpoints_constraint):
            mat[i, indices[i]:indices[i+1]] = 1/(indices[i+1]-indices[i])
        self.mat = mat
        self.thisgrad = jit(lambda l: grad(lambda x: curve_arclengthvariation_pure(x, mat))(l))

    def J(self):
        return curve_arclengthvariation_pure(self.curve.incremental_arclength(), self.mat)

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        return self.curve.dincremental_arclength_by_dcoeff_vjp(
            self.thisgrad(self.curve.incremental_arclength()))


class MPIObjective(Optimizable):

    def __init__(self, Js, comm):
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=Js)
        self.Js = Js
        self.comm = comm
        self.n = np.sum(self.comm.allgather(len(self.Js)))

    def J(self):
        local_vals = [J.J() for J in self.Js]
        res = np.sum([i for o in self.comm.allgather(local_vals) for i in o])
        return res/self.n

    @derivative_dec
    def dJ(self):
        if len(self.Js) == 0:
            raise NotImplementedError("This currently only works if there is at least one objective per process.")
        local_derivs = sum([J.dJ(partials=True) for J in self.Js], start=Derivative({}))
        all_derivs = sum_across_comm(local_derivs, self.comm)
        all_derivs *= 1./self.n
        return all_derivs


def create_curves(fil=0, ig=0, nsamples=0, stoch_seed=0, sigma=1e-3, zero_mean=False, order=12, comm=MPI.COMM_WORLD, sym=False):
    ncoils = 4
    R0 = 1.1
    R1 = 0.6
    order = order
    PPP = 10
    GAUSS_SIGMA_SYS = 2*sigma if sym else sigma
    GAUSS_LEN_SYS = 0.25
    GAUSS_SIGMA_STA = sigma
    GAUSS_LEN_STA = 0.5
    nfp = 2

    h = 0.02
    NFIL = (2*fil+1)*(2*fil+1)

    def create_multifil(c):
        if fil == 0:
            return [c]
        rotation = FilamentRotation(c.quadpoints, order)
        cs = []
        for i in range(-fil, fil+1):
            for j in range(-fil, fil+1):
                cs.append(CurveShiftedRotated(c, i*h, j*h, rotation))
        return cs

    base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=PPP*order)

    sampler_systematic = GaussianSampler(base_curves[0].quadpoints, GAUSS_SIGMA_SYS, GAUSS_LEN_SYS, n_derivs=1)
    sampler_statistic = GaussianSampler(base_curves[0].quadpoints, GAUSS_SIGMA_STA, GAUSS_LEN_STA, n_derivs=1)

    if ig != 0:
        np.random.seed(ig)
        for c in base_curves:
            n = len(c.x)//3
            k = 3  # how many dofs per dim to randomly perturb
            x = c.x
            x[:k] += 0.01 * np.random.standard_normal(size=(k, ))
            x[n:n+k] += 0.01 * np.random.standard_normal(size=(k, ))
            x[2*n:2*n+k] += 0.01 * np.random.standard_normal(size=(k, ))
            c.x = x

    base_currents = []
    for i in range(ncoils):
        curr = Current(1.)
        # since the target field is zero, one possible solution is just to set all
        # currents to 0. to avoid the minimizer finding that solution, we fix one
        # of the currents
        if i == 0:
            curr.fix_all()
        base_currents.append(ScaledCurrent(curr, 1e5/NFIL))

    fil_curves = []
    fil_currents = []

    for i in range(ncoils):
        fil_curves += create_multifil(base_curves[i])
        fil_currents += NFIL * [base_currents[i]]

    coils_fil = coils_via_symmetries(fil_curves, fil_currents, nfp, True)

    seeds_sys = SeedSequence(stoch_seed).spawn(nsamples)
    seeds_sta = SeedSequence(99999+stoch_seed).spawn(nsamples)

    coils_fil_pert = []
    for j in range(*parallel_loop_bounds(comm, nsamples)):
        if j % 100 == 0:
            print(j)
        base_curves_perturbed = []
        rg = np.random.Generator(PCG64(seeds_sys[j], inc=0))
        for i in range(ncoils):
            pert = PerturbationSample(sampler_systematic, randomgen=rg)
            for k in range(NFIL):
                base_curves_perturbed.append(
                    CurvePerturbed(fil_curves[i*NFIL+k], pert, zero_mean=zero_mean))

        coils_perturbed_rep = coils_via_symmetries(base_curves_perturbed, fil_currents, nfp, True)
        if not sym:
            rg = np.random.Generator(PCG64(seeds_sta[j], inc=0))
            for i in range(nfp * ncoils * 2):
                pert = PerturbationSample(sampler_statistic, randomgen=rg)
                for k in range(NFIL):
                    c = coils_perturbed_rep[i*NFIL + k]
                    coils_perturbed_rep[i*NFIL + k] = Coil(
                        CurvePerturbed(c.curve, pert, zero_mean=zero_mean), c.current)
        coils_fil_pert.append(coils_perturbed_rep)

    return base_curves, base_currents, coils_fil, coils_fil_pert

def add_correction_to_coils(coils, correction_level, already_fixed=False):
    if correction_level == 0:
        return coils
    elif correction_level == 1: # fix curve and current dofs, add curve correction
        if not already_fixed:
            fix_all_dofs(coils)
        return [Coil(CurveCorrected(co.curve), co.current) for co in coils]
    elif correction_level == 2: # fix curve dofs, add curve correction, leave current dofs free
        if not already_fixed:
            for c in coils:
                fix_all_dofs(c.curve)
        return [Coil(CurveCorrected(co.curve), co.current) for co in coils]
    elif correction_level == 3: # fix curve and current dofs, add curve and current correction
        if not already_fixed:
            fix_all_dofs(coils)
        return [Coil(CurveCorrected(co.curve), co.current + ScaledCurrent(Current(0.), 1e5)) for co in coils]
    else:
        raise NotImplementedError()

def fix_all_dofs(optims):
    if not isinstance(optims, list):
        optims = [optims]
    for o in optims:
        for a in o._get_ancestors():
            a.fix_all()

def get_outdir(well, idx):
    if well:
        outdirs = [
            "output/well_True_lengthbound_18.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_1_order_16_alstart_0_expquad/",
            "output/well_True_lengthbound_20.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_3_order_16_alstart_0_expquad/",
            "output/well_True_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_alstart_0_expquad/",
            "output/well_True_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_5_order_16_alstart_0_expquad/",
            "output/well_True_lengthbound_18.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_alstart_0_expquad_samples_4096_sigma_0.001/",
            "output/well_True_lengthbound_20.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_1_order_16_alstart_0_expquad_samples_4096_sigma_0.001_usedetig/",
            "output/well_True_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_alstart_0_expquad_samples_4096_sigma_0.001_usedetig/",
            "output/well_True_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_alstart_0_expquad_samples_4096_sigma_0.001_usedetig/",

        ]
    else:
        outdirs = [
            "output/well_False_lengthbound_18.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_3_order_16_alstart_0_expquad/",
            "output/well_False_lengthbound_20.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_alstart_0_expquad/",
            "output/well_False_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_alstart_0_expquad/",
            "output/well_False_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_2_order_16_alstart_0_expquad/",
            "output/well_False_lengthbound_18.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_alstart_0_expquad_samples_4096_sigma_0.001_usedetig/",
            "output/well_False_lengthbound_20.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_alstart_0_expquad_samples_4096_sigma_0.001/",
            "output/well_False_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_0_order_16_alstart_0_expquad_samples_4096_sigma_0.001/",
            "output/well_False_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_alstart_0_expquad_samples_4096_sigma_0.001_usedetig/",
        ]

    # if well:
    #     outdirs = [
    #         "output/well_True_lengthbound_18.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_noalen_expquad/",
    #         "output/well_True_lengthbound_20.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_noalen_expquad/",
    #         "output/well_True_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_noalen_expquad/",
    #         "output/well_True_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_noalen_expquad/",
    #         "output/well_True_lengthbound_18.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_noalen_expquad_samples_4096_sigma_0.001_usedetig/",
    #         "output/well_True_lengthbound_20.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_0_order_16_noalen_expquad_samples_4096_sigma_0.001_usedetig/",
    #         "output/well_True_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_2_order_16_noalen_expquad_samples_4096_sigma_0.001_usedetig/",
    #         "output/well_True_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_noalen_expquad_samples_4096_sigma_0.001/",
    #     ]
    # else:

    #     outdirs = [
    #         "output/well_False_lengthbound_18.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_3_order_16_noalen_expquad/",
    #         "output/well_False_lengthbound_20.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_noalen_expquad/",
    #         "output/well_False_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_noalen_expquad/",
    #         "output/well_False_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_3_order_16_noalen_expquad/",
    #         "output/well_False_lengthbound_18.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_3_order_16_noalen_expquad_samples_4096_sigma_0.001_usedetig/",
    #         "output/well_False_lengthbound_20.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_1_order_16_noalen_expquad_samples_4096_sigma_0.001/",
    #         "output/well_False_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_4_order_16_noalen_expquad_samples_4096_sigma_0.001/",
    #         "output/well_False_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_noalen_expquad_samples_4096_sigma_0.001/",
    #     ]




    return outdirs[idx]



def minor_radius(surface):
    # see explanation for Surface.aspect_ratio in https://github.com/hiddenSymmetries/simsopt/blob/master/src/simsopt/geo/surface.py
    xyz = surface.gamma()
    x2y2 = xyz[:, :, 0]**2 + xyz[:, :, 1]**2
    dgamma1 = surface.gammadash1()
    dgamma2 = surface.gammadash2()

    # compute the average cross sectional area
    J = np.zeros((xyz.shape[0], xyz.shape[1], 2, 2))
    J[:, :, 0, 0] = (xyz[:, :, 0] * dgamma1[:, :, 1] - xyz[:, :, 1] * dgamma1[:, :, 0])/x2y2
    J[:, :, 0, 1] = (xyz[:, :, 0] * dgamma2[:, :, 1] - xyz[:, :, 1] * dgamma2[:, :, 0])/x2y2
    J[:, :, 1, 0] = 0.
    J[:, :, 1, 1] = 1.

    detJ = np.linalg.det(J)
    Jinv = np.linalg.inv(J)

    dZ_dtheta = dgamma1[:, :, 2] * Jinv[:, :, 0, 1] + dgamma2[:, :, 2] * Jinv[:, :, 1, 1]
    mean_cross_sectional_area = np.abs(np.mean(np.sqrt(x2y2) * dZ_dtheta * detJ))/(2 * np.pi)

    R_minor = np.sqrt(mean_cross_sectional_area / np.pi)
    return R_minor
