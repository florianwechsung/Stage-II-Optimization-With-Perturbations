from simsopt._core.graph_optimizable import Optimizable
from simsopt._core.derivative import Derivative
import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux, FOCUSObjective
from simsopt.geo.curve import RotatedCurve, curves_to_vtk
from simsopt.geo.multifilament import CurveShiftedRotated, FilamentRotation
from simsopt.field.biotsavart import BiotSavart, Current, Coil, ScaledCurrent
from simsopt.geo.coilcollection import coils_via_symmetries, create_equally_spaced_curves
from simsopt.geo.curveobjectives import CurveLength, CoshCurveCurvature
from simsopt.geo.curveobjectives import MinimumDistance
from simsopt.geo.curveperturbed import GaussianSampler, CurvePerturbed
from simsopt.field.tracing import parallel_loop_bounds
from randomgen import SeedSequence, PCG64
from mpi4py import MPI
comm = MPI.COMM_WORLD


def sum_across_comm(derivative, comm):
    newdict = {}
    for k in derivative.data.keys():
        data = derivative.data[k]
        alldata = sum(comm.allgather(data))
        if isinstance(alldata, float):
            alldata = np.asarray([alldata])
        newdict[k] = alldata
    return Derivative(newdict)


class CoshCurveLength(Optimizable):

    def __init__(self, Jls, threshold, alpha):
        self.Jls = Jls
        self.threshold = threshold
        self.alpha = alpha

    def J(self):
        sumlen = sum([J.J() for J in self.Jls])
        return (np.cosh(self.alpha*np.maximum(sumlen-self.threshold, 0))-1)**2

    def dJ(self):
        sumlen = sum([J.J() for J in self.Jls])
        dsumlen = sum([J.dJ(partials=True) for J in self.Jls], start=Derivative({}))
        return 2*self.alpha*(np.cosh(self.alpha*np.maximum(sumlen-self.threshold, 0))-1)*np.sinh(self.alpha*np.maximum(sumlen-self.threshold, 0))*dsumlen


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

    def dJ(self):
        if len(self.Js) == 0:
            raise NotImplementedError("This currently only works if there is at least one objective per process.")
        local_derivs = sum([J.dJ() for J in self.Js], start=Derivative({}))
        all_derivs = sum_across_comm(local_derivs, self.comm)
        all_derivs *= 1./self.n
        return all_derivs


def create_curves(fil=0, ig=0, nsamples=0, stoch_seed=0, sigma=1e-3):
    ncoils = 4
    R0 = 1.0
    R1 = 0.5
    order = 10
    PPP = 15
    GAUSS_SIGMA_SYS = sigma
    GAUSS_LEN_SYS = 0.3
    GAUSS_SIGMA_STA = sigma
    GAUSS_LEN_STA = 0.6
    nfp = 2

    h = 0.02
    NFIL = (2*fil+1)*(2*fil+1)

    def create_multifil(c):
        if fil == 0:
            return [c]
        rotation = FilamentRotation(order)
        cs = []
        for i in range(-fil, fil+1):
            for j in range(-fil, fil+1):
                cs.append(CurveShiftedRotated(c, i*h, j*h, rotation))
        return cs

    base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym=True, R0=R0, R1=R1, order=order, PPP=PPP)

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
        curr = Current(1/NFIL)
        # since the target field is zero, one possible solution is just to set all
        # currents to 0. to avoid the minimizer finding that solution, we fix one
        # of the currents
        if i == 0:
            curr.fix_all()
        base_currents.append(ScaledCurrent(curr, 1e5))

    fil_curves = []
    fil_currents = []

    for i in range(ncoils):
        fil_curves += create_multifil(base_curves[i])
        fil_currents += NFIL * [base_currents[i]]

    coils_fil = coils_via_symmetries(fil_curves, fil_currents, nfp, True)

    seeds_sys = SeedSequence(stoch_seed).spawn(ncoils * nsamples)
    seeds_sta = SeedSequence(999+stoch_seed).spawn(nfp * ncoils * 2 * nsamples)
    # Jfs = []

    coils_fil_pert = []
    for j in range(*parallel_loop_bounds(comm, nsamples)):
        base_curves_perturbed = []
        for i in range(ncoils):
            for k in range(NFIL):
                rg = np.random.Generator(PCG64(seeds_sys[j*ncoils + i]))
                base_curves_perturbed.append(CurvePerturbed(fil_curves[i*NFIL+k], sampler_systematic, randomgen=rg))
        coils_perturbed_rep = coils_via_symmetries(base_curves_perturbed, fil_currents, nfp, True)

        for i in range(nfp * ncoils * 2):
            for k in range(NFIL):
                rg = np.random.Generator(PCG64(seeds_sta[j*nfp*ncoils*2 + i]))
                c = coils_perturbed_rep[i*NFIL + k]
                coils_perturbed_rep[i*NFIL + k] = Coil(
                    CurvePerturbed(c.curve, sampler_statistic, randomgen=rg), c.current)
        # full_curves_perturbed = [c.curve for c in coils_perturbed_rep]
        # curves_to_vtk(fil_curves, "/tmp/fil_curves")
        # curves_to_vtk(base_curves_perturbed, f"/tmp/base_curves_perturbed_{j}")
        # curves_to_vtk(full_curves_perturbed, f"/tmp/full_curves_perturbed_{j}")
        # Jfs.append(SquaredFlux(s, bs))
        coils_fil_pert.append(coils_perturbed_rep)

    return base_curves, base_currents, coils_fil, coils_fil_pert
