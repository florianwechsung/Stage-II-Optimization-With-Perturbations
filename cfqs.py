import numpy as np
import coilpy
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curve import curves_to_vtk
focuscoils = coilpy.coils.Coil().read_makegrid('2b40c16').data
for coil_order in range(5, 25):
    curves = []
    # coil_order = 6
    maxkappa = 0
    for i, c in enumerate(focuscoils):
        xyz = np.vstack((c.x, c.y, c.z)).T[:-1,:]
        n = xyz.shape[0]
        newcurve = CurveXYZFourier(np.linspace(0, 1, n, endpoint=False), coil_order)
        newcurve.least_squares_fit(xyz)
        newcurvehighres = CurveXYZFourier(np.linspace(0, 1, 4*n, endpoint=False), coil_order)
        newcurvehighres.x = newcurve.x
        curves.append(newcurve)
        kappa = np.max(newcurve.kappa())
        # kappa = np.max(newcurvehighres.kappa())
        if kappa > maxkappa:
            maxkappa = kappa
    print(f"maxkappa for coilorder {coil_order}={maxkappa}")
# curves_to_vtk(curves, "/tmp/cfqs")
from simsopt.geo.curveobjectives import MinimumDistance
Jdist = MinimumDistance(curves, 0.1, penalty_type="cosh", alpha=1.)
print("Jdist.shortest_distance()", Jdist.shortest_distance())
