"""Error measures of a trained model against the reference (test function 1000).

On the Chebyshev-Lobatto evaluation grid (129 nodes; plate 49 x 49) with Clenshaw-Curtis weights w and spectral
differentiation D of the sampled values (the same operator for the network and the reference):
  MSE = sum w e^2 / sum w,   L2 = ||e|| / ||u||,   H1 = sqrt(||e||^2 + ||grad e||^2) / sqrt(||u||^2 + ||grad u||^2),
e = u_network - u_reference.  Bar and cable: at the final time T (reference slice u(x, T) of the space-time table).
"""
from __future__ import annotations

import numpy as np

from .reference import cheb_lobatto
from . import problems as PB


def _norms_1d(e, u, w, Dm):
    de, du = Dm @ e, Dm @ u
    l2e, l2u = np.sum(w * e ** 2), np.sum(w * u ** 2)
    return dict(mse=l2e / w.sum(), l2e=l2e, l2u=l2u, h1e=l2e + np.sum(w * de ** 2), h1u=l2u + np.sum(w * du ** 2))


def solution_errors(problem, model):
    """MSE, relative L2 and relative H1 errors."""
    xe, we = problem.xe, problem.we
    if problem.name == "eplate":
        n = int(round(np.sqrt(len(xe))))
        _, w1, Dm = cheb_lobatto(n - 1)
        P = problem.predict(model, xe[None]).reshape(n, n)
        Y = problem.Ye.reshape(n, n)
        W = np.outer(w1, w1)
        E = P - Y

        def h1(A):
            return np.sum(W * A ** 2) + np.sum(W * (Dm @ A) ** 2) + np.sum(W * (A @ Dm.T) ** 2)
        return dict(mse=float(np.sum(W * E ** 2) / W.sum()), l2=float(np.sqrt(np.sum(W * E ** 2) / np.sum(W * Y ** 2))),
                    h1=float(np.sqrt(h1(E) / h1(Y))))
    _, _, Dm = cheb_lobatto(len(xe) - 1)
    if problem.T is None:
        P, Y = problem.predict(model, problem.eval_points()), problem.Ye
    else:
        P, Y = problem.predict(model, problem.eval_points(problem.ts[-1])), problem.Ust[-1]
    s = _norms_1d(P - Y, Y, we, Dm)
    return dict(mse=float(s["mse"]), l2=float(np.sqrt(s["l2e"] / s["l2u"])), h1=float(np.sqrt(s["h1e"] / s["h1u"])))


def quantities_of_interest(problem, U):
    """Engineering quantities from the solution values U on the evaluation grid (t = T for the bar and the cable)."""
    xe = problem.xe
    if problem.name == "epanto":
        return dict(peak=np.abs(U).max())                               # peak pantograph displacement
    if problem.name == "efin":
        w = cheb_lobatto(len(xe) - 1)[1]
        return dict(max=U.max(), mean=float(np.sum(w * U) / np.sum(w)))  # max and mean fin temperature
    if problem.name == "eheat":
        _, _, Dm = cheb_lobatto(len(xe) - 1)
        k0 = PB.coef("eheat", np.array([0.0]))[0][0]
        return dict(sensor=U[len(xe) // 2], flux0=-k0 * (Dm @ U)[0])     # sensor temperature, heat flux at x = 0
    if problem.name == "ecable":
        return dict(midspan=U[len(xe) // 2], maxabs=np.abs(U).max())    # midspan deflection, max |w|
    return dict(hotspot=np.abs(U).max())                                # hot-spot temperature


def qoi_errors(problem, model):
    """Relative errors of the quantities of interest."""
    ref = quantities_of_interest(problem, problem.Ye)
    net = quantities_of_interest(problem, problem.predict(model, problem.eval_points()))
    return {k: float(abs(net[k] - ref[k]) / abs(ref[k])) for k in ref}
