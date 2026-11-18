"""Spectral reference solutions of the five problems (never used for training; used only to measure errors).

Random input functions: one stream (numpy default_rng(2026)) per problem, 1200 functions, of which function 1000 is the
test function solved in the paper.  Reference solvers (Chebyshev-Lobatto collocation):
  epanto  collocation of the Volterra equation (cumulative Gauss quadrature, barycentric delay interpolation), N = 160
  efin    collocation of the Fredholm equation (kernel split at s = x, Gauss quadrature), N = 160
  eheat   all discrete Sturm-Liouville modes of L (N = 256), exact delay response of every mode (method of steps)
  ecable  the same for the delayed wave equation (N = 384)
  eplate  2-D collocation, N = 96 in each direction
The solution is evaluated on 129 Chebyshev-Lobatto nodes (plate: 49 x 49) with Clenshaw-Curtis weights, and for the
bar and the cable also at 11 times in [0, T].
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import polynomial as NP
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eig, lu_factor, lu_solve

from . import problems as PB

SEED, N_FUN, TEST_FUNCTION, NT = 2026, 1200, 1000, 11
PHYS = dict(epanto=("mm", 10.0), efin=("K", 50.0), eheat=("K", 50.0), ecable=("mm", 100.0), eplate=("K", 20.0))


# ---------------------------------------------------------------------------------------------- Chebyshev tools
def cheb_lobatto(N):
    """Chebyshev-Lobatto nodes on [0, 1] (ascending), Clenshaw-Curtis weights, differentiation matrix."""
    k = np.arange(N + 1)
    t = -np.cos(np.pi * k / N)
    c = np.ones(N + 1)
    c[[0, -1]] = 2.0
    c = c * (-1.0) ** k
    X = np.tile(t, (N + 1, 1)).T
    dX = X - X.T
    Dm = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    Dm -= np.diag(Dm.sum(1))
    theta = np.pi * k / N
    w = np.zeros(N + 1)
    v = np.ones(N - 1)
    if N % 2 == 0:
        w[0] = w[N] = 1.0 / (N * N - 1)
        for j in range(1, N // 2):
            v -= 2.0 * np.cos(2 * j * theta[1:-1]) / (4 * j * j - 1)
        v -= np.cos(N * theta[1:-1]) / (N * N - 1)
    else:
        w[0] = w[N] = 1.0 / (N * N)
        for j in range(1, (N - 1) // 2 + 1):
            v -= 2.0 * np.cos(2 * j * theta[1:-1]) / (4 * j * j - 1)
    w[1:-1] = 2.0 * v / N
    return 0.5 * (t + 1.0), 0.5 * w, 2.0 * Dm


def bary_weights(xn):
    w = (-1.0) ** np.arange(len(xn))
    w[[0, -1]] *= 0.5
    return w


def bary_matrix(xn, wb, xq):
    """Barycentric interpolation matrix [len(xq), len(xn)]."""
    d = xq[:, None] - xn[None, :]
    exact = np.isclose(d, 0.0, atol=1e-15)
    d[exact] = 1.0
    M = wb[None, :] / d
    M /= M.sum(1, keepdims=True)
    r, c = np.nonzero(exact)
    M[r] = 0.0
    M[r, c] = 1.0
    return M


class Cheb:
    def __init__(self, N):
        self.x, self.w, self.Dm = cheb_lobatto(N)
        self.wb = bary_weights(self.x)
        self.N = N

    def B(self, xq):
        return bary_matrix(self.x, self.wb, np.atleast_1d(xq))


def gauss_ab(a, b, n):
    g, w = leggauss(n)
    return a + (b - a) * (g + 1) / 2, (b - a) / 2 * w


# ---------------------------------------------------------------------------------------------- random inputs
class TrigSeries:
    """f(x) = sum_k a_k cos(k pi x) + b_k sin(k pi x), amplitudes ~ (1 + k)^(-decay)."""

    def __init__(self, rng, kmax=8, decay=1.5, sine_only=False):
        k = np.arange(kmax + 1)
        amp = 1.0 / (1.0 + k) ** decay
        self.k = k
        self.a = np.zeros(kmax + 1) if sine_only else rng.standard_normal(kmax + 1) * amp
        self.b = rng.standard_normal(kmax + 1) * amp
        self.b[0] = 0.0


def trig(fs, x, d=0):
    out = []
    for f in fs:
        w = np.pi * f.k
        c, s = np.cos(np.outer(x, w)), np.sin(np.outer(x, w))
        out.append(c @ f.a + s @ f.b if d == 0 else -s @ (f.a * w) + c @ (f.b * w))
    return np.stack(out)


def windowed(gs, x):
    """1024 [x(1-x)]^5 sum_k b_k sin(k pi x)."""
    return trig(gs, x) * (1024 * (x * (1 - x)) ** 5)


class Field2D:
    def __init__(self, rng, kmax=4, decay=1.5):
        k = np.arange(kmax + 1)
        amp = 1.0 / (1.0 + k[:, None] + k[None, :]) ** decay
        self.k = k
        self.A = rng.standard_normal((kmax + 1, kmax + 1)) * amp
        self.B = rng.standard_normal((kmax + 1, kmax + 1)) * amp

    def __call__(self, x, y):
        cx, sx = np.cos(np.pi * np.outer(x, self.k)), np.sin(np.pi * np.outer(x, self.k))
        cy = np.cos(np.pi * np.outer(y, self.k))
        return cx @ self.A @ cy.T + sx @ self.B @ cy.T


def draw_inputs(name):
    """All N_FUN input functions of a problem, in the order of the random stream."""
    rng = np.random.default_rng(SEED)
    if name in ("epanto", "efin"):
        return dict(f=[TrigSeries(rng) for _ in range(N_FUN)])
    if name == "eheat":
        f0 = [TrigSeries(rng, kmax=4, decay=2.0, sine_only=True) for _ in range(N_FUN)]
        f1 = [TrigSeries(rng, kmax=4, decay=2.0, sine_only=True) for _ in range(N_FUN)]
        for f in f1:                                    # constant history: theta = theta0 on [-tau, 0]
            f.b[:] = 0.0
        return dict(f0=f0, f1=f1)
    if name == "ecable":
        return dict(f0=[TrigSeries(rng, kmax=6, decay=1.0, sine_only=True) for _ in range(N_FUN)],
                    g1=[TrigSeries(rng, kmax=6, decay=1.0, sine_only=True) for _ in range(N_FUN)])
    return dict(phi=[Field2D(rng) for _ in range(N_FUN)])


# ---------------------------------------------------------------------------------------------- integral equations
def solve_volterra(f, a, b, p, N=160):
    """y(t) - int_0^t [a y(s) + b y(p s)] ds = f(t): nodal values on Chebyshev-Lobatto nodes."""
    C = Cheb(N)
    x = C.x
    Q = np.zeros((N + 1, N + 1))                        # (Q v)_i = int_0^{x_i} v
    for i, xi in enumerate(x):
        if xi > 0:
            t, w = gauss_ab(0.0, xi, N + 2)
            Q[i] = w @ C.B(t)
    Op = np.eye(N + 1) - Q @ (a * np.eye(N + 1) + b * C.B(p * x))
    V = np.linalg.solve(Op, trig([f], x).T).T[0]
    return lambda xq: V @ C.B(xq).T


def solve_fredholm(f, a, b, p, N=160, nq=200):
    """y(x) - int_0^1 e^{-|x-s|} [a y(s) + b y(p s)] ds = f(x)."""
    C = Cheb(N)
    x = C.x
    Hm = a * np.eye(N + 1) + b * C.B(p * x)

    def kernel_rows(xs):
        R = np.zeros((len(xs), N + 1))
        for i, xi in enumerate(xs):
            for lo, hi in ((0.0, xi), (xi, 1.0)):
                if hi - lo <= 0:
                    continue
                t, w = gauss_ab(lo, hi, nq)
                R[i] += (w * np.exp(-np.abs(xi - t))) @ C.B(t)
        return R
    V = np.linalg.solve(np.eye(N + 1) - kernel_rows(x) @ Hm, trig([f], x).T).T[0]
    Hv = V @ Hm.T
    return lambda xq: trig([f], xq)[0] + Hv @ kernel_rows(np.atleast_1d(xq)).T


# ---------------------------------------------------------------------------------------------- delay PDEs (modal)
def heat_mode_eval(alpha, beta, tau, K, hist, ts):
    """T' = alpha T + beta T(t - tau), history polynomial in s = t + tau on [-tau, 0]; values at times ts in [0, K tau].
    Exact piecewise form on [k tau, (k+1) tau]:  T = p_k(s) + e^{alpha s} r_k(s),  s = t - k tau."""
    p_prev, r_prev = np.atleast_1d(np.asarray(hist, float)), np.zeros(1)
    end = NP.polyval(tau, p_prev)
    out = np.full(len(ts), np.nan)
    out[np.isclose(ts, 0.0)] = end
    for k in range(K):
        g = beta * p_prev
        P = np.zeros(1)
        dg, j = g.copy(), 0
        while np.any(dg != 0) and j < 60:
            P = NP.polyadd(P, -dg / alpha ** (j + 1))
            dg = NP.polyder(dg) if len(dg) > 1 else np.zeros(1)
            j += 1
        R = NP.polyint(beta * r_prev)
        Cc = end - NP.polyval(0.0, P)
        p_prev, r_prev = P, NP.polyadd(R, [Cc])
        sel = (ts > k * tau - 1e-15) & (ts <= (k + 1) * tau + 1e-15) & ~np.isclose(ts, 0.0)
        s = ts[sel] - k * tau
        out[sel] = NP.polyval(s, p_prev) + np.exp(alpha * s) * NP.polyval(s, r_prev)
        end = NP.polyval(tau, p_prev) + np.exp(alpha * tau) * NP.polyval(tau, r_prev)
    return out


def wave_mode_response(alpha, beta, tau, K, hist, tend):
    """T'' = alpha T + beta T(t - tau) (alpha < 0), history polynomial in s = t + tau; value at t = tend.
    On [k tau, (k+1) tau]:  T = p_k(s) + Re(e^{i w s} r_k(s)),  w = sqrt(-alpha)."""
    w = np.sqrt(-alpha)
    p_prev, r_prev = np.atleast_1d(np.asarray(hist, float)), np.zeros(1, complex)
    T0, dT0 = NP.polyval(tau, p_prev), NP.polyval(tau, NP.polyder(p_prev)) if len(p_prev) > 1 else 0.0
    k = 0
    while True:
        g = beta * p_prev
        P = np.zeros(1)
        dg, j = g.copy(), 0
        while np.any(dg != 0) and j < 60:
            P = NP.polyadd(P, -dg / alpha ** (j + 1))
            dg = NP.polyder(NP.polyder(dg)) if len(dg) > 2 else np.zeros(1)
            j += 1
        h = beta * r_prev
        Q = np.zeros(1, complex)
        dh, j = h.copy(), 0
        while np.any(dh != 0) and j < 60:
            Q = NP.polyadd(Q, (-1) ** j * dh / (2j * w) ** (j + 1))
            dh = NP.polyder(dh) if len(dh) > 1 else np.zeros(1, complex)
            j += 1
        R = NP.polyint(Q)
        P1 = NP.polyder(P) if len(P) > 1 else np.zeros(1)
        reC = T0 - NP.polyval(0.0, P)
        imC = -(dT0 - NP.polyval(0.0, P1) - NP.polyval(0.0, Q).real) / w
        r = NP.polyadd(R, [reC + 1j * imC])
        s_end = min(tau, tend - k * tau)
        e = np.exp(1j * w * s_end)
        rv, rd = NP.polyval(s_end, r), NP.polyval(s_end, NP.polyder(r)) if len(r) > 1 else 0.0
        T = NP.polyval(s_end, P) + (e * rv).real
        dT = NP.polyval(s_end, P1) + (e * (1j * w * rv + rd)).real
        if s_end < tau or abs((k + 1) * tau - tend) < 1e-14:
            return T
        p_prev, r_prev, T0, dT0 = P, r, T, dT
        k += 1


class ModalDelaySolver:
    """u_t = L[u] + c2 u(x, t - tau) (bar) or u_tt = L[u] + c2 u(x, t - tau) (cable), u = 0 at x = 0, 1:
    Chebyshev collocation of L with all discrete modes, exact delay response of each mode."""

    def __init__(self, name, N):
        x, _, Dm = cheb_lobatto(N)
        Pp, dP, s, q = PB.coef(name, x)
        Lm = (-np.diag(Pp) @ Dm @ Dm - np.diag(dP) @ Dm + np.diag(q))[1:-1, 1:-1] / s[1:-1, None]   # -L
        lam, V = eig(Lm)
        o = np.argsort(lam.real)
        self.lam, self.V = lam.real[o], V.real[:, o]
        self.Vi = np.linalg.inv(self.V)
        self.x, self.Dm, self.wb = x, Dm, bary_weights(x)
        pr = PB.PRM[name]
        self.name, self.tau, self.K = name, pr["tau"], int(round(pr["T"] / pr["tau"]))
        self.alpha, self.beta = -pr["a1"] * self.lam, pr["c2"] + 0.0 * self.lam

    def apply(self, R, F):
        """nodal history functions F [n, N+1] -> nodal solution (zero at the ends), mode responses R."""
        U = np.zeros_like(F)
        U[:, 1:-1] = ((F[:, 1:-1] @ self.Vi.T) * R) @ self.V.T
        return U

    def responses(self, hist, t):
        if self.name == "eheat":
            return np.array([heat_mode_eval(a, b, self.tau, self.K, hist, np.array([t]))[0]
                             for a, b in zip(self.alpha, self.beta)])
        return np.array([wave_mode_response(a, b, self.tau, self.K, hist, t) for a, b in zip(self.alpha, self.beta)])

    def solution(self, F0, F1, t, xq):
        """u(xq, t) for nodal history components F0 (constant part) and F1 (bar: (t + tau)/tau part; cable: t g1)."""
        h1 = [-1.0, 1.0 / self.tau] if self.name == "eheat" else [-self.tau, 1.0]
        U = self.apply(self.responses([1.0], t), F0) + self.apply(self.responses(h1, t), F1)
        return U @ bary_matrix(self.x, self.wb, xq).T


# ---------------------------------------------------------------------------------------------- elliptic
def solve_plate(phi, d, p, pts, N=96):
    """(1 + x/2) u_xx + u_yy + u_x/2 - (1 + x) u + d u(p x, y) = -Phi, u = 0 on the boundary; values at pts."""
    C = Cheb(N)
    x, Dm = C.x, C.Dm
    I = np.eye(N + 1)
    Dxx = Dm @ Dm
    Op = (np.kron(np.diag(1.0 + 0.5 * x) @ Dxx, I) + np.kron(I, Dxx) + np.kron(np.diag(0.5 + 0.0 * x) @ Dm, I)
          + np.kron(np.diag(-(1.0 + x)), I) + d * np.kron(C.B(p * x), I))
    inner = np.zeros((N + 1, N + 1), bool)
    inner[1:-1, 1:-1] = True
    idx = np.nonzero(inner.ravel())[0]
    W = np.zeros((N + 1) ** 2)
    W[idx] = lu_solve(lu_factor(Op[np.ix_(idx, idx)]), -phi(x, x).ravel()[idx])
    W = W.reshape(N + 1, N + 1)
    xs, ix = np.unique(pts[:, 0], return_inverse=True)
    ys, iy = np.unique(pts[:, 1], return_inverse=True)
    return (C.B(xs) @ W @ C.B(ys).T)[ix, iy]


# ---------------------------------------------------------------------------------------------- quantities of interest
def qoi(name, xe, Ye):
    """Engineering quantities of interest from values Ye [n_functions, n_points] on the evaluation grid."""
    if name == "epanto":
        return dict(peak=np.abs(Ye).max(-1))                       # peak pantograph displacement
    if name == "efin":
        return dict(max=Ye.max(-1))                                # maximum fin temperature
    if name == "eheat":
        _, _, Dm = cheb_lobatto(len(xe) - 1)
        kap0 = PB.coef("eheat", np.array([0.0]))[0][0]
        return dict(sensor=Ye[:, len(xe) // 2], flux0=-kap0 * (Ye @ Dm.T)[:, 0])   # theta(L/2, T), q(0, T)
    if name == "ecable":
        return dict(midspan=Ye[:, len(xe) // 2], maxabs=np.abs(Ye).max(-1))         # w(L/2, T), max |w(., T)|
    return dict(hotspot=Ye.max(-1))                                # hot-spot temperature


# ---------------------------------------------------------------------------------------------- data file
def make_data(name, fun=TEST_FUNCTION):
    """Input coefficients and reference solution of one test function (content of data/<name>.npz)."""
    xe, we, _ = cheb_lobatto(128)
    ins = draw_inputs(name)
    out = dict(function_index=fun, seed=SEED, xe=xe, we=we)
    pr = PB.PRM[name]
    if name in ("epanto", "efin"):
        f = ins["f"][fun]
        out.update(k=f.k.astype(float), a=f.a[None], b=f.b[None])
        sol = (solve_volterra if name == "epanto" else solve_fredholm)(f, pr["a"], pr["b"], pr["p"])
        out["Ye"] = sol(xe)[None]
    elif name in ("eheat", "ecable"):
        S = ModalDelaySolver(name, 256 if name == "eheat" else 384)
        if name == "eheat":
            f0, f1 = ins["f0"][fun], ins["f1"][fun]
            out.update(k=f0.k.astype(float), a0=f0.a[None], b0=f0.b[None], a1=f1.a[None], b1=f1.b[None])
            F0, F1 = trig([f0], S.x), trig([f1], S.x)
        else:
            f0, g1 = ins["f0"][fun], ins["g1"][fun]
            out.update(k=f0.k.astype(float), b0=f0.b[None], b1=g1.b[None])
            F0, F1 = windowed([f0], S.x), 4.0 * windowed([g1], S.x)
        ts = np.linspace(0.0, S.tau * S.K, NT)
        Ust = np.stack([S.solution(F0, F1, t, xe)[0] for t in ts])
        out.update(Ye=Ust[-1][None], ts=ts, Ust=Ust[None], tau=S.tau, T=S.tau * S.K)
    else:
        phi = ins["phi"][fun]
        out.update(k=phi.k.astype(float), A=phi.A[None], B=phi.B[None])
        x48, w48, _ = cheb_lobatto(48)
        X, Y = np.meshgrid(x48, x48, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel()], -1)
        out.update(xe=pts, we=np.outer(w48, w48).ravel(), Ye=solve_plate(phi, pr["d"], pr["p"], pts)[None])
    for k, v in qoi(name, out["xe"] if out["xe"].ndim == 1 else None, out["Ye"]).items():
        out["qoi_" + k] = v
    out["phys_unit"], out["phys_scale"] = PHYS[name]
    return out
