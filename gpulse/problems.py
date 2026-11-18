"""The five engineering problems with delay (nondimensional form) and their physics-informed residuals.

  epanto  Volterra, proportional delay   y(t) - int_0^t [-y(s) + 0.8 y(s/3)] ds = f(t),               t in [0, 1]
          (linearised pantograph-catenary current collection, Ockendon & Tayler 1971; lambda = 1/3)
  efin    Fredholm, proportional delay    th(x) - int_0^1 e^{-|x-s|} [0.5 th(s) + 0.5 th(s/2)] ds = f(x)
          (cooling fin with heat recovery from x/2 to x; Green's function of the fin equation)
  eheat   parabolic, constant delay       th_t = (kappa th_x)_x - th - 10 th(x, t - tau),  kappa = 1 + 3.1 (1 - x)^2,
          tau = 0.01, T = 0.04; th = 0 at x = 0, 1; history th = f0(x) on [-tau, 0]   (graded bar, delayed cooling)
  ecable  hyperbolic, constant delay      mu w_tt = w_xx - 20 mu w(x, t - tau),  mu = 1 + x/2,  tau = 0.1, T = 0.3;
          w = 0 at x = 0, 1; history w = f0(x) + t g1(x) on [-tau, 0]                (cable, delayed active control)
  eplate  elliptic, proportional arg.     (1 + x/2) u_xx + u_yy + u_x/2 - (1 + x) u + 8 u(x/2, y) = -Phi(x, y),
          u = 0 on the boundary of the unit square                                  (graded cold plate, heat pipe)

Hard constraints (trial functions):  integral equations u = N;  bar  u = f0 + (t/T) x(1-x) N;
cable  u = f0 + t g1 + (t/T)^2 x(1-x) N;  plate  u = 16 x(1-x) y(1-y) N.
The input functions (f, f0, g1, Phi) are those of test function 1000 of the random input set (seed 2026); their
coefficients and the reference solution are stored in data/<problem>.npz (see reference.py).
"""
from __future__ import annotations

import math
import os

import numpy as np

from . import autodiff as ad
from .networks import F

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PROBLEMS = ["epanto", "efin", "eheat", "ecable", "eplate"]
KIND = dict(epanto="volterra", efin="fredholm", eheat="parabolic", ecable="hyperbolic", eplate="elliptic")
PRM = dict(
    epanto=dict(a=-1.0, b=0.8, p=1.0 / 3.0),
    efin=dict(a=0.5, b=0.5, p=0.5),
    eheat=dict(a1=1.0, c2=-10.0, tau=0.01, T=0.04),
    ecable=dict(a1=1.0, c2=-20.0, tau=0.1, T=0.3),
    eplate=dict(d=8.0, p=0.5),
)
DIM = dict(epanto=1, efin=1, eheat=2, ecable=2, eplate=2)
# residual scaling: time-dependent problems are written in t/T, so that the residual is O(1)
RES_SCALE = dict(epanto=1.0, efin=1.0, eheat=PRM["eheat"]["T"], ecable=PRM["ecable"]["T"] ** 2, eplate=1.0)
NQ = 24                                                 # Gauss-Legendre nodes for the integrals
PI = math.pi


# ---------------------------------------------------------------------------------------------- coefficients
def coef(name, x):
    """(P, P', s, Q) of L[u] = ((P u_x)_x - Q u)/s for the bar and the cable."""
    if name == "eheat":                 # graded conductivity kappa = 1 + 3.1 (1 - x)^2 (Al -> steel), Biot 1
        return 1.0 + 3.1 * (1.0 - x) ** 2, -6.2 * (1.0 - x), 1.0 + 0.0 * x, 1.0 + 0.0 * x
    if name == "ecable":                # uniform tension, tapered mass per length mu = 1 + x/2
        return 1.0 + 0.0 * x, 0.0 * x, 1.0 + 0.5 * x, 0.0 * x
    raise ValueError(name)


def Lop(name, x, u, ux, uxx):
    """L[u] = (P u_xx + P' u_x - Q u)/s."""
    Pp, dP, s, q = coef(name, x)
    return (Pp * uxx + dP * ux - q * u) / s


# ---------------------------------------------------------------------------------------------- input functions
def trig(x, a, b, k, d=0):
    """sum_k a_k cos(k pi x) + b_k sin(k pi x) and its x-derivatives; x [B, n], a, b [B, K], k [K] -> [B, n]."""
    w = PI * k
    z = x[..., None] * w
    c, s = np.cos(z), np.sin(z)
    A, Bc = a[:, None, :], b[:, None, :]
    if d == 0:
        return (c * A + s * Bc).sum(-1)
    if d == 1:
        return (-s * A * w + c * Bc * w).sum(-1)
    if d == 2:
        return (-(c * A + s * Bc) * w * w).sum(-1)
    raise ValueError(d)


def windowed(x, b, k, d=0):
    """g = 1024 [x(1-x)]^5 sum_k b_k sin(k pi x) and its x-derivatives (cable initial data)."""
    zero = b * 0.0
    q = x * (1.0 - x)
    W = 1024.0 * q ** 5
    if d == 0:
        return W * trig(x, zero, b, k, 0)
    dq, d2q = 1.0 - 2.0 * x, -2.0
    W1 = 1024.0 * 5.0 * q ** 4 * dq
    if d == 1:
        return W1 * trig(x, zero, b, k, 0) + W * trig(x, zero, b, k, 1)
    W2 = 1024.0 * (20.0 * q ** 3 * dq * dq + 5.0 * q ** 4 * d2q)
    return W2 * trig(x, zero, b, k, 0) + 2.0 * W1 * trig(x, zero, b, k, 1) + W * trig(x, zero, b, k, 2)


def field2d(x, y, A, Bm, k):
    """Phi(x, y) = sum_ij A_ij cos(i pi x) cos(j pi y) + B_ij sin(i pi x) cos(j pi y)."""
    cx, sx = np.cos(PI * x[..., None] * k), np.sin(PI * x[..., None] * k)
    cy = np.cos(PI * y[..., None] * k)
    tA = (cx[..., :, None] * A[:, None, :, :]).sum(-2)
    tB = (sx[..., :, None] * Bm[:, None, :, :]).sum(-2)
    return ((tA + tB) * cy).sum(-1)


class Inputs:
    """Coefficients of the input function(s) of a problem (leading axis = function)."""

    def __init__(self, name, c):
        self.name, self.c = name, c

    def f(self, x, d=0):                                # integral equations: right-hand side
        return trig(x, self.c["a"], self.c["b"], self.c["k"], d)

    def f0(self, x, d=0):                               # bar / cable: initial profile
        if self.name == "eheat":
            return trig(x, self.c["a0"], self.c["b0"], self.c["k"], d)
        return windowed(x, self.c["b0"], self.c["k"], d)

    def f1(self, x, d=0):                               # bar: f1 (zero here) ; cable: g1 (incl. the factor 4)
        if self.name == "eheat":
            return trig(x, self.c["a1"], self.c["b1"], self.c["k"], d)
        return 4.0 * windowed(x, self.c["b1"], self.c["k"], d)

    def phi(self, x, y):                                # plate: power map
        return field2d(x, y, self.c["A"], self.c["B"], self.c["k"])


def history(name, x, t, inp, d=0):
    """Solution on [-tau, 0] (initial history) and its x-derivatives."""
    tau = PRM[name]["tau"]
    if name == "eheat":
        return inp.f0(x, d) + (t / tau) * inp.f1(x, d)
    return inp.f0(x, d) + t * inp.f1(x, d)


def gauss01(n):
    g, w = np.polynomial.legendre.leggauss(n)
    return (g + 1) / 2, w / 2


def quad_points(name, x, nq):
    """Quadrature pieces (nodes [B, n, q], weights [B, n, q]) of the integral at x [B, n]; delayed nodes = p * nodes."""
    g, w = gauss01(nq)
    X = x[..., None]
    if name == "epanto":
        return [(X * g, X * w)]
    return [(X * g, X * w), (X + (1.0 - X) * g, (1.0 - X) * w)]          # efin: split at s = x (kernel kink)


# ---------------------------------------------------------------------------------------------- residuals
def residual_integral(name, x, y, pieces, yq, yqd, inp):
    pr = PRM[name]
    integ = 0.0
    for (s, w), v, vd in zip(pieces, yq, yqd):
        ker = 1.0 if name == "epanto" else np.exp(-np.abs(x[..., None] - s))
        integ = integ + (w * ker * (pr["a"] * v + pr["b"] * vd)).sum(-1)
    return (y - integ - inp.f(x)) * RES_SCALE[name]


def residual_heat(name, x, u, ut, ux, uxx, w):
    pr = PRM[name]
    r = ut - pr["a1"] * Lop(name, x, u, ux, uxx) - pr["c2"] * w
    return r * RES_SCALE[name]


def residual_wave(name, x, u, utt, ux, uxx, w):
    pr = PRM[name]
    r = utt - pr["a1"] * Lop(name, x, u, ux, uxx) - pr["c2"] * w
    return r * RES_SCALE[name]


def residual_elliptic(name, x, y, u, ux, uxx, uyy, ushift, inp):
    pr = PRM[name]
    r = (1.0 + 0.5 * x) * uxx + uyy + (0.5 + 0.0 * x) * ux + (-(1.0 + x)) * u + pr["d"] * ushift + inp.phi(x, y)
    return r * RES_SCALE[name]


# ---------------------------------------------------------------------------------------------- problem
class Problem:
    def __init__(self, name):
        self.name, self.kind, self.d = name, KIND[name], DIM[name]
        z = np.load(os.path.join(DATA, f"{name}.npz"))
        self.coef = {k: z[k] for k in ("a", "b", "a0", "b0", "a1", "b1", "A", "B") if k in z.files}
        self.k = z["k"]
        self.xe, self.we, self.Ye = z["xe"], z["we"], z["Ye"][0]          # reference on the evaluation grid
        self.qoi_ref = {k[4:]: float(z[k][0]) for k in z.files if k.startswith("qoi_")}
        self.phys_scale, self.phys_unit = float(z["phys_scale"]), str(z["phys_unit"])
        self.T = PRM[name].get("T")
        if self.T is not None:
            self.ts, self.Ust = z["ts"], z["Ust"][0]                        # space-time slices u(x, t_j)
        self.scale = {"x": (0, 2.0), "y": (1, 2.0)}                          # d z / d (physical coordinate)
        if self.T is not None:
            self.scale["t"] = (1, 2.0 / self.T)
        self.inp = Inputs(name, {"k": self.k, **self.coef})

    @property
    def integral(self):
        return self.kind in ("volterra", "fredholm")

    def normalise(self, pts):
        if self.T is not None:
            return np.stack([2 * pts[..., 0] - 1, 2 * pts[..., 1] / self.T - 1], -1)
        return 2 * pts - 1

    def sample(self, B, n, rng):
        pts = rng.uniform(0, 1, (B, n, self.d))
        if self.T is not None:
            pts[..., 1] *= self.T
        return pts

    # ------------------------------------------------------------------ trial solution and its derivatives
    def U(self, net, pts, dirs1=(), dirs2=()):
        """u = B + G N (hard constraints) with derivatives by the Leibniz rule; `net` returns N as an F."""
        N = net(self.normalise(pts), dirs1, dirs2, self.scale)
        if self.integral:
            return N
        inp = self.inp
        x = pts[..., 0]
        if self.kind == "elliptic":
            y = pts[..., 1]
            gx, gy = x * (1 - x), y * (1 - y)
            G = {"v": 16 * gx * gy, "x": 16 * (1 - 2 * x) * gy, "y": 16 * gx * (1 - 2 * y), "xx": -32 * gy, "yy": -32 * gx}
            Bs = {}
        else:
            t, T = pts[..., 1], self.T
            q = x * (1 - x)
            if self.kind == "parabolic":
                G = {"v": t / T * q, "x": t / T * (1 - 2 * x), "xx": -2 * t / T + 0 * x, "t": q / T}
                Bs = {"v": inp.f0(x), "x": inp.f0(x, 1), "xx": inp.f0(x, 2)}
            else:
                G = {"v": (t / T) ** 2 * q, "x": (t / T) ** 2 * (1 - 2 * x), "xx": -2 * (t / T) ** 2 + 0 * x,
                     "t": 2 * t / T ** 2 * q, "tt": 2 / T ** 2 * q}
                Bs = {"v": inp.f0(x) + t * inp.f1(x), "x": inp.f0(x, 1) + t * inp.f1(x, 1),
                      "xx": inp.f0(x, 2) + t * inp.f1(x, 2), "t": inp.f1(x)}
        out = {}
        for k in N.keys():
            if k == "v":
                val = ad.mul(N["v"], G["v"])
            elif len(k) == 1:
                val = ad.add(ad.mul(N["v"], G[k]), ad.mul(N[k], G["v"]))
            else:
                val = ad.add(ad.add(ad.mul(N["v"], G[k]), ad.mul(N[k[0]], 2.0 * G[k[0]])), ad.mul(N[k], G["v"]))
            out[k] = ad.add(val, Bs[k]) if k in Bs else val
        return F(out)

    def residual(self, net, pts):
        name, inp = self.name, self.inp
        x = pts[..., 0]
        if self.integral:
            y = self.U(net, pts)["v"]
            pieces = quad_points(name, x, NQ)
            p = PRM[name]["p"]
            B, n = x.shape
            yq, yqd = [], []
            for s, _ in pieces:
                a1 = self.U(net, s.reshape(B, -1, 1))["v"]
                a2 = self.U(net, (p * s).reshape(B, -1, 1))["v"]
                yq.append(ad.reshape(a1, a1.v.shape[:-1] + (n, NQ)))     # a leading basis axis is allowed
                yqd.append(ad.reshape(a2, a2.v.shape[:-1] + (n, NQ)))
            return residual_integral(name, x, y, pieces, yq, yqd, inp)
        if self.kind == "elliptic":
            Uf = self.U(net, pts, ("x", "y"), ("xx", "yy"))
            ush = self.U(net, np.stack([PRM[name]["p"] * x, pts[..., 1]], -1))["v"]
            return residual_elliptic(name, x, pts[..., 1], Uf["v"], Uf["x"], Uf["xx"], Uf["yy"], ush, inp)
        t = pts[..., 1]
        td = t - PRM[name]["tau"]
        pd = np.stack([x, td], -1)
        m = (td >= 0).astype(ad.DT)                     # delayed point inside the domain (else: history)
        hist = lambda d: history(name, x, td, inp, d) * (1 - m)          # noqa: E731
        if self.kind == "parabolic":
            Uf = self.U(net, pts, ("x", "t"), ("xx",))
            Wv = self.U(net, pd)["v"]
            w = ad.add(ad.mul(Wv, m), hist(0))
            return residual_heat(name, x, Uf["v"], Uf["t"], Uf["x"], Uf["xx"], w)
        Uf = self.U(net, pts, ("x", "t"), ("xx", "tt"))
        Wv = self.U(net, pd)["v"]
        w = ad.add(ad.mul(Wv, m), hist(0))
        return residual_wave(name, x, Uf["v"], Uf["tt"], Uf["x"], Uf["xx"], w)

    # ------------------------------------------------------------------ evaluation
    def eval_points(self, t=None):
        """Evaluation grid (Chebyshev-Lobatto; plate 49 x 49), for the bar and the cable at time t (default T)."""
        if self.T is not None:
            t = self.T if t is None else t
            return np.stack([self.xe, np.full_like(self.xe, t)], -1)[None]
        if self.d == 1:
            return self.xe[None, :, None]
        return self.xe[None]

    def predict(self, net, pts):
        with ad.no_grad():
            return self.U(net, pts)["v"].v[0].astype(np.float64)
