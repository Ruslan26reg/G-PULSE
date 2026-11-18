"""Physics-informed networks with forward Taylor-mode input derivatives.

A field and its input derivatives are kept together in `F`: key 'v' is the value, one-letter keys ('x', 't', 'y') are
first derivatives, two-letter keys ('xx', 'tt', 'yy') pure second derivatives.  Every layer maps such an `F` to the next
one, so the residual of a differential equation is available without nested automatic differentiation; gradients with
respect to the parameters are obtained by reverse mode (`autodiff`).

`FeatureNet` returns the hidden features phi_1..phi_K (+ a constant) of
  PINN  : MLP with Fourier input features and tanh activations; features = last hidden layer,
  PIKAN : Chebyshev-KAN; features = T_k(tanh h_i), k = 1..degree, of the last hidden layer.
The output u = sum_k c_k phi_k is linear in the head c, which is computed by least squares (see varpro.py).
"""
from __future__ import annotations

import math

import numpy as np

from . import autodiff as ad


class F:
    def __init__(self, d):
        self.d = d

    def __getitem__(self, k):
        return self.d[k]

    def keys(self):
        return self.d.keys()


# ---------------------------------------------------------------------------------------------- building blocks
def linear(Fin, W, b=None):
    out = {}
    for k, v in Fin.d.items():
        z = ad.einsum("...f,fo->...o", v, W)
        out[k] = ad.add(z, b) if (k == "v" and b is not None) else z
    return F(out)


def _sq(a):
    return ad.square(ad.const(a))


def tanh_taylor(Fa, d1, d2):
    """tanh with first and pure second input derivatives (chain rule)."""
    s = ad.tanh(ad.const(Fa["v"]))
    sp = ad.sub(1.0, ad.square(s))
    out = {"v": s}
    for k in d1:
        out[k] = ad.mul(sp, Fa[k])
    for k in d2:
        spp = ad.mul(ad.mul(s, sp), -2.0)
        out[k] = ad.add(ad.mul(sp, Fa[k]), ad.mul(spp, _sq(Fa[k[0]])))
    return F(out)


def chebyshev_taylor(Hin, degree, d1, d2):
    """z = tanh(h) and the Chebyshev polynomials T_0..T_degree(z) with their input derivatives.
    Returns lists Tv, Td[k], Tdd[k] (recurrences T_n = 2 z T_{n-1} - T_{n-2} differentiated by the product rule)."""
    z = ad.tanh(ad.const(Hin["v"]))
    zp = ad.sub(1.0, ad.square(z))
    Z = {"v": z}
    for k in d1:
        Z[k] = ad.mul(zp, Hin[k])
    for k in d2:
        Z[k] = ad.sub(ad.mul(zp, Hin[k]), ad.mul(ad.mul(ad.mul(z, zp), 2.0), _sq(Hin[k[0]])))
    one, zero = np.ones(z.v.shape), np.zeros(z.v.shape)
    Tv = [ad.const(one), z]
    Td = {k: [ad.const(zero), Z[k]] for k in d1}
    Tdd = {k: [ad.const(zero), Z[k]] for k in d2}
    for _ in range(2, degree + 1):
        for k in d2:                                    # T_n'' = 2 z'' T + 4 z' T' + 2 z T'' - T_{n-2}''
            a = ad.mul(ad.mul(Z[k], Tv[-1]), 2.0)
            b = ad.mul(ad.mul(Z[k[0]], Td[k[0]][-1]), 4.0)
            c = ad.mul(ad.mul(z, Tdd[k][-1]), 2.0)
            Tdd[k].append(ad.sub(ad.add(ad.add(a, b), c), Tdd[k][-2]))
        for k in d1:                                    # T_n' = 2 z' T + 2 z T' - T_{n-2}'
            Td[k].append(ad.sub(ad.add(ad.mul(ad.mul(Z[k], Tv[-1]), 2.0), ad.mul(ad.mul(z, Td[k][-1]), 2.0)), Td[k][-2]))
        Tv.append(ad.sub(ad.mul(ad.mul(z, Tv[-1]), 2.0), Tv[-2]))
    return Tv, Td, Tdd


class MLP:
    """Hidden layers of a tanh MLP (Xavier-normal initialisation)."""

    def __init__(self, rng, din, width, depth):
        self.W, self.b = [], []
        last = din
        for _ in range(depth):
            self.W.append(ad.param(rng.standard_normal((last, width)) * math.sqrt(2.0 / (last + width))))
            self.b.append(ad.param(np.zeros(width)))
            last = width

    def params(self):
        return self.W + self.b

    def __call__(self, Fx, d1, d2):
        h = Fx
        for W, b in zip(self.W, self.b):
            h = tanh_taylor(linear(h, W, b), d1, d2)
        return h


class ChebKAN:
    """Chebyshev-KAN layers: y_o = sum_{i,k} C[o, i, k] T_k(tanh x_i)."""

    def __init__(self, rng, din, widths, degree=5):
        dims = [din] + list(widths)
        self.deg = degree
        self.C = [ad.param(rng.standard_normal((b, a, degree + 1)) / math.sqrt(a * (degree + 1)))
                  for a, b in zip(dims[:-1], dims[1:])]

    def params(self):
        return list(self.C)

    def __call__(self, Fx, d1, d2):
        h = Fx
        for C in self.C:
            Tv, Td, Tdd = chebyshev_taylor(h, self.deg, d1, d2)
            out = {"v": ad.einsum("...ik,oik->...o", ad.stack(Tv, -1), C)}
            for k in d1:
                out[k] = ad.einsum("...ik,oik->...o", ad.stack(Td[k], -1), C)
            for k in d2:
                out[k] = ad.einsum("...ik,oik->...o", ad.stack(Tdd[k], -1), C)
            h = F(out)
        return h


# ---------------------------------------------------------------------------------------------- inputs
def coords(Z, d1, d2, scale):
    """Normalised coordinates Z in [-1, 1]^d and their derivatives with respect to the physical coordinates."""
    out = {"v": Z.astype(ad.DT)}
    for k in d1:
        ci, s = scale[k]
        a = np.zeros_like(Z)
        a[..., ci] = s
        out[k] = a.astype(ad.DT)
    for k in d2:
        out[k] = np.zeros_like(Z).astype(ad.DT)
    return F(out)


def fourier(Z, d1, d2, scale, freqs):
    """Fourier features [z_c, sin(pi f z_c), cos(pi f z_c)] and their derivatives (constants, no graph)."""
    feats = {"v": []}
    for k in list(d1) + list(d2):
        feats[k] = []
    for c in range(Z.shape[-1]):
        z = Z[..., c]

        def dz(k):                                      # d z_c / d (physical coordinate k)
            ci, s = scale[k[0]]
            return s if ci == c else 0.0
        feats["v"].append(z)
        for k in d1:
            feats[k].append(np.full_like(z, dz(k)))
        for k in d2:
            feats[k].append(np.zeros_like(z))
        for f in freqs:
            w = math.pi * f
            sn, cs = np.sin(w * z), np.cos(w * z)
            feats["v"] += [sn, cs]
            for k in d1:
                s = dz(k)
                feats[k] += [w * s * cs, -w * s * sn]
            for k in d2:
                s = dz(k[0])
                feats[k] += [-(w * s) ** 2 * sn, -(w * s) ** 2 * cs]
    return F({k: np.stack(v, -1).astype(ad.DT) for k, v in feats.items()})


def time_features(Fx, Z, d1, d2, scale, rates):
    """Multiscale time features exp(-a t/T) appended to the input features (bar problem, PINN)."""
    s = (Z[..., 1] + 1.0) / 2.0                         # t / T
    st = scale.get("t", (1, 0.0))[1] / 2.0              # d s / d t
    E = np.stack([np.exp(-a * s) for a in rates], -1)
    A = np.array(rates)
    add = {"v": E}
    for k in d1:
        add[k] = -A * E * st if k == "t" else np.zeros_like(E)
    for k in d2:
        add[k] = (A * st) ** 2 * E if k == "tt" else np.zeros_like(E)
    return F({k: np.concatenate([Fx[k], add[k]], -1) for k in Fx.keys()})


# ---------------------------------------------------------------------------------------------- feature network
class FeatureNet:
    """features(Z, d1, d2, scale) -> F with arrays [..., K + 1]: K hidden features and a constant (last channel)."""

    def __init__(self, arch, d, size, rng):
        self.arch = arch
        if arch == "pinn":
            self.freqs = tuple(size["freqs"])
            self.tfeat = tuple(size.get("tfeat", ()))
            nf = d * (1 + 2 * len(self.freqs)) + len(self.tfeat)
            self.body = MLP(rng, nf, size["width"], size["depth"])
        elif arch == "pikan":
            self.body = ChebKAN(rng, d, size["kan_hidden"])
        else:
            raise ValueError(arch)

    def params(self):
        return self.body.params()

    def features(self, Z, d1, d2, scale):
        if self.arch == "pinn":
            Fx = fourier(Z, d1, d2, scale, self.freqs)
            if self.tfeat:
                Fx = time_features(Fx, Z, d1, d2, scale, self.tfeat)
            H = self.body(Fx, d1, d2)
        else:
            H = self._last_basis(self.body(coords(Z, d1, d2, scale), d1, d2), d1, d2)
        return self._add_const(H)

    def _last_basis(self, Hin, d1, d2):
        """T_k(tanh h_i), k = 1..degree, flattened over (i, k): the basis of the last (least-squares) KAN layer."""
        Tv, Td, Tdd = chebyshev_taylor(Hin, self.body.deg, d1, d2)

        def flat(lst):
            s = ad.stack(lst[1:], -1)                   # T_0 is dropped (one constant is added for the whole head)
            return ad.reshape(s, s.v.shape[:-2] + (s.v.shape[-2] * s.v.shape[-1],))
        out = {"v": flat(Tv)}
        for k in d1:
            out[k] = flat(Td[k])
        for k in d2:
            out[k] = flat(Tdd[k])
        return F(out)

    @staticmethod
    def _add_const(H):
        out = {}
        for k, v in H.d.items():
            shp = v.v.shape[:-1] + (1,)
            c = np.ones(shp) if k == "v" else np.zeros(shp)
            out[k] = ad.concat([v, ad.const(c)], -1)
        return F(out)
