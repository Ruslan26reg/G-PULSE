"""Optimizers (NumPy).  Each one updates the parameters `p.v` in place from their gradients `p.g`; the learning rate
attribute `lr` is set by the schedule before every step.

  strider    G-PULSE (Algorithm 1 of the paper; developed under the name Strider, hence the identifier)
  strider_o  G-PULSE-O: G-PULSE whose preconditioned direction r_t is orthogonalised by Newton-Schulz iterations
             (matrices and matricised KAN coefficient tensors); step normalisation and update clipping as in G-PULSE
  adamw      Adam with decoupled weight decay 1e-6
  yogi       Yogi
  adabelief  AdaBelief
  muon       Muon: Newton-Schulz orthogonalised Nesterov momentum for matrices; KAN coefficient tensors
             [out, in, K+1] are matricised to [out, in (K+1)]; vectors are updated by Adam
"""
from __future__ import annotations

import numpy as np

NS_COEF = (3.4445, -4.7750, 2.0315)                     # quintic Newton-Schulz coefficients of Muon


def newton_schulz(m, steps=5, eps=1e-7):
    """Approximate orthogonal polar factor of a matrix (Muon's iteration on the Frobenius-normalised matrix)."""
    tr = m.shape[0] > m.shape[1]
    if tr:
        m = m.T
    m = m / (np.linalg.norm(m) + eps)
    a, b, c = NS_COEF
    for _ in range(steps):
        mm = m @ m.T
        m = a * m + (b * mm + c * (mm @ mm)) @ m
    return m.T if tr else m


class _Base:
    def __init__(self, params, lr):
        self.ps, self.lr = params, lr
        self.st = [None] * len(params)

    def _state(self, i, p, keys):
        if self.st[i] is None:
            self.st[i] = dict(t=0, **{k: np.zeros_like(p.v) for k in keys})
        return self.st[i]


class Strider(_Base):
    """G-PULSE (identifier 'strider').  ortho=True gives G-PULSE-O ('strider_o')."""

    def __init__(self, params, lr, betas=(0.7, 0.9, 0.999, 0.95), gamma=0.05, eps=1e-12, delta=1e-8,
                 max_update_norm=1.0, ortho=False, ns_steps=5):
        super().__init__(params, lr)
        self.b0, self.b1, self.b2, self.b3 = betas
        self.gamma, self.eps, self.delta, self.mun = gamma, eps, delta, max_update_norm
        self.ortho, self.ns = ortho, ns_steps

    def _orthogonalise(self, r):
        m = r.reshape(r.shape[0], -1)                   # [out, in] or [out, in (K+1)]
        if min(m.shape) < 2:
            return r
        return newton_schulz(m, self.ns).reshape(r.shape)

    def step(self):
        for i, p in enumerate(self.ps):
            if p.g is None:
                continue
            g = p.g
            s = self._state(i, p, ["m", "m_prev", "u", "u_max", "g_prev", "s"])
            s["t"] += 1
            t = s["t"]
            s["m_prev"] = s["m"].copy()
            s["m"] = self.b1 * s["m"] + (1 - self.b1) * g
            pnm = (1 + self.b0) * s["m"] - self.b0 * s["m_prev"]          # positive-negative momentum
            s["u"] = self.b2 * s["u"] + (1 - self.b2) * g * g
            s["u_max"] = np.maximum(s["u_max"], s["u"])
            q = pnm + self.gamma * (g - s["g_prev"])                        # + gradient-change term
            d = q / (s["u_max"] + self.delta / float(t) + self.eps)
            r = np.sqrt(np.mean(s["u_max"]) + self.eps) * d                 # preconditioned direction
            if self.ortho and p.v.ndim >= 2:
                r = self._orthogonalise(r)
            s["s"] = self.b3 * s["s"] + (1 - self.b3) * r * r
            eta = self.lr / (np.sqrt(np.mean(s["s"])) + self.eps)           # tensor-wise step normalisation
            upd = eta * r
            if self.mun is not None:
                n = np.linalg.norm(upd)
                if np.isfinite(n) and n > self.mun:
                    upd = upd * (self.mun / (n + self.eps))
            p.v = p.v - upd
            s["g_prev"] = g.copy()


class AdamW(_Base):
    def __init__(self, params, lr, weight_decay=1e-6, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.wd, self.b1, self.b2, self.eps = weight_decay, b1, b2, eps

    def step(self):
        for i, p in enumerate(self.ps):
            if p.g is None:
                continue
            g = p.g
            s = self._state(i, p, ["m", "v"])
            s["t"] += 1
            t = s["t"]
            p.v = p.v * (1 - self.lr * self.wd)
            s["m"] = self.b1 * s["m"] + (1 - self.b1) * g
            s["v"] = self.b2 * s["v"] + (1 - self.b2) * g * g
            mh = s["m"] / (1 - self.b1 ** t)
            vh = s["v"] / (1 - self.b2 ** t)
            p.v = p.v - self.lr * mh / (np.sqrt(vh) + self.eps)


class Yogi(_Base):
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.b1, self.b2, self.eps = b1, b2, eps

    def step(self):
        for i, p in enumerate(self.ps):
            if p.g is None:
                continue
            g = p.g
            s = self._state(i, p, ["m", "v"])
            s["t"] += 1
            t = s["t"]
            s["m"] = self.b1 * s["m"] + (1 - self.b1) * g
            s["v"] = s["v"] - (1 - self.b2) * np.sign(s["v"] - g * g) * g * g
            mh = s["m"] / (1 - self.b1 ** t)
            vh = s["v"] / (1 - self.b2 ** t)
            p.v = p.v - self.lr * mh / (np.sqrt(np.maximum(vh, self.eps)) + self.eps)


class AdaBelief(_Base):
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-12):
        super().__init__(params, lr)
        self.b1, self.b2, self.eps = b1, b2, eps

    def step(self):
        for i, p in enumerate(self.ps):
            if p.g is None:
                continue
            g = p.g
            s = self._state(i, p, ["m", "s"])
            s["t"] += 1
            t = s["t"]
            s["m"] = self.b1 * s["m"] + (1 - self.b1) * g
            bel = g - s["m"]
            s["s"] = self.b2 * s["s"] + (1 - self.b2) * bel * bel
            mh = s["m"] / (1 - self.b1 ** t)
            sh = s["s"] / (1 - self.b2 ** t)
            p.v = p.v - self.lr * mh / (np.sqrt(sh) + self.eps)


class Muon(_Base):
    def __init__(self, params, lr, momentum=0.95, ns_steps=5, b1=0.9, b2=0.999, eps=1e-8, weight_decay=1e-6,
                 max_update_norm=1.0):
        super().__init__(params, lr)
        self.mu, self.ns, self.b1, self.b2, self.eps = momentum, ns_steps, b1, b2, eps
        self.wd, self.mun = weight_decay, max_update_norm

    def step(self):
        for i, p in enumerate(self.ps):
            if p.g is None or not np.all(np.isfinite(p.g)):
                continue
            shape = p.g.shape
            g = p.g.reshape(shape[0], -1) if p.g.ndim > 2 else p.g          # matricise KAN coefficient tensors
            if g.ndim == 2 and min(g.shape) >= 2:
                s = self._state(i, p, ["buf"])
                s["buf"] = self.mu * s["buf"].reshape(g.shape) + g
                u = g + self.mu * s["buf"]                                  # Nesterov
                u = newton_schulz(u, self.ns, self.eps)
                u = u * max(1.0, u.shape[0] / max(1, u.shape[1])) ** 0.5
                n = np.linalg.norm(u)
                if np.isfinite(n) and n > self.mun:
                    u = u * (self.mun / (n + self.eps))
                p.v = p.v * (1 - self.lr * self.wd)
                p.v = p.v - self.lr * u.reshape(shape)
            else:                                                           # vectors: Adam
                g = p.g
                s = self._state(i, p, ["m", "v"])
                s["t"] += 1
                t = s["t"]
                p.v = p.v * (1 - self.lr * self.wd)
                s["m"] = self.b1 * s["m"] + (1 - self.b1) * g
                s["v"] = self.b2 * s["v"] + (1 - self.b2) * g * g
                mh = s["m"] / (1 - self.b1 ** t)
                vh = s["v"] / (1 - self.b2 ** t)
                p.v = p.v - self.lr * mh / (np.sqrt(vh) + self.eps)


OPTIMIZERS = ["adamw", "yogi", "adabelief", "muon", "strider", "strider_o"]


def make_optimizer(name, params, lr):
    if name == "strider":
        return Strider(params, lr)
    if name == "strider_o":
        return Strider(params, lr, ortho=True)
    if name == "adamw":
        return AdamW(params, lr)
    if name == "yogi":
        return Yogi(params, lr)
    if name == "adabelief":
        return AdaBelief(params, lr)
    if name == "muon":
        return Muon(params, lr)
    raise ValueError(f"unknown optimizer {name!r}; choose from {OPTIMIZERS}")
