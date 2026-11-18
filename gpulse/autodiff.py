"""Minimal reverse-mode automatic differentiation on NumPy arrays (float64).

Only what the experiments need: broadcasting arithmetic, tanh, square, sums, reshape, slicing, concatenation,
stacking and einsum.  A value is wrapped in `T`; parameters are created with `param` and receive their gradient in
`.g` after `loss.backward()`.
"""
from __future__ import annotations

import numpy as np

DT = np.float64
_GRAD = [True]


class no_grad:
    """Context manager: operations inside do not record the graph."""

    def __enter__(self):
        self.prev = _GRAD[0]
        _GRAD[0] = False

    def __exit__(self, *args):
        _GRAD[0] = self.prev


class T:
    """A node of the computational graph: value `v`, gradient `g`, parents `prev`, backward rule `back`."""

    __slots__ = ("v", "g", "prev", "back", "req")
    __array_ufunc__ = None          # ndarray (op) T is dispatched to T.__r*__ and keeps the graph

    def __init__(self, v, prev=(), back=None, req=False):
        self.v = v if (isinstance(v, np.ndarray) and v.dtype == DT) else np.asarray(v, dtype=DT)
        self.g = None
        self.prev = prev
        self.back = back
        self.req = req

    @property
    def shape(self):
        return self.v.shape

    def backward(self):
        topo, seen = [], set()                          # iterative topological sort
        stack = [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                topo.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node.prev:
                if p.req and id(p) not in seen:
                    stack.append((p, False))
        self.g = np.ones_like(self.v)
        for node in reversed(topo):
            if node.back is not None and node.g is not None:
                node.back(node.g)
                if node.prev and node is not self:      # free intermediate gradients (parameters keep theirs)
                    node.g = None

    def __add__(self, o): return add(self, o)
    def __radd__(self, o): return add(o, self)
    def __sub__(self, o): return sub(self, o)
    def __rsub__(self, o): return sub(o, self)
    def __mul__(self, o): return mul(self, o)
    def __rmul__(self, o): return mul(o, self)
    def __truediv__(self, o): return div(self, o)
    def __rtruediv__(self, o): return div(o, self)
    def __neg__(self): return mul(self, -1.0)
    def __getitem__(self, idx): return getitem(self, idx)
    def sum(self, axis=None, keepdims=False): return tsum(self, axis, keepdims)


def param(v):
    return T(np.asarray(v, dtype=DT), req=True)


def const(v):
    return v if isinstance(v, T) else T(v)


def _acc(p, g):
    if p.req:
        g = _unbroadcast(g, p.v.shape)
        p.g = g if p.g is None else p.g + g


def _unbroadcast(g, shape):
    if g.shape == shape:
        return g
    while g.ndim > len(shape):
        g = g.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(i, keepdims=True)
    return g


def _out(v, prev, back):
    if _GRAD[0] and any(p.req for p in prev):
        return T(v, prev, back, True)
    return T(v)


# ------------------------------------------------------------------ arithmetic
def add(a, b):
    a, b = const(a), const(b)
    return _out(a.v + b.v, (a, b), lambda g: (_acc(a, g), _acc(b, g)))


def sub(a, b):
    a, b = const(a), const(b)
    return _out(a.v - b.v, (a, b), lambda g: (_acc(a, g), _acc(b, -g)))


def mul(a, b):
    a, b = const(a), const(b)
    return _out(a.v * b.v, (a, b), lambda g: (_acc(a, g * b.v), _acc(b, g * a.v)))


def div(a, b):
    a, b = const(a), const(b)
    return _out(a.v / b.v, (a, b), lambda g: (_acc(a, g / b.v), _acc(b, -g * a.v / (b.v * b.v))))


def tanh(x):
    y = np.tanh(x.v)
    return _out(y, (x,), lambda g: _acc(x, g * (1.0 - y * y)))


def square(x):
    return _out(x.v * x.v, (x,), lambda g: _acc(x, 2.0 * g * x.v))


# ------------------------------------------------------------------ reductions and shapes
def tsum(x, axis=None, keepdims=False):
    v = x.v.sum(axis=axis, keepdims=keepdims)

    def back(g):
        if axis is not None and not keepdims:
            ax = axis if isinstance(axis, tuple) else (axis,)
            for a in sorted(a % x.v.ndim for a in ax):
                g = np.expand_dims(g, a)
        _acc(x, np.broadcast_to(g, x.v.shape))

    return _out(v, (x,), back)


def mean(x):
    return mul(tsum(x), 1.0 / x.v.size)


def reshape(x, shape):
    s0 = x.v.shape
    return _out(x.v.reshape(shape), (x,), lambda g: _acc(x, g.reshape(s0)))


def getitem(x, idx):
    def back(g):
        if x.req:
            full = np.zeros_like(x.v)
            full[idx] = g
            _acc(x, full)

    return _out(x.v[idx], (x,), back)


def concat(xs, axis=-1):
    xs = [const(t) for t in xs]
    v = np.concatenate([t.v for t in xs], axis=axis)
    sizes = np.cumsum([0] + [t.v.shape[axis] for t in xs])

    def back(g):
        for t, a, b in zip(xs, sizes[:-1], sizes[1:]):
            sl = [slice(None)] * g.ndim
            sl[axis] = slice(a, b)
            _acc(t, g[tuple(sl)])

    return _out(v, tuple(xs), back)


def stack(xs, axis=-1):
    xs = [const(t) for t in xs]
    ax = axis if axis >= 0 else xs[0].v.ndim + 1 + axis
    v = np.stack([t.v for t in xs], axis=ax)

    def back(g):
        for i, t in enumerate(xs):
            _acc(t, np.take(g, i, axis=ax))

    return _out(v, tuple(xs), back)


# ------------------------------------------------------------------ einsum (contraction paths cached)
_PATHS: dict = {}


def _ein(subs, *vals):
    key = (subs, tuple(v.shape for v in vals))
    path = _PATHS.get(key)
    if path is None:
        path = np.einsum_path(subs, *vals, optimize="greedy")[0]
        _PATHS[key] = path
    return np.einsum(subs, *vals, optimize=path)


def einsum(subs, *ops):
    ops = [const(o) for o in ops]
    ins, out = subs.split("->")
    ins = ins.split(",")
    v = _ein(subs, *[o.v for o in ops])

    def back(g):
        for i, o in enumerate(ops):
            if not o.req:
                continue
            others = [ins[j] for j in range(len(ops)) if j != i]
            ovals = [ops[j].v for j in range(len(ops)) if j != i]
            pool = out + "".join(others)
            keep = "".join(c for c in ins[i] if c in pool)
            gi = _ein(",".join([out] + others) + "->" + keep, g, *ovals)
            if keep != ins[i]:                          # indices private to operand i: broadcast back
                for ax, c in enumerate(ins[i]):
                    if c not in keep:
                        gi = np.expand_dims(gi, ax)
                gi = np.broadcast_to(gi, o.v.shape)
            _acc(o, gi)

    return _out(v, tuple(ops), back)
