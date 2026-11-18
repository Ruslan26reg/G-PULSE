"""Training with an exact least-squares output layer (variable projection, VarPro).

All five problems are linear in u and the trial solution is affine in the network output, so the residual is affine in
the coefficients c of the last linear layer:  r(c) = r0 + J^T c.  At every step
  1. the hidden features phi_k (and their input derivatives) are computed once on the collocation batch,
  2. J and r0 are assembled exactly (the residual of problems.py evaluated on every feature at once),
  3. c* = argmin ||J^T c + r0||^2 + mu ||c||^2 (column-scaled ridge least squares, QR),
  4. the loss mean(r(c*)^2) is differentiated with respect to the hidden parameters with c* held fixed
     (envelope theorem) and the optimizer updates the hidden layers.
Split head (cable and plate with PIKAN, plate with PINN): c* is computed on one batch and the loss on a fresh batch.
After training, the checkpoint with the smallest validation residual is restored and its head is recomputed on a
large batch (4096 points; 1024 for the integral equations).
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from . import autodiff as ad
from .config import settings
from .networks import F, FeatureNet
from .optimizers import make_optimizer
from .problems import Problem


class Basis:
    """The `net` passed to Problem.U.
    mode 'basis': returns the K+1 channels [0, phi_1, ..., phi_K] stacked on a leading axis (no graph), so that the
                  residual of every channel is obtained at once, and records the features;
    mode 'head' : replays the recorded features contracted with c (same points, with graph);
    mode 'eval' : fresh features contracted with c."""

    def __init__(self, fnet):
        self.fnet, self.cache, self.mode, self.i, self.c = fnet, [], "basis", 0, None

    def __call__(self, Z, d1, d2, scale):
        if self.mode == "basis":
            Fe = self.fnet.features(Z, d1, d2, scale)
            self.cache.append(Fe)
            out = {}
            for k, v in Fe.d.items():
                a = np.moveaxis(v.v, -1, 0)
                out[k] = ad.T(np.concatenate([np.zeros((1,) + a.shape[1:]), a], 0))
            return F(out)
        if self.mode == "head":
            Fe = self.cache[self.i]
            self.i += 1
        else:
            Fe = self.fnet.features(Z, d1, d2, scale)
        return F({k: ad.einsum("...k,k->...", v, self.c) for k, v in Fe.d.items()})


def assemble(problem, basis, pts, grad=True):
    """J [K, M] and r0 [M]; with grad=True the recorded features keep their graph for the following 'head' pass."""
    basis.mode, basis.cache = "basis", []
    if grad:
        R = problem.residual(basis, pts).v
    else:
        with ad.no_grad():
            R = problem.residual(basis, pts).v
    R = R.reshape(R.shape[0], -1)
    r0 = R[0]
    return R[1:] - r0[None, :], r0


def solve_head(J, r0, mu_rel):
    """argmin ||J^T c + r0||^2 + mu ||c||^2 by QR least squares on the column-scaled, ridge-augmented system."""
    A = J.T
    s = np.sqrt((A * A).sum(0)) + 1e-300
    As = A / s
    K = As.shape[1]
    Aa = np.vstack([As, math.sqrt(mu_rel) * np.eye(K)])
    ba = np.concatenate([-r0, np.zeros(K)])
    return np.linalg.lstsq(Aa, ba, rcond=None)[0] / s


def clip_grad_norm(params, max_norm):
    n = math.sqrt(sum(float((p.g ** 2).sum()) for p in params if p.g is not None))
    if n > max_norm:
        for p in params:
            if p.g is not None:
                p.g = p.g * (max_norm / (n + 1e-6))
    return n


# ---------------------------------------------------------------------------------------------- evaluation
def errors(problem, model):
    """Relative L2 error and RMSE on the evaluation grid (Clenshaw-Curtis weights), at t = T for the bar and the cable,
    engineering quantities of interest and (bar, cable) the space-time error over 11 times in [0, T]."""
    from .reference import qoi
    P, Y, w = problem.predict(model, problem.eval_points()), problem.Ye, problem.we
    e2 = (w * (P - Y) ** 2).sum(-1)
    res = dict(l2=float(np.sqrt(e2 / (w * Y ** 2).sum(-1))), rmse=float(np.sqrt(e2 / w.sum())))
    for k, v in qoi(problem.name, problem.xe if problem.xe.ndim == 1 else None, P[None]).items():
        ref = problem.qoi_ref[k]
        res[f"qoi_{k}_relerr"] = float(np.abs(v[0] - ref) / max(abs(ref), 1e-12))
        res[f"qoi_{k}_abserr_phys"] = float(np.abs(v[0] - ref) * problem.phys_scale)
    res["rmse_phys"], res["phys_unit"] = res["rmse"] * problem.phys_scale, problem.phys_unit
    if problem.T is not None:
        num = den = 0.0
        for j, tj in enumerate(problem.ts):
            Pj, Yj = problem.predict(model, problem.eval_points(tj)), problem.Ust[j]
            num = num + (w * (Pj - Yj) ** 2).sum(-1)
            den = den + (w * Yj ** 2).sum(-1)
        res["l2_spacetime"] = float(np.sqrt(num / den))
        res["rmse_spacetime"] = float(np.sqrt(num / (len(problem.ts) * w.sum())))
    return res


# ---------------------------------------------------------------------------------------------- training
def train(problem_name, arch, optimizer, lr, seed=0, iters=None, out=None, eval_every=100, stop_at=None,
          monitor=None, monitor_every=100, verbose=True):
    """Train one model.  Returns the result dictionary (also written to `out`.json, weights to `out`.npz).

    iters      iteration budget (default from config: 1500 integral equations, 4000 PDEs); the cosine schedule uses it
    stop_at    stop after this iteration (the schedule still uses `iters`); used to compare with logged runs
    monitor    callable(problem, model) -> dict, evaluated every `monitor_every` iterations on the deployed model
    """
    cfg = settings(problem_name, arch)
    iters = iters or cfg["iters"]
    mu_rel, split_head = cfg["mu_rel"], cfg["split_head"]
    rng = np.random.default_rng(seed)                                    # collocation sampling
    problem = Problem(problem_name)
    fnet = FeatureNet(arch, problem.d, cfg["size"], np.random.default_rng(100 + seed))   # initialisation
    ps = fnet.params()
    opt = make_optimizer(optimizer, ps, lr)
    basis = Basis(fnet)
    n_col = cfg["n_col"]
    vpts = problem.sample(1, cfg["n_val"], np.random.default_rng(12345))  # fixed validation points
    n_final = cfg["n_final"]

    def val_res(c):
        basis.mode, basis.c = "eval", c
        with ad.no_grad():
            r = problem.residual(basis, vpts).v
        return float(np.mean(r ** 2))

    def head_on_large_batch():
        pts = problem.sample(1, n_final, np.random.default_rng(777))
        J, r0 = assemble(problem, basis, pts, grad=False)
        return solve_head(J, r0, mu_rel)

    hist, best = [], dict(val=np.inf, it=0, state=None)
    t0 = time.time()
    it = 0
    for it in range(1, (stop_at or iters) + 1):
        opt.lr = lr * (cfg["min_ratio"] + 0.5 * (1 - cfg["min_ratio"]) * (1 + math.cos(math.pi * it / iters)))
        pts = problem.sample(1, n_col, rng)
        if split_head:                                  # head on one batch, loss on a fresh batch
            J, r0 = assemble(problem, basis, pts, grad=False)
            c = solve_head(J, r0, mu_rel)
            pts = problem.sample(1, n_col, rng)
            for p in ps:
                p.g = None
            basis.mode, basis.c = "eval", c
        else:                                           # head and loss on the same batch (features replayed)
            J, r0 = assemble(problem, basis, pts)
            c = solve_head(J, r0, mu_rel)
            for p in ps:
                p.g = None
            basis.mode, basis.c, basis.i = "head", c, 0
        r = problem.residual(basis, pts)
        if not np.all(np.isfinite(c)):
            break
        loss = ad.mean(ad.square(r))
        loss.backward()
        clip_grad_norm(ps, 5.0)
        opt.step()
        if it % eval_every == 0 or it == 1 or it == iters:
            v = val_res(c)
            hist.append(dict(it=it, loss=float(loss.v), val=v, lr=opt.lr, time=time.time() - t0))
            if v < best["val"]:
                best.update(val=v, it=it, state=[p.v.copy() for p in ps])
            if verbose:
                print(f"  [{problem_name}/{arch}/{optimizer} lr={lr:g} seed={seed}] it {it:5d} loss {float(loss.v):.3e} "
                      f"val {v:.3e} best {best['val']:.3e}@{best['it']} {time.time() - t0:.0f}s", flush=True)
        if monitor is not None and (it % monitor_every == 0 or it == 1 or it == iters):
            basis.mode, basis.c = "eval", head_on_large_batch()
            rec = monitor(problem, basis)
            rec.update(it=it, time=time.time() - t0)
            hist.append(dict(monitor=rec))
    train_s = time.time() - t0
    if best["state"] is None:
        raise RuntimeError("training diverged before the first evaluation")
    for p, v in zip(ps, best["state"]):                                  # selected checkpoint
        p.v = v
    c_final = head_on_large_batch()
    basis.mode, basis.c = "eval", c_final
    res = dict(task=problem_name, arch=arch, optimizer=optimizer, lr=lr, seed=seed, iters=iters, iters_done=it,
               train_s=train_s, best_it=best["it"], val_res=best["val"], val_res_final_head=val_res(c_final),
               params=int(sum(p.v.size for p in ps)) + len(c_final), head=len(c_final), size=cfg["size"],
               mu_rel=mu_rel, split_head=split_head)
    res["best"] = errors(problem, basis)
    from .environment import describe
    res["environment"] = describe()
    res["history"] = hist
    if out:
        save(res, ps, c_final, out)
    if verbose:
        print(f"RESULT {problem_name} {arch} {optimizer} lr={lr:g} seed={seed}: L2 {res['best']['l2']:.3e} "
              f"val {res['val_res_final_head']:.2e} ({train_s / 60:.1f} min)", flush=True)
    return res


def save(res, params, c, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f)
    np.savez(out.replace(".json", ".npz"), *[p.v for p in params], c=c)


def load_model(json_path):
    """Problem and trained model (hidden layers + head) of a saved run."""
    with open(json_path) as f:
        r = json.load(f)
    problem = Problem(r["task"])
    fnet = FeatureNet(r["arch"], problem.d, r["size"], np.random.default_rng(100 + r["seed"]))
    z = np.load(json_path.replace(".json", ".npz"))
    arrs = [z[k] for k in sorted((k for k in z.files if k.startswith("arr_")), key=lambda s: int(s[4:]))]
    for p, a in zip(fnet.params(), arrs):
        p.v = a
    model = Basis(fnet)
    model.mode, model.c = "eval", z["c"]
    return problem, model, r
