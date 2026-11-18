"""Experiment runner: resumable single runs, the learning-rate tuning protocol and parallel execution.

Results directory layout (default `results/`):
  tune/<problem>_<arch>_<opt>_lr<lr>.json    tuning runs (1/4 budget, seed 0)
  final/<problem>_<arch>_<opt>_lr<lr>.json   full-budget runs of the selected (and fallback) learning rates, seed 0
  runs/<problem>_<arch>_<opt>_s<seed>.json   the runs used in the statistics (+ .npz weights)
  selected_lr.json                           tuning record: validation residual of every learning rate tried,
                                             selected lr and, after the instability fallback, lr_final
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import traceback

from . import config as C


def run_name(problem, arch, opt):
    return f"{problem}_{arch}_{opt}"


def run(path, problem, arch, opt, lr, seed=0, iters=None, eval_every=100, monitor=None, monitor_every=100):
    """Train unless `path` exists; a failed run (exception) is recorded as {'failed': ...} instead of raising."""
    if not os.path.exists(path):
        from .varpro import train
        try:
            train(problem, arch, opt, lr, seed=seed, iters=iters, out=path, eval_every=eval_every, verbose=False,
                  monitor=monitor, monitor_every=monitor_every)
        except Exception as e:                                          # divergence etc.: kept, not dropped
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w") as f:
                json.dump(dict(task=problem, arch=arch, optimizer=opt, lr=lr, seed=seed, failed=repr(e)[:300],
                               traceback=traceback.format_exc()[-2000:]), f)
    with open(path) as f:
        return json.load(f)


def final_residual(r):
    """Validation residual of the final model (hidden layers + large-batch head); inf for failed runs."""
    v = float("inf") if "failed" in r else r.get("val_res_final_head", float("inf"))
    return v if v == v else float("inf")


def lr_key(lr):
    return f"{lr:g}"


def _log(msg):
    print(msg, flush=True)


def tune(results, problem, arch, opt, log=_log):
    """Learning-rate protocol for one optimizer on one configuration (only the learning rate is tuned):
      (i)   the initial grid with 1/4 of the budget; selection by the validation residual of the final model;
      (ii)  if the optimum is on the edge of the grid, extension in that direction by the edge factor (x3, Yogi x10),
            up to 6 points while the residual improves;
      (iii) the selected rate with the full budget; if its final residual is > 3x the tuning residual (instability that
            appears only over the longer schedule), the second-best rate is also run and the lower residual is kept.
    The selected full-budget run is copied to runs/<name>_s0.json.  Returns the tuning record."""
    name = run_name(problem, arch, opt)
    iters = C.settings(problem, arch)["iters"]
    grid = C.LR_GRID[opt]
    val = {}

    def tune_run(lr):
        r = run(f"{results}/tune/{name}_lr{lr_key(lr)}.json", problem, arch, opt, lr, 0, iters // C.TUNE_FRACTION, 25)
        val[lr] = final_residual(r)
        log(f"TUNE  {name} lr={lr:g} residual {val[lr]:.2e}")

    for lr in grid:                                                     # (i)
        tune_run(lr)
    b0 = min(grid, key=lambda lr: val[lr])
    if b0 in (min(grid), max(grid)) and val[b0] < float("inf"):        # (ii)
        fac = C.EDGE_FACTOR.get(opt, 3.0)
        fac = fac if b0 == max(grid) else 1.0 / fac
        lr, best = b0, val[b0]
        for _ in range(C.EDGE_STEPS):
            lr = float(f"{lr * fac:.3g}")
            tune_run(lr)
            if not val[lr] < best:
                break
            best = val[lr]
    lrs = sorted(val)
    sel = min(lrs, key=lambda lr: val[lr])
    rec = dict(lr=sel, val={lr_key(lr): val[lr] for lr in lrs})
    final = run(f"{results}/final/{name}_lr{lr_key(sel)}.json", problem, arch, opt, sel, 0, iters, 100)   # (iii)
    chosen, chosen_lr = final, sel
    if final_residual(final) > C.FALLBACK_RATIO * val[sel]:
        lr2 = float(sorted(rec["val"], key=lambda k: rec["val"][k])[1])
        alt = run(f"{results}/final/{name}_lr{lr_key(lr2)}.json", problem, arch, opt, lr2, 0, iters, 100)
        log(f"FALLBACK {name}: lr {sel:g} final residual {final_residual(final):.1e} > {C.FALLBACK_RATIO:g} x "
            f"{val[sel]:.1e}; lr {lr2:g} final residual {final_residual(alt):.1e}")
        if final_residual(alt) < final_residual(final):
            chosen, chosen_lr = alt, lr2
        rec["lr_final"] = chosen_lr
    src = f"{results}/final/{name}_lr{lr_key(chosen_lr)}.json"
    os.makedirs(f"{results}/runs", exist_ok=True)
    for ext in (".json", ".npz"):
        if os.path.exists(src.replace(".json", ext)):
            shutil.copy(src.replace(".json", ext), f"{results}/runs/{name}_s0{ext}")
    log(f"SELECTED {name}: lr {chosen_lr:g}" + (f" (L2 {chosen['best']['l2']:.2e})" if "best" in chosen else " (failed)"))
    return rec


def tune_job(results, problem, arch, opt):
    return (problem, arch, opt), tune(results, problem, arch, opt)


def selected_lr(sel, problem, arch, opt):
    e = sel[f"{problem}/{arch}/{opt}"]
    return e.get("lr_final", e["lr"])


# ---------------------------------------------------------------------------------------------- parallel execution
def _call(job):
    fn, args = job
    return fn(*args)


def parallel(fn, jobs, workers):
    """Run fn(*args) for every args in jobs (one thread per process); yields results as they finish."""
    os.environ["OMP_NUM_THREADS"] = "1"
    if workers <= 1:
        for args in jobs:
            yield fn(*args)
        return
    with mp.get_context("spawn").Pool(workers) as pool:
        yield from pool.imap_unordered(_call, [(fn, args) for args in jobs])
