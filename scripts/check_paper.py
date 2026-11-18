"""Check this code against the logged runs of the paper (results/paper/).

1. Weights check (always, ~1 min): the shipped trained models (G-PULSE and G-PULSE-O, all seeds) are evaluated with this
   code and the errors are compared with the logged ones.  Tests the data, the networks and the error evaluation.
2. Rerun check: the logged runs are retrained for the first --iters iterations (same schedule, seed and learning rate)
   and the logged training loss and validation residual (iterations 1, 100, 200, ...) are compared.

    python scripts/check_paper.py                         # weights + first 100 iterations of G-PULSE, seed 0 (~5 min)
    python scripts/check_paper.py --optimizers all --iters 200 --workers 4
    python scripts/check_paper.py --iters 0               # weights only

With the same NumPy/OpenBLAS build the rerun numbers are identical to the logs (difference 0).  Other builds or CPUs
round differently; the differences then grow during training, so expect agreement to several digits early on and only
statistical agreement for complete runs.
"""
import argparse
import glob
import json
import os

import _setup
from gpulse.experiments import parallel

PAPER = os.path.join(_setup.ROOT, "results/paper")


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def check_weights(path):
    from gpulse.varpro import errors, load_model
    problem, model, r = load_model(path)
    new, old = errors(problem, model), r["best"]
    keys = [k for k in ("l2", "rmse", "l2_spacetime") if k in old] + [k for k in old if k.endswith("_relerr")]
    return os.path.basename(path), max(rel(new[k], old[k]) for k in keys), old["l2"], new["l2"]


def check_rerun(path, iters):
    from gpulse.varpro import train
    with open(path) as f:
        r = json.load(f)
    new = train(r["task"], r["arch"], r["optimizer"], r["lr"], seed=r["seed"], iters=r["iters"], stop_at=iters,
                eval_every=100, verbose=False)
    old_h = {h["it"]: h for h in r["history"] if "it" in h and h["it"] <= iters}
    new_h = {h["it"]: h for h in new["history"] if "it" in h}
    d = max(max(rel(new_h[i]["loss"], old_h[i]["loss"]), rel(new_h[i]["val"], old_h[i]["val"])) for i in old_h)
    last = max(old_h)
    return os.path.basename(path), d, last, old_h[last]["val"], new_h[last]["val"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=100, help="iterations to rerun (0: weights check only)")
    ap.add_argument("--optimizers", default="strider", help="comma list or 'all'")
    ap.add_argument("--seeds", default="0", help="comma list, e.g. 0,1,2,3,4")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tol", type=float, default=1e-6, help="relative tolerance for PASS")
    a = ap.parse_args()

    print("1. Shipped weights evaluated with this code vs logged errors (L2, RMSE, QoI)")
    worst_w = 0.0
    for name, d, l2_old, l2_new in parallel(check_weights, [(p,) for p in sorted(glob.glob(f"{PAPER}/runs/*.json"))
                                                            if os.path.exists(p.replace(".json", ".npz"))], a.workers):
        worst_w = max(worst_w, d)
        print(f"   {name:32s} L2 logged {l2_old:.6e}  recomputed {l2_new:.6e}  max rel. diff {d:.1e}", flush=True)
    print(f"   worst relative difference {worst_w:.1e} -> {'PASS' if worst_w <= a.tol else 'FAIL'}\n")

    if a.iters <= 0:
        return
    opts = None if a.optimizers == "all" else set(a.optimizers.split(","))
    seeds = {int(s) for s in a.seeds.split(",")}
    paths = []
    for p in sorted(glob.glob(f"{PAPER}/runs/*.json")):
        stem = os.path.basename(p)[:-5]
        task, arch, rest = stem.split("_", 2)
        opt, seed = rest.rsplit("_s", 1)
        if (opts is None or opt in opts) and int(seed) in seeds:
            paths.append((p, a.iters))
    print(f"2. Retraining the first {a.iters} iterations of {len(paths)} logged runs; logged vs recomputed "
          f"training loss and validation residual")
    worst = 0.0
    for name, d, last, v_old, v_new in parallel(check_rerun, paths, a.workers):
        worst = max(worst, d)
        print(f"   {name:32s} it {last:4d}: val logged {v_old:.10e} recomputed {v_new:.10e}  max rel. diff {d:.1e}",
              flush=True)
    print(f"   worst relative difference {worst:.1e} -> {'PASS' if worst <= a.tol else 'FAIL'} "
          f"({'bit-identical' if worst == 0 else 'not bit-identical: different floating-point build or CPU'})")


if __name__ == "__main__":
    main()
