"""Controlled timing: all optimizers on all configurations in one process, under identical conditions.

    python scripts/time_benchmark.py [--iters 50] [--repeats 3]      # -> <results>/timing.json

Each measurement trains for `iters` + 1 iterations (the first iteration, which includes one-off setup such as einsum
path search, is excluded) and reports milliseconds per iteration of the complete training step: features and input
derivatives, assembly and least-squares solve of the head, backward pass and optimizer update.  The order of the
optimizers is rotated between repeats.  The wall-clock times stored in the run logs depend on the machine load at the
time of each run and are not comparable across sessions; use this benchmark for cost comparisons.
"""
import argparse
import json
import os

import numpy as np

import _setup
from gpulse import config as C
from gpulse.optimizers import OPTIMIZERS
from gpulse.varpro import train


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--results", default=os.path.join(_setup.ROOT, "results/paper"))
    a = ap.parse_args()
    sel = json.load(open(f"{a.results}/selected_lr.json"))
    out = {}
    for rep in range(a.repeats):
        opts = OPTIMIZERS[rep % len(OPTIMIZERS):] + OPTIMIZERS[:rep % len(OPTIMIZERS)]
        for (p, ar) in C.configs("strider"):
            for o in opts:
                ent = sel[f"{p}/{ar}/{o}"]
                n = a.iters + 1
                r = train(p, ar, o, ent.get("lr_final", ent["lr"]), seed=0, stop_at=n, eval_every=n, verbose=False)
                t = {h["it"]: h["time"] for h in r["history"] if "it" in h}
                ms = 1000 * (t[n] - t[1]) / a.iters
                out.setdefault(f"{p}/{ar}/{o}", []).append(ms)
                print(f"repeat {rep} {p} {ar} {o}: {ms:.0f} ms/iteration", flush=True)
        with open(f"{a.results}/timing.json", "w") as f:
            json.dump(dict(iters=a.iters, repeats=rep + 1, ms_per_iteration=out), f, indent=1)
    print("mean ms/iteration:", {k: round(float(np.mean(v))) for k, v in out.items()})


if __name__ == "__main__":
    main()
