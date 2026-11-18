"""Seed repeats at the selected learning rates (seed 0 is the final run of tune_lr.py).

    python scripts/run_seeds.py --workers 4
    python scripts/run_seeds.py --lr-from results/paper/selected_lr.json --seeds 0,1,2,3,4   # skip the tuning

The seed changes the initialisation and the collocation sampling, identically for every optimizer.  Resumable.
"""
import argparse
import json
import time

import _setup  # noqa: F401
from gpulse import config as C
from gpulse.experiments import parallel, run, run_name, selected_lr
from gpulse.optimizers import OPTIMIZERS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--optimizers", default=",".join(OPTIMIZERS))
    ap.add_argument("--problems", default="epanto,efin,eheat,ecable,eplate")
    ap.add_argument("--archs", default="pinn,pikan")
    ap.add_argument("--seeds", default="1,2,3,4")
    ap.add_argument("--results", default="results")
    ap.add_argument("--lr-from", default=None, help="selected_lr.json to use (default: <results>/selected_lr.json)")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    sel = json.load(open(a.lr_from or f"{a.results}/selected_lr.json"))
    jobs = []
    for s in [int(v) for v in a.seeds.split(",")]:
        for o in a.optimizers.split(","):
            for (p, ar) in C.configs(o):
                if p in a.problems.split(",") and ar in a.archs.split(","):
                    jobs.append((f"{a.results}/runs/{run_name(p, ar, o)}_s{s}.json", p, ar, o, selected_lr(sel, p, ar, o), s))
    print(f"{len(jobs)} runs, {a.workers} workers", flush=True)
    t0 = time.time()
    for n, r in enumerate(parallel(run, jobs, a.workers), 1):
        msg = "FAILED" if "failed" in r else f"L2 {r['best']['l2']:.2e} ({r['train_s'] / 60:.1f} min)"
        print(f"[{n}/{len(jobs)} {(time.time() - t0) / 60:.0f} min] {r['task']} {r['arch']} {r['optimizer']} "
              f"lr={r['lr']:g} seed={r['seed']}: {msg}", flush=True)


if __name__ == "__main__":
    main()
