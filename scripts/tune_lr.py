"""Learning-rate tuning protocol + full-budget seed-0 runs (identical for every optimizer).

    python scripts/tune_lr.py --workers 4                      # all optimizers, all configurations
    python scripts/tune_lr.py --optimizers strider --problems epanto --archs pinn

Resumable: finished runs are read back from the results directory.  Writes <results>/selected_lr.json and
<results>/runs/<problem>_<arch>_<opt>_s0.json.  Cost: about 2-3 full runs per optimizer and configuration
(a full run takes 5-20 min on one CPU core).
"""
import argparse
import json
import os
import time

import _setup  # noqa: F401
from gpulse import config as C
from gpulse.experiments import parallel, tune_job
from gpulse.optimizers import OPTIMIZERS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--optimizers", default=",".join(OPTIMIZERS))
    ap.add_argument("--problems", default="epanto,efin,eheat,ecable,eplate")
    ap.add_argument("--archs", default="pinn,pikan")
    ap.add_argument("--results", default="results")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    jobs = [(a.results, p, ar, o) for o in a.optimizers.split(",") for (p, ar) in C.configs(o)
            if p in a.problems.split(",") and ar in a.archs.split(",")]
    sel_path = os.path.join(a.results, "selected_lr.json")
    sel = json.load(open(sel_path)) if os.path.exists(sel_path) else {}
    t0 = time.time()
    print(f"{len(jobs)} optimizer-configuration pairs, {a.workers} workers", flush=True)

    for (p, ar, o), rec in parallel(tune_job, jobs, a.workers):
        sel[f"{p}/{ar}/{o}"] = rec
        os.makedirs(a.results, exist_ok=True)
        with open(sel_path, "w") as f:
            json.dump(sel, f, indent=1)
        print(f"[{(time.time() - t0) / 60:.0f} min] done {p}/{ar}/{o}: lr {rec.get('lr_final', rec['lr']):g}", flush=True)
    print("saved", sel_path)


if __name__ == "__main__":
    main()
