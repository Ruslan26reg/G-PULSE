"""Convergence curves of G-PULSE (seed 0, selected lr): MSE, relative L2 and relative H1 of the deployed model
(current hidden layers + least-squares head on the large batch) every 50/100 iterations.

    python scripts/run_curves.py --workers 4 [--lr-from results/paper/selected_lr.json]
"""
import argparse
import json

import _setup  # noqa: F401
from gpulse import config as C
from gpulse.experiments import parallel, run, selected_lr
from gpulse.metrics import solution_errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--lr-from", default=None)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    sel = json.load(open(a.lr_from or f"{a.results}/selected_lr.json"))
    jobs = [(f"{a.results}/curves/{p}_{ar}_strider.json", p, ar, "strider", selected_lr(sel, p, ar, "strider"), 0, None, 100,
             solution_errors, 50 if p in C.INTEGRAL else 100) for (p, ar) in C.configs("strider")]
    for r in parallel(run, jobs, a.workers):
        print(r["task"], r["arch"], "done", "FAILED" if "failed" in r else f"L2 {r['best']['l2']:.2e}", flush=True)


if __name__ == "__main__":
    main()
