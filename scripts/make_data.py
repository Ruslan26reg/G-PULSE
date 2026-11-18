"""Recompute the reference data of the test function from scratch (spectral solvers, ~20 s) and compare it with data/.

    python scripts/make_data.py            # compare only
    python scripts/make_data.py --write    # overwrite data/*.npz with the recomputed files
"""
import argparse
import os

import numpy as np

import _setup
from gpulse.problems import PROBLEMS
from gpulse.reference import make_data

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--write", action="store_true")
ap.add_argument("--function", type=int, default=1000, help="index of the test function (paper: 1000)")
ap.add_argument("--out", default="data", help="output directory (relative to the repository)")
a = ap.parse_args()
worst = 0.0
for name in PROBLEMS:
    new = make_data(name, a.function)
    os.makedirs(os.path.join(_setup.ROOT, a.out), exist_ok=True)
    path = os.path.join(_setup.ROOT, a.out, f"{name}.npz")
    if os.path.exists(path):
        old = np.load(path)
        diffs = []
        for k in old.files:
            x, y = np.asarray(new[k]), old[k]
            if y.dtype.kind == "f":
                d = float(np.max(np.abs(x - y)) / max(np.max(np.abs(y)), 1e-300))
                worst = max(worst, d)
                diffs.append(f"{k} {d:.0e}")
            elif not np.array_equal(x, y):
                diffs.append(f"{k} DIFFERENT")
        print(f"{name}: max relative difference per array: " + ", ".join(diffs))
    if a.write:
        np.savez_compressed(path, **new)
        print(f"{name}: written {path}")
print(f"worst relative difference {worst:.1e} (round-off level is ~1e-14)")
