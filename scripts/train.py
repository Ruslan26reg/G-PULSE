"""Train one model.

Example (G-PULSE, bar problem, PINN, the learning rate selected in the paper, seed 0):
    python scripts/train.py --problem eheat --arch pinn --optimizer strider --lr 1e-3 --seed 0
"""
import argparse

import _setup  # noqa: F401
from gpulse.optimizers import OPTIMIZERS
from gpulse.problems import PROBLEMS
from gpulse.varpro import train

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--problem", required=True, choices=PROBLEMS)
ap.add_argument("--arch", required=True, choices=["pinn", "pikan"])
ap.add_argument("--optimizer", required=True, choices=OPTIMIZERS)
ap.add_argument("--lr", type=float, required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--iters", type=int, default=None, help="iteration budget (default: 1500 integral, 4000 PDE)")
ap.add_argument("--out", default=None, help="result file (.json; weights are written next to it as .npz)")
a = ap.parse_args()
out = a.out or f"results/single/{a.problem}_{a.arch}_{a.optimizer}_lr{a.lr:g}_s{a.seed}.json"
train(a.problem, a.arch, a.optimizer, a.lr, seed=a.seed, iters=a.iters, out=out)
print("saved", out)
