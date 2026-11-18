"""Reproduction tests against the logged runs of the paper (results/paper/), ~1 min:
    python -m unittest discover -s tests -v"""
import glob
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import repro_env  # noqa: E402,F401  (before numpy)
from gpulse.varpro import errors, load_model, train  # noqa: E402

RUNS = os.path.join(ROOT, "results/paper", "runs")


class TestPaperLogs(unittest.TestCase):
    def test_shipped_weights_reproduce_logged_errors(self):
        for name in ("epanto_pinn_strider_s0", "eheat_pikan_strider_s3", "eplate_pinn_strider_o_s1"):
            prob, model, r = load_model(f"{RUNS}/{name}.json")
            e = errors(prob, model)
            for k in ("l2", "rmse"):
                self.assertAlmostEqual(e[k] / r["best"][k], 1.0, delta=1e-9, msg=f"{name} {k}")

    def test_first_iteration_matches_log(self):
        """Loss and validation residual after the first step of every G-PULSE run (seed 0) agree with the paper's
        log: bit-identical with the same NumPy/OpenBLAS build, within 1e-6 otherwise."""
        for path in sorted(glob.glob(f"{RUNS}/*_strider_s0.json")):
            with open(path) as f:
                r = json.load(f)
            new = train(r["task"], r["arch"], r["optimizer"], r["lr"], seed=r["seed"], iters=r["iters"], stop_at=1,
                        verbose=False)
            old, cur = r["history"][0], new["history"][0]
            self.assertEqual((old["it"], cur["it"]), (1, 1))
            for k in ("loss", "val"):
                self.assertAlmostEqual(cur[k] / old[k], 1.0, delta=1e-6, msg=f"{os.path.basename(path)} {k}")


if __name__ == "__main__":
    unittest.main()
