"""Hardware/software report and whether this machine can reproduce the paper's logs bit for bit.

    python scripts/env_report.py [--save environment.json]
"""
import argparse
import json

import _setup  # noqa: F401
import repro_env
from gpulse.environment import describe, exact_reproduction_possible

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--save", default=None)
a = ap.parse_args()
env = describe()
chk = exact_reproduction_possible(env)
env["exact_reproduction_check"] = chk
print(json.dumps(env, indent=1))
if chk["exact"]:
    print("\nThis environment uses the same instructions as the paper's runs: expect bit-identical results "
          "(confirm with: python scripts/check_paper.py).")
else:
    why = [k for k, v in chk.items() if k != "exact" and not v]
    print(f"\nNot the paper's floating-point environment ({', '.join(why)}); "
          + ("the CPU has no AVX-512 or the variables were overridden. " if not repro_env.EXACT_MODE else "")
          + "Results will agree statistically (within the seed spread), not bit for bit. "
            "See README, section 'Exact reproduction'.")
if a.save:
    with open(a.save, "w") as f:
        json.dump(env, f, indent=1)
    print("saved", a.save)
