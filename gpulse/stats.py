"""Statistics over seeds and configurations (on log10 of the relative L2 error).

  cell        geometric mean over the seeds with a 95% t-interval; a failed run (exception, non-finite error or
              L2 >= 1) enters with L2 = 1, i.e. as a complete failure, so that it is not dropped
  paired      seed-paired t-test on log10 L2 (the same seed gives the same initialisation and collocation points for
              every optimizer), ratio of geometric means with 95% interval, number of seeds won
  holm        Holm step-down correction over a family of tests
  friedman    Friedman test over configurations with the Nemenyi critical difference (alpha = 0.05)
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
from scipy import stats

Q_NEMENYI_005 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949}


def load_l2(path):
    """Relative L2 error of a run (1.0 for a failed run); None if the file does not exist."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        r = json.load(f)
    if "failed" in r:
        return 1.0
    l2 = r["best"]["l2"]
    return l2 if (np.isfinite(l2) and l2 < 1.0) else 1.0


def cell(l2s):
    x = np.log10(np.asarray(l2s, float))
    n = len(x)
    m, sd = float(x.mean()), float(x.std(ddof=1))
    h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return dict(n=n, mean=m, sd=sd, gm=10 ** m, lo=10 ** (m - h), hi=10 ** (m + h),
                n_failed=int(sum(v > 0.1 for v in l2s)))


def paired(l2_a, l2_b):
    """a against b over common seeds (dicts seed -> L2); ratio < 1 means a is more accurate."""
    common = sorted(set(l2_a) & set(l2_b))
    xa = np.log10([l2_a[s] for s in common])
    xb = np.log10([l2_b[s] for s in common])
    d = xa - xb
    n = len(d)
    p = float(stats.ttest_rel(xa, xb).pvalue) if np.std(d) > 0 else (0.0 if d.mean() != 0 else 1.0)
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / math.sqrt(n)
    return dict(n=n, ratio=10 ** float(d.mean()), lo=10 ** float(d.mean() - h), hi=10 ** float(d.mean() + h), p=p,
                wins=int((d < 0).sum()))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def friedman(M):
    """M [configurations, optimizers] of mean log10 L2.  Average ranks (1 = best), chi^2, p, Nemenyi CD."""
    M = np.asarray(M)
    N, k = M.shape
    ranks = np.array([stats.rankdata(row) for row in M])
    chi2, p = stats.friedmanchisquare(*M.T)
    return dict(avg=ranks.mean(0).tolist(), chi2=float(chi2), p=float(p),
                cd=Q_NEMENYI_005[k] * math.sqrt(k * (k + 1) / (6 * N)), N=N, k=k)
