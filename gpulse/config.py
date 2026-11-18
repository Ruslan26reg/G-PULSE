"""Experimental settings (identical for all optimizers) and the learning-rate tuning protocol."""
from __future__ import annotations

INTEGRAL = ("epanto", "efin")
PDE = ("eheat", "ecable", "eplate")
ARCHS = ("pinn", "pikan")


def settings(problem, arch):
    """Network and training settings of one problem-network configuration."""
    integral = problem in INTEGRAL
    if arch == "pinn":
        freqs = {"epanto": (1, 2, 4, 8), "efin": (1, 2, 4, 8), "eheat": (1, 2, 4, 8, 16), "ecable": (1, 2, 4),
                 "eplate": (1, 2, 4, 8)}[problem]
        size = dict(width=128, depth=4, freqs=list(freqs), tfeat=[4, 16, 64] if problem == "eheat" else [])
        mu_rel = 1e-10 if problem in ("ecable", "eplate") else 1e-12
        split_head = problem == "eplate"
    elif arch == "pikan":
        size = dict(kan_hidden=[32, 32, 32])          # Chebyshev degree 5, tanh normalisation
        mu_rel = 1e-8 if problem in ("ecable", "eplate") else 1e-12
        split_head = problem in ("ecable", "eplate")
    else:
        raise ValueError(arch)
    return dict(size=size, mu_rel=mu_rel, split_head=split_head,
                iters=1500 if integral else 4000,      # iteration budget
                n_col=256 if integral else 1024,       # collocation points per step
                n_val=256 if integral else 512,        # fixed validation points (checkpoint selection)
                n_final=1024 if integral else 4096,    # points for the final head
                min_ratio=0.02)                        # cosine schedule from lr to 0.02 lr


# learning-rate tuning protocol (Section "Optimizers and tuning protocol")
LR_GRID = dict(adamw=[3e-4, 1e-3, 3e-3, 1e-2], yogi=[3e-4, 1e-3, 3e-3, 1e-2], adabelief=[3e-4, 1e-3, 3e-3, 1e-2],
               muon=[1e-3, 3e-3, 1e-2, 3e-2], strider=[3e-4, 1e-3, 3e-3, 1e-2], strider_o=[3e-4, 1e-3, 3e-3, 1e-2])
EDGE_FACTOR = dict(yogi=10.0)                         # grid extension step (default x3)
EDGE_STEPS = 6                                        # at most 6 extra points while the residual improves
TUNE_FRACTION = 4                                     # tuning runs use 1/4 of the budget
FALLBACK_RATIO = 3.0                                  # instability fallback threshold
SEEDS = (0, 1, 2, 3, 4)


def configs(optimizer):
    """Problem-network configurations on which an optimizer is evaluated (all ten for every optimizer)."""
    return [(p, a) for p in INTEGRAL + PDE for a in ARCHS]
