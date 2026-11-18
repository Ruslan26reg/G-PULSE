"""Figures of the paper's experimental section from a results directory (default: the paper's logs).

    python scripts/make_figures.py [--results results/paper]      (run analyze.py first: it provides summary.json)

  fig_solutions    network solutions of G-PULSE, seed 0 (needs the weights)
  fig_convergence  MSE, relative L2 and H1 of G-PULSE during training (needs <results>/curves, see run_curves.py)
  fig_seeds        log10 L2 on all seeds with mean and 95% CI, six optimizers
  fig_cd           average ranks with the Friedman test and the Nemenyi critical difference
  fig_seeds_pinn, fig_seeds_pikan, fig_ranks   the same results in the layout of the manuscript
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

import _setup  # noqa: E402,F401
from gpulse import config as C  # noqa: E402
from gpulse import stats as S  # noqa: E402

TITLE = dict(epanto="(a) Pantograph-catenary (Volterra)", efin="(b) Cooling fin (Fredholm)",
             eheat="(c) Graded bar, delayed cooling", ecable="(d) Cable, delayed control", eplate="(e) Cold plate (elliptic)")
SHORT = dict(epanto="Pantograph", efin="Fin", eheat="Bar", ecable="Cable", eplate="Plate")
ARCH = dict(pinn=("PINN", "#1f5fa8"), pikan=("PIKAN-Cheb", "#e15759"))
OPT6 = ["adamw", "yogi", "adabelief", "muon", "strider", "strider_o"]
LAB = dict(adamw="AdamW", yogi="Yogi", adabelief="AdaBelief", muon="Muon", strider="G-PULSE", strider_o="G-PULSE-O",
           strider_star="G-PULSE*")         # G-PULSE: code identifier 'strider'
COL = dict(adamw="#8c8c8c", yogi="#b07aa1", adabelief="#59a14f", muon="#e15759", strider="#1f5fa8", strider_o="#08306b",
           strider_star="#08306b")
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5})


def save(fig, out, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}/{name}.{ext}", dpi=150)
    plt.close(fig)
    print("saved", f"{out}/{name}.pdf")


def fig_solutions(R, out):
    from gpulse.varpro import load_model
    fig, ax = plt.subplots(5, 2, figsize=(7.2, 11.0))
    for i, p in enumerate(C.INTEGRAL + C.PDE):
        for k, ar in enumerate(C.ARCHS):
            f = f"{R}/runs/{p}_{ar}_strider_s0.json"
            A = ax[i, k]
            cax = make_axes_locatable(A).append_axes("right", size="4%", pad=0.06)   # equal panel sizes
            A.set_title(f"{TITLE[p]}, {ARCH[ar][0]}")
            if not os.path.exists(f.replace(".json", ".npz")):
                A.axis("off"), cax.axis("off")
                continue
            prob, model, _ = load_model(f)
            xe = prob.xe
            if p in C.INTEGRAL:
                A.plot(xe, prob.predict(model, prob.eval_points()), c=ARCH[ar][1], lw=1.4)
                A.set_xlim(0, 1)
                A.set_xlabel("t" if p == "epanto" else "x")
                A.set_ylabel("y" if p == "epanto" else r"$\theta$")
                A.grid(alpha=0.3)
                cax.set_visible(False)
            elif p in ("eheat", "ecable"):
                tf = np.linspace(0, prob.T, 61)
                P = np.stack([prob.predict(model, prob.eval_points(t)) for t in tf])
                im = A.imshow(P.T, origin="lower", aspect="auto", extent=[0, prob.T, 0, 1], cmap="viridis",
                              interpolation="bilinear")
                A.set_xlabel("t")
                A.set_ylabel("x")
                fig.colorbar(im, cax=cax, label=r"$\theta$" if p == "eheat" else "w")
            else:
                n = int(round(np.sqrt(len(xe))))
                xg = xe[:, 0].reshape(n, n)[:, 0]
                P = prob.predict(model, xe[None]).reshape(n, n)
                im = A.pcolormesh(xg, xg, P.T, shading="gouraud", cmap="viridis")
                A.set_xlabel("x")
                A.set_ylabel("y")
                fig.colorbar(im, cax=cax, label="u")
    save(fig, out, "fig_solutions")


def fig_convergence(R, out):
    fig, ax = plt.subplots(5, 3, figsize=(7.2, 10))
    for i, p in enumerate(C.INTEGRAL + C.PDE):
        for ar, (name, col) in ARCH.items():
            f = f"{R}/curves/{p}_{ar}_strider.json"
            if not os.path.exists(f):
                continue
            r = json.load(open(f))
            m = [h["monitor"] for h in r["history"] if "monitor" in h]
            it = np.array([x["it"] for x in m])
            for k, key in enumerate(("mse", "l2", "h1")):
                y = np.array([x[key] for x in m])
                A = ax[i, k]
                A.semilogy(it, y, c=col, lw=1.2, label=name)
                A.plot([r["best_it"]], [np.interp(r["best_it"], it, y)], "o", c=col, ms=3.5)   # selected checkpoint
                A.grid(alpha=0.3, which="both")
                if i == 4:
                    A.set_xlabel("iteration")
                if i == 0:
                    A.set_title(["MSE", r"relative $L^2$", r"relative $H^1$"][k])
        ax[i, 0].set_ylabel(SHORT[p])
    ax[0, 0].legend(fontsize=7)
    save(fig, out, "fig_convergence")


def fig_seeds(R, out):
    fig, axs = plt.subplots(2, 5, figsize=(7.2, 4.4))
    for j, p in enumerate(C.INTEGRAL + C.PDE):
        for i, ar in enumerate(C.ARCHS):
            A = axs[i, j]
            for k, o in enumerate(OPT6):
                l2 = [S.load_l2(f"{R}/runs/{p}_{ar}_{o}_s{s}.json") for s in C.SEEDS]
                l2 = [v for v in l2 if v is not None]
                if len(l2) < 2:
                    continue
                y = np.log10(l2)
                c = S.cell(l2)
                A.scatter(k + np.linspace(-0.15, 0.15, len(y)), y, s=6, color=COL[o], zorder=3,
                          marker="D" if o == "strider_o" else "o")
                A.errorbar(k + 0.32, c["mean"], yerr=[[c["mean"] - np.log10(c["lo"])], [np.log10(c["hi"]) - c["mean"]]],
                           fmt="_", color=COL[o], ms=6, lw=1, capsize=2)
            A.set_xticks(range(len(OPT6)))
            A.set_xticklabels([LAB[o] for o in OPT6], rotation=60, fontsize=6)
            A.set_title(f"{SHORT[p]}, {ARCH[ar][0].split('-')[0]}", fontsize=7.5)
            A.grid(axis="y", alpha=0.3)
            A.tick_params(axis="y", labelsize=6)
            if j == 0:
                A.set_ylabel(r"$\log_{10}$ relative $L^2$")
    save(fig, out, "fig_seeds")


def fig_cd(R, out, sets="abcde"):
    f = f"{R}/tables/summary.json"
    if not os.path.exists(f):
        print("fig_cd: run analyze.py first")
        return
    FL = {k: v for k, v in json.load(open(f))["friedman"].items() if k[1] in sets}
    fig, axs = plt.subplots(len(FL), 1, figsize=(7.2, 1.8 * len(FL)), squeeze=False)
    for A, (title, F) in zip(axs.ravel(), FL.items()):
        k, avg = F["k"], np.array(F["avg"])
        A.set_xlim(0.8, k + 0.2)
        A.set_ylim(0, 1.05)
        A.invert_xaxis()
        A.axhline(0.75, color="k", lw=0.8)
        for r in range(1, k + 1):
            A.plot([r, r], [0.72, 0.78], "k", lw=0.8)
            A.text(r, 0.82, str(r), ha="center", fontsize=7)
        for n, idx in enumerate(np.argsort(avg)):
            r, yy, o = avg[idx], 0.55 - 0.1 * n, F["opts"][idx]
            A.plot([r, r], [0.75, yy], color=COL[o], lw=1)
            A.text(r, yy - 0.07, f"{LAB[o]} ({r:.2f})", ha="center", fontsize=6.5, color=COL[o])
        best = avg.min()
        A.plot([best, best + F["cd"]], [0.93, 0.93], "k", lw=2)
        A.text(best + F["cd"] / 2, 0.95, f"CD = {F['cd']:.2f}", ha="center", va="bottom", fontsize=6.5)
        A.set_title(f"{title}: Friedman $\\chi^2$ = {F['chi2']:.2f}, p = {F['p']:.2g}", fontsize=7.5, pad=10)
        A.axis("off")
    save(fig, out, "fig_cd")


TAB10 = dict(adamw="#1f77b4", yogi="#ff7f0e", adabelief="#2ca02c", muon="#d62728", strider="#9467bd",
             strider_o="#8c564b")
LONG = dict(epanto="Pantograph", efin="Cooling fin", eheat="Graded bar", ecable="Cable", eplate="Cold plate")


def fig_seeds_by_arch(R, out):
    """Manuscript layout: one figure per architecture, one panel per problem, one row per optimizer."""
    from matplotlib.lines import Line2D
    for ar, title in (("pinn", "PINN"), ("pikan", "PIKAN-Chebyshev")):
        fig, axs = plt.subplots(3, 2, figsize=(7.2, 8.6))
        for A, p in zip(axs.ravel(), C.INTEGRAL + C.PDE):
            for k, o in enumerate(OPT6):
                l2 = [S.load_l2(f"{R}/runs/{p}_{ar}_{o}_s{s}.json") for s in C.SEEDS]
                l2 = [v for v in l2 if v is not None]
                if len(l2) < 2:
                    continue
                y, c = np.log10(l2), S.cell(l2)
                A.scatter(y, k + np.linspace(-0.12, 0.12, len(y)), s=14 if o == "strider_o" else 12,
                          marker="D" if o == "strider_o" else "o", color=TAB10[o], alpha=0.85, zorder=3, linewidths=0)
                A.errorbar(c["mean"], k, xerr=[[c["mean"] - np.log10(c["lo"])], [np.log10(c["hi"]) - c["mean"]]],
                           fmt="|", color="k", ms=12, mew=1.6, lw=0.9, capsize=3, zorder=4)
            A.set_yticks(range(len(OPT6)))
            A.set_yticklabels([LAB[o] for o in OPT6], fontsize=7.5)
            A.set_ylim(len(OPT6) - 0.5, -0.5)
            A.set_title(LONG[p], loc="left", fontweight="bold", fontsize=9)
            A.set_xlabel(r"$\log_{10}$ relative $L^2$ error  $\longleftarrow$ better", fontsize=7.5)
            A.grid(axis="x", alpha=0.3)
            A.tick_params(axis="x", labelsize=7)
        L = axs.ravel()[-1]
        L.axis("off")
        L.legend(handles=[Line2D([], [], marker="o", ls="", color="gray", label="Individual seed (n=5)"),
                          Line2D([], [], marker="|", color="k", ms=12, mew=1.6, lw=0.9, label="Mean and 95% t-CI"),
                          Line2D([], [], marker="D", ls="", color=TAB10["strider_o"], label="G-PULSE-O seed")],
                 loc="center", frameon=False, fontsize=8)
        fig.suptitle(f"Optimizer comparison: {title}", fontweight="bold", fontsize=10.5)
        save(fig, out, f"fig_seeds_{ar}")


def fig_ranks(R, out, sets="abc"):
    """Manuscript layout of the Friedman/Nemenyi summary: average ranks as labelled dots, one panel per set."""
    f = f"{R}/tables/summary.json"
    if not os.path.exists(f):
        print("fig_ranks: run analyze.py first")
        return
    FL = {k: v for k, v in json.load(open(f))["friedman"].items() if k[1] in sets}
    short = {"a": "(a) Five original optimizers", "b": "(b) G-PULSE-O in place of G-PULSE",
             "c": "(c) All six optimizers", "d": "(d) All six optimizers", "e": "(e) Post-hoc hybrid G-PULSE*"}
    fig, axs = plt.subplots(len(FL), 1, figsize=(7.2, 2.0 * len(FL)), squeeze=False)
    for A, (title, F) in zip(axs.ravel(), FL.items()):
        n_cfg = "6 PDE configurations" if F["N"] == 6 else f"{F['N']} configurations"
        A.set_title(f"{short[title[1]]}, {n_cfg}  |  Friedman p={F['p']:.3g}; CD={F['cd']:.2f}", loc="left", fontsize=7.5)
        for i, (o, r) in enumerate(zip(F["opts"], F["avg"])):
            col = "#1a7a3a" if o.startswith("strider") else "#444444"
            A.plot(r, i, "o", color=col, ms=3.5)
            A.text(r + 0.06, i, f"{LAB[o]} ({r:.2f})", va="center", fontsize=6.5, color=col)
        A.set_xlim(0.8, 7.0)
        A.set_ylim(len(F["opts"]) - 0.4, -0.6)
        A.set_yticks([])
        A.set_xlabel("Average rank (smaller is better)", fontsize=7)
        A.grid(axis="x", alpha=0.3)
        A.tick_params(axis="x", labelsize=7)
    save(fig, out, "fig_ranks")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(_setup.ROOT, "results/paper"))
    ap.add_argument("--cd-sets", default="abcde", help="Friedman sets shown in fig_cd (letters of analyze.py)")
    a = ap.parse_args()
    out = f"{a.results}/figures"
    os.makedirs(out, exist_ok=True)
    fig_seeds(a.results, out)
    fig_cd(a.results, out, a.cd_sets)
    fig_seeds_by_arch(a.results, out)
    fig_ranks(a.results, out, a.cd_sets)
    fig_convergence(a.results, out)
    fig_solutions(a.results, out)


if __name__ == "__main__":
    main()
