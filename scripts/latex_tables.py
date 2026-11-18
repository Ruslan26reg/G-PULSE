"""LaTeX tables of the manuscript (booktabs/makecell) from a results directory; run analyze.py first.

    python scripts/analyze.py && python scripts/latex_tables.py      # -> results/paper/tables/tab_delay_*.tex
"""
import argparse
import json
import os

import numpy as np

import _setup  # noqa: F401
from gpulse import config as C

NAME = dict(adamw="AdamW", yogi="Yogi", adabelief="AdaBelief", muon="Muon", strider="G-PULSE", strider_o="G-PULSE-O")
PROB = dict(epanto="Pantograph", efin="Fin", eheat="Bar", ecable="Cable", eplate="Plate")
NET = dict(pinn="PINN", pikan="PIKAN")
COMP = ["adamw", "yogi", "adabelief", "muon"]
OPT6 = COMP + ["strider", "strider_o"]
PROBS = list(C.INTEGRAL + C.PDE)
QN = dict(epanto=[("peak", r"peak $\max|y|$")], efin=[("max", r"$\max\theta$"), ("mean", r"mean $\theta$")],
          eheat=[("sensor", r"sensor $\theta(\tfrac12,T)$"), ("flux0", r"heat flux $q(0,T)$")],
          ecable=[("midspan", r"midspan $w(\tfrac12,T)$"), ("maxabs", r"$\max|w(\cdot,T)|$")],
          eplate=[("hotspot", r"hot spot $\max|u|$")])


def e(v, d=1):
    m, x = f"{v:.{d}e}".split("e")
    return f"{m}{{\\times}}10^{{{int(x)}}}"


def ms(vals):
    v = np.array(vals, float)
    return f"${e(v.mean())}\\pm{e(v.std(ddof=1))}$"


def g(v):
    return f"{v:.2g}".replace("e-0", "e-").replace("e+0", "e")


def tabular(cols, header, body):
    return "\n".join([r"\begin{tabular}{" + cols + "}", r"\toprule", header + r"\\", r"\midrule"] + body
                     + [r"\bottomrule", r"\end{tabular}"])


def by_problem(rows_of):
    """Rows grouped by problem with \\addlinespace between problems."""
    body = []
    for i, p in enumerate(PROBS):
        rows = rows_of(p)
        if rows:
            body += rows
            if i < len(PROBS) - 1:
                body.append(r"\addlinespace")
    while body and body[-1] == r"\addlinespace":
        body.pop()
    return body


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(_setup.ROOT, "results/paper"))
    a = ap.parse_args()
    R = a.results
    S = json.load(open(f"{R}/tables/summary.json"))
    sel = json.load(open(f"{R}/selected_lr.json"))
    out = f"{R}/tables"
    files = {}

    # accuracy (G-PULSE, G-PULSE-O)
    for opt, key in (("strider", "accuracy_strider"), ("strider_o", "accuracy_strider_o")):
        acc = S[key]

        def rows(p, acc=acc):
            r = []
            for ar in C.ARCHS:
                m = acc.get(f"{p}/{ar}")
                if m:
                    r.append(f"{PROB[p] if not r else ''} & {NET[ar]} & ${m[0]['lr']:g}$ & "
                             + " & ".join(ms([x[k] for x in m]) for k in ("mse", "l2", "h1")) + r"\\")
            return r
        files[f"tab_delay_{opt}"] = tabular("llcccc", r"Problem & Network & lr & MSE & rel. $L^2$ & rel. $H^1$",
                                            by_problem(rows))

    # quantities of interest (G-PULSE)
    acc = S["accuracy_strider"]
    body = []
    for p in PROBS:
        for i, (k, lab) in enumerate(QN[p]):
            body.append(f"{PROB[p] if i == 0 else ''} & {lab} & "
                        + " & ".join(ms([x["qoi"][k] for x in acc[f"{p}/{ar}"]]) for ar in C.ARCHS) + r"\\")
    files["tab_delay_qoi"] = tabular("llcc", r"Problem & Quantity of interest & PINN & PIKAN", body)

    # selected learning rates
    def lr_cell(p, ar, o):
        k = f"{p}/{ar}/{o}"
        if k not in sel:
            return "---"
        ent = sel[k]
        lrs = sorted(float(x) for x in ent["val"])
        mark = "$^{*}$" if len(lrs) > 4 and ent["lr"] in (lrs[0], lrs[-1]) else ""
        mark += "$^{\\dagger}$" if "lr_final" in ent and ent["lr_final"] != ent["lr"] else ""
        return f"${ent.get('lr_final', ent['lr']):g}${mark}"
    body = [f"{PROB[p] if ar == 'pinn' else ''} & {NET[ar]} & " + " & ".join(lr_cell(p, ar, o) for o in OPT6) + r"\\"
            for p in PROBS for ar in C.ARCHS]
    files["tab_delay_opt_lr"] = tabular("ll" + "c" * 6, "Problem & Net & " + " & ".join(NAME[o] for o in OPT6), body)

    # geometric means with CI
    cells = S["cells"]

    def stats_rows(p):
        r = []
        for ar in C.ARCHS:
            avail = [o for o in OPT6 if f"{p}/{ar}/{o}" in cells]
            best = min(avail, key=lambda o: cells[f"{p}/{ar}/{o}"]["mean"])
            cs = []
            for o in OPT6:
                c = cells.get(f"{p}/{ar}/{o}")
                if c is None:
                    cs.append("---")
                    continue
                top = f"$\\mathbf{{{e(c['gm'])}}}$" if o == best else f"${e(c['gm'])}$"
                low = f"{{\\scriptsize$[{e(c['lo'])},\\,{e(c['hi'])}]$" + (f"; {c['n_failed']} failed" if c["n_failed"] else "") + "}"
                cs.append("\\makecell{" + top + "\\\\" + low + "}")
            r.append(f"{PROB[p] if ar == 'pinn' else ''} & {NET[ar]} & " + " & ".join(cs) + r"\\")
        return r
    files["tab_delay_stats"] = tabular("ll" + "c" * 6, "Problem & Net & " + " & ".join(NAME[o] for o in OPT6),
                                       by_problem(stats_rows))

    # paired tests
    for key, others, name in (("paired_strider", COMP, "tab_delay_pairs"),
                              ("paired_strider_o", COMP + ["strider"], "tab_delay_pairs_o")):
        T = S[key]

        def pair_rows(p, T=T, others=others):
            r = []
            for ar in C.ARCHS:
                cs = []
                for o in others:
                    t = T.get(f"{p}/{ar}/{o}")
                    if t is None:
                        cs.append("---")
                        continue
                    top = f"{g(t['ratio'])} [{g(t['lo'])}, {g(t['hi'])}]"
                    if t["p_holm"] < 0.05:
                        top = (r"\textbf{" if t["ratio"] < 1 else r"\textit{") + top + "}"
                    ph = "<0.001" if t["p_holm"] < 0.001 else "=" + format(t["p_holm"], ".3f")
                    cs.append("\\makecell{" + top + "\\\\" + f"{{\\scriptsize $p_{{\\mathrm{{H}}}}${ph}, {t['wins']}/{t['n']}}}" + "}")
                if any(c != "---" for c in cs):
                    r.append(f"{PROB[p] if not r else ''} & {NET[ar]} & " + " & ".join(cs) + r"\\")
            return r
        files[name] = tabular("ll" + "c" * len(others), "Problem & Net & " + " & ".join(NAME[o] for o in others),
                              by_problem(pair_rows))

    # the same tables split into column groups, as typeset in the manuscript (problem name on every row)
    def gm_cell(p, ar, o):
        avail = [x for x in OPT6 if f"{p}/{ar}/{x}" in cells]
        best = min(avail, key=lambda x: cells[f"{p}/{ar}/{x}"]["mean"])            # best among all six methods
        c = cells.get(f"{p}/{ar}/{o}")
        if c is None:
            return "---"
        top = f"$\\mathbf{{{e(c['gm'])}}}$" if o == best else f"${e(c['gm'])}$"
        low = f"$[{e(c['lo'])},\\,{e(c['hi'])}]$" + (f"; {c['n_failed']} failed" if c["n_failed"] else "")
        return "\\makecell{" + top + "\\\\" + low + "}"

    def pair_cell(T, p, ar, o):
        t = T.get(f"{p}/{ar}/{o}")
        if t is None:
            return "---"
        top = f"{g(t['ratio'])} [{g(t['lo'])}, {g(t['hi'])}]"
        if t["p_holm"] < 0.05:
            top = (r"\textbf{" if t["ratio"] < 1 else r"\textit{") + top + "}"
        ph = "<0.001" if t["p_holm"] < 0.001 else "=" + format(t["p_holm"], ".3f")
        return "\\makecell{" + top + "\\\\" + f"{{\\footnotesize $p_{{\\mathrm{{H}}}}${ph}, {t['wins']}/{t['n']}}}" + "}"

    def split(name, opts, cell):
        body = [f"{PROB[p]} & {NET[ar]} & " + " & ".join(cell(p, ar, o) for o in opts) + r"\\"
                for p in PROBS for ar in C.ARCHS]
        files[name] = tabular("ll" + "c" * len(opts), "Problem & Net & " + " & ".join(NAME[o] for o in opts), body)

    split("tab_ms_gm_adamw_yogi_adabelief", ["adamw", "yogi", "adabelief"], gm_cell)
    split("tab_ms_gm_muon_gpulse_gpulseo", ["muon", "strider", "strider_o"], gm_cell)
    split("tab_ms_pairs_gpulse_adamw_yogi", ["adamw", "yogi"], lambda p, ar, o: pair_cell(S["paired_strider"], p, ar, o))
    split("tab_ms_pairs_gpulse_adabelief_muon", ["adabelief", "muon"],
          lambda p, ar, o: pair_cell(S["paired_strider"], p, ar, o))
    split("tab_ms_pairs_gpulseo_adamw_yogi_adabelief", ["adamw", "yogi", "adabelief"],
          lambda p, ar, o: pair_cell(S["paired_strider_o"], p, ar, o))
    split("tab_ms_pairs_gpulseo_muon_gpulse", ["muon", "strider"],
          lambda p, ar, o: pair_cell(S["paired_strider_o"], p, ar, o))

    # cost per iteration (controlled benchmark)
    if S.get("time"):
        body = []
        for ar in C.ARCHS:
            for cls in ("integral", "PDE"):
                t = [S["time"].get(f"{ar}/{cls}/{o}") for o in OPT6]
                body.append(f"{NET[ar]}, {cls} & " + " & ".join(f"${x[0]:.0f}\\pm{x[1]:.0f}$" if x else "---" for x in t) + r"\\")
        files["tab_delay_time"] = tabular("l" + "c" * 6, " & " + " & ".join(NAME[o] for o in OPT6), body)

    # Friedman summary
    body = []
    for title, F in S["friedman"].items():
        lab = {o: NAME.get(o, "G-PULSE$^\\ast$") for o in F["opts"]}
        ranks = ", ".join(f"{lab[o]} {r:.2f}" for o, r in sorted(zip(F["opts"], F["avg"]), key=lambda x: x[1]))
        body.append(f"{title.split(')')[0]}) & {F['chi2']:.2f} & {F['p']:.2g} & {F['cd']:.2f} & {ranks}" + r"\\")
    files["tab_delay_friedman"] = tabular("lcccl", r"Set & $\chi^2$ & $p$ & CD & average ranks", body)

    for k, v in files.items():
        with open(f"{out}/{k}.tex", "w") as f:
            f.write(v)
        print("saved", f"{out}/{k}.tex")


if __name__ == "__main__":
    main()
