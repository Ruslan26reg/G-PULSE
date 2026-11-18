"""All tables of the paper's experimental section from a results directory (default: the paper's logs).

    python scripts/analyze.py                      # results/paper/ -> results/paper/tables/
    python scripts/analyze.py --results results    # your own runs (tune_lr.py + run_seeds.py)

Writes tables.md (human-readable) and summary.json.  Accuracy and QoI tables need the weights (.npz) of the G-PULSE and
G-PULSE-O runs; the statistics need only the .json logs.
"""
import argparse
import json
import os

import numpy as np

import _setup  # noqa: F401
from gpulse import config as C
from gpulse import stats as S

NAMES = dict(epanto="Pantograph", efin="Fin", eheat="Bar", ecable="Cable", eplate="Plate")
ARCH = dict(pinn="PINN", pikan="PIKAN")
# display names: G-PULSE (code identifier 'strider') and G-PULSE-O ('strider_o')
ON = dict(adamw="AdamW", yogi="Yogi", adabelief="AdaBelief", muon="Muon", strider="G-PULSE", strider_o="G-PULSE-O",
          strider_star="G-PULSE*")
COMP = ["adamw", "yogi", "adabelief", "muon"]
OPT6 = COMP + ["strider", "strider_o"]
CFG10 = [(p, a) for p in C.INTEGRAL + C.PDE for a in C.ARCHS]
CFG6 = [(p, a) for p in C.PDE for a in C.ARCHS]
QOI_NAMES = dict(peak="peak abs(y)", max="max theta", mean="mean theta", sensor="sensor theta(1/2,T)",
                 flux0="heat flux q(0,T)", midspan="midspan w(1/2,T)", maxabs="max abs(w(.,T))", hotspot="hot spot max abs(u)")


def e(v):
    return f"{v:.1e}"


def ms(v):
    v = np.asarray(v, float)
    return f"{e(v.mean())} ± {e(v.std(ddof=1))}" if len(v) > 1 else e(v[0])


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(_setup.ROOT, "results/paper"))
    a = ap.parse_args()
    R = a.results
    run = lambda p, ar, o, s: f"{R}/runs/{p}_{ar}_{o}_s{s}.json"        # noqa: E731
    sel = json.load(open(f"{R}/selected_lr.json"))

    # ------------------------------------------------------------------ load L2 of all runs
    D = {}
    for o in OPT6:
        for (p, ar) in C.configs(o):
            runs = {s: S.load_l2(run(p, ar, o, s)) for s in C.SEEDS}
            runs = {s: v for s, v in runs.items() if v is not None}
            if len(runs) >= 2:
                D[(p, ar, o)] = runs
    for (p, ar) in CFG10:                                               # G-PULSE by problem class
        src = "strider_o" if p in C.PDE else "strider"
        if (p, ar, src) in D:
            D[(p, ar, "strider_star")] = D[(p, ar, src)]
    cells = {k: S.cell(list(v.values())) for k, v in D.items()}
    md, summary = [f"# Tables generated from `{os.path.relpath(R, _setup.ROOT)}`", ""], {}

    # ------------------------------------------------------------------ accuracy and QoI (need weights)
    from gpulse.metrics import qoi_errors, solution_errors
    from gpulse.varpro import load_model

    def accuracy(opt, cfgs, title):
        rows, js = [], {}
        for (p, ar) in cfgs:
            m = []
            for s in C.SEEDS:
                f = run(p, ar, opt, s)
                if os.path.exists(f.replace(".json", ".npz")):
                    prob, model, r = load_model(f)
                    m.append(dict(solution_errors(prob, model), lr=r["lr"], qoi=qoi_errors(prob, model)))
            if not m:
                continue
            js[f"{p}/{ar}"] = m
            rows.append([NAMES[p], ARCH[ar], f"{m[0]['lr']:g}"] + [ms([x[k] for x in m]) for k in ("mse", "l2", "h1")])
        if rows:
            md.extend([f"## {title}", "", table(["Problem", "Net", "lr", "MSE", "rel. L2", "rel. H1"], rows), ""])
        return js

    acc = accuracy("strider", CFG10, "G-PULSE: errors, mean ± sd over seeds (bar and cable at t = T)")
    summary["accuracy_strider"] = acc
    if acc:
        rows = []
        for p in C.INTEGRAL + C.PDE:
            for k in (acc.get(f"{p}/pinn") or [{"qoi": {}}])[0]["qoi"]:
                rows.append([NAMES[p], QOI_NAMES[k]] + [ms([x["qoi"][k] for x in acc[f"{p}/{ar}"]]) if f"{p}/{ar}" in acc
                                                        else "—" for ar in C.ARCHS])
        md.extend(["## G-PULSE: relative errors of the quantities of interest, mean ± sd", "",
                   table(["Problem", "Quantity", "PINN", "PIKAN"], rows), ""])
    summary["accuracy_strider_o"] = accuracy("strider_o", CFG10, "G-PULSE-O: errors, mean ± sd over seeds")

    # ------------------------------------------------------------------ selected learning rates
    def lr_cell(p, ar, o):
        k = f"{p}/{ar}/{o}"
        if k not in sel:
            return "—"
        ent = sel[k]
        lrs = sorted(float(x) for x in ent["val"])
        mark = "*" if len(lrs) > 4 and ent["lr"] in (lrs[0], lrs[-1]) else ""
        mark += "†" if "lr_final" in ent and ent["lr_final"] != ent["lr"] else ""
        return f"{ent.get('lr_final', ent['lr']):g}{mark}"
    md.extend(["## Selected learning rates (* optimum on the boundary of the extended grid; † instability fallback)", "",
               table(["Problem", "Net"] + [ON[o] for o in OPT6],
                     [[NAMES[p], ARCH[ar]] + [lr_cell(p, ar, o) for o in OPT6] for (p, ar) in CFG10]), ""])

    # ------------------------------------------------------------------ geometric means
    rows = []
    for (p, ar) in CFG10:
        avail = [o for o in OPT6 if (p, ar, o) in cells]
        best = min(avail, key=lambda o: cells[(p, ar, o)]["mean"])
        row = [NAMES[p], ARCH[ar]]
        for o in OPT6:
            if o not in avail:
                row.append("—")
                continue
            c = cells[(p, ar, o)]
            txt = f"{e(c['gm'])} [{e(c['lo'])}, {e(c['hi'])}]" + (f"; {c['n_failed']} failed" if c["n_failed"] else "")
            row.append(f"**{txt}**" if o == best else txt)
        rows.append(row)
    md.extend(["## Relative L2 error: geometric mean over seeds [95% CI] (bold: best; failed: L2 > 0.1)", "",
               table(["Problem", "Net"] + [ON[o] for o in OPT6], rows), ""])
    summary["cells"] = {"/".join(k): v for k, v in cells.items()}

    # ------------------------------------------------------------------ paired tests
    def paired_table(a_opt, others, cfgs, title):
        keys, res = [], []
        for (p, ar) in cfgs:
            for o in others:
                if (p, ar, a_opt) in D and (p, ar, o) in D:
                    keys.append((p, ar, o))
                    res.append(S.paired(D[(p, ar, a_opt)], D[(p, ar, o)]))
        for r, q in zip(res, S.holm([r["p"] for r in res])):
            r["p_holm"] = q
        T = dict(zip(keys, res))
        rows = []
        for (p, ar) in cfgs:
            row = [NAMES[p], ARCH[ar]]
            for o in others:
                r = T.get((p, ar, o))
                if r is None:
                    row.append("—")
                    continue
                txt = f"{r['ratio']:.2g} [{r['lo']:.2g}, {r['hi']:.2g}]; pH={r['p_holm']:.3f}; {r['wins']}/{r['n']}"
                row.append(f"**{txt}**" if (r["p_holm"] < 0.05 and r["ratio"] < 1) else
                           f"_{txt}_" if r["p_holm"] < 0.05 else txt)
            rows.append(row)
        n_for = sum(r["p_holm"] < 0.05 and r["ratio"] < 1 for r in res)
        n_ag = sum(r["p_holm"] < 0.05 and r["ratio"] > 1 for r in res)
        md.extend([f"## {title}", "", f"Holm correction over {len(res)} comparisons; significant in favour: {n_for}, "
                   f"against: {n_ag}. Cell: ratio of geometric means [95% CI]; Holm p; seeds won.", "",
                   table(["Problem", "Net"] + [ON[o] for o in others], rows), ""])
        return {"/".join(k): v for k, v in T.items()}
    summary["paired_strider"] = paired_table("strider", COMP, CFG10,
                                             "G-PULSE vs each competitor (ratio < 1: G-PULSE better)")
    summary["paired_strider_o"] = paired_table("strider_o", COMP + ["strider"], CFG10,
                                               "G-PULSE-O vs each competitor and G-PULSE (ratio < 1: G-PULSE-O better)")

    # ------------------------------------------------------------------ Friedman / Nemenyi
    sets = [("(a) AdamW, Yogi, AdaBelief, Muon, G-PULSE; 10 configurations", CFG10, COMP + ["strider"]),
            ("(b) AdamW, Yogi, AdaBelief, Muon, G-PULSE-O; 10 configurations", CFG10, COMP + ["strider_o"]),
            ("(c) all six optimizers; 10 configurations", CFG10, OPT6),
            ("(d) all six optimizers; 6 PDE configurations", CFG6, OPT6),
            ("(e) post-hoc hybrid G-PULSE* (G-PULSE on integral eqs., G-PULSE-O on PDEs); 10 configurations", CFG10,
             COMP + ["strider_star"])]
    rows, summary["friedman"] = [], {}
    for title, cfgs, opts in sets:
        if not all((p, ar, o) in cells for (p, ar) in cfgs for o in opts):
            continue
        F = S.friedman([[cells[(p, ar, o)]["mean"] for o in opts] for (p, ar) in cfgs])
        F["opts"] = opts
        summary["friedman"][title] = F
        ranks = ", ".join(f"{ON[o]} {r:.2f}" for o, r in sorted(zip(opts, F["avg"]), key=lambda x: x[1]))
        rows.append([title, f"{F['chi2']:.2f}", f"{F['p']:.2g}", f"{F['cd']:.2f}", ranks])
    md.extend(["## Friedman test and Nemenyi critical difference (average ranks, 1 = best)", "",
               table(["Set", "chi2", "p", "CD", "average ranks"], rows), ""])

    # ------------------------------------------------------------------ cost per iteration
    # controlled benchmark (time_benchmark.py): all optimizers under identical conditions; the times in the run logs
    # depend on the machine load of each session and are reported only as total CPU time
    summary["time"] = {}
    tf = f"{R}/timing.json"
    if os.path.exists(tf):
        T = json.load(open(tf))["ms_per_iteration"]
        rows = []
        for ar in C.ARCHS:
            for cls, probs in (("integral", C.INTEGRAL), ("PDE", C.PDE)):
                row = [f"{ARCH[ar]}, {cls}"]
                for o in OPT6:
                    reps = np.array([T[f"{p}/{ar}/{o}"] for p in probs if f"{p}/{ar}/{o}" in T])   # [problems, repeats]
                    if reps.size == 0:
                        row.append("—")
                        continue
                    per_rep = reps.mean(0)                                  # class mean in each repeat
                    sd = float(per_rep.std(ddof=1)) if len(per_rep) > 1 else 0.0
                    summary["time"][f"{ar}/{cls}/{o}"] = [float(per_rep.mean()), sd]
                    row.append(f"{per_rep.mean():.0f} ± {sd:.0f}")
                rows.append(row)
        md.extend(["## Cost per training iteration, ms (controlled benchmark; mean ± sd over repeats)", "",
                   table([""] + [ON[o] for o in OPT6], rows), ""])
    cpu = 0.0
    for o in OPT6:
        for (p, ar) in C.configs(o):
            for s_ in C.SEEDS:
                f = run(p, ar, o, s_)
                if os.path.exists(f):
                    with open(f) as fh:
                        cpu += json.load(fh).get("train_s", 0.0)
    summary["total_cpu_hours_runs"] = cpu / 3600
    md.extend([f"Total training time of the logged runs: {cpu / 3600:.0f} CPU-hours.", ""])

    os.makedirs(f"{R}/tables", exist_ok=True)
    with open(f"{R}/tables/tables.md", "w") as f:
        f.write("\n".join(md))
    with open(f"{R}/tables/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("\n".join(md))
    print(f"\nsaved {R}/tables/tables.md and summary.json")


if __name__ == "__main__":
    main()
