"""Improvements 1 and 2 to the training-free monitor, with validation-only selection.

    python scripts/ibm_aml_improve_monitors.py --name HI-Small

1. Closed-walk monitor with a longer cap: M_i = sum_{L=2..cap} 2^-L (A^L)_ii on the
   cumulative binary graph, cap in {6, 8, 10, 12}.
2. Time-respecting cycle monitor (scripts/ibm_aml_temporal_cycles.py), window W in
   {6, 24, 72, 168} h and cap in {6, 8, 10, 12}.

For each family the configuration is chosen by validation-window ROC-AUC of CYCLE
accounts against licit accounts (ties: smaller W, then smaller cap); thresholds come from
the training snapshot. Every configuration is evaluated on every split and written out,
but only the validation numbers enter the choice. Writes
results/ibm_aml/<NAME>/improved_monitors.json.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ibm_aml_common as C  # noqa: E402
from ibm_aml_temporal_cycles import CAPS, WINDOWS  # noqa: E402


def ev(S, f):
    tr = S["train"]; st = f(tr)
    thr1, thr2 = C.best_f1_threshold(tr["y"], st), C.fpr_threshold(tr["y"], st)
    return {k: C.evaluate(s["y"], f(s), s["pmask"], thr1, thr2) for k, s in S.items()}


def select(res):
    key = lambda kv: (-kv[1]["val"]["by_pattern"]["CYCLE"]["AUC_vs_licit"], kv[1]["_order"])
    return min(res.items(), key=key)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="HI-Small")
    a = ap.parse_args()
    snaps = pickle.load(open(os.path.join(C.ROOT, "data", "ibm_aml_proc", f"{a.name}_snapshots.pkl"), "rb"))
    S = {k: snaps[k] for k in ("train", "val", "test")}
    out = {"dataset": a.name, "selection": "validation ROC-AUC, CYCLE accounts vs licit; ties -> smaller W, cap"}

    walk = {}
    for i, cap in enumerate(CAPS):
        f = lambda s, cap=cap: C.monitor_from_diag(s["diag_bin12"][s["nodes"]][:, : cap - 1], 2)
        walk[f"Closed-walk monitor, binary, L=2..{cap}"] = {**ev(S, f), "_order": i}
    out["closed_walk"] = walk
    out["closed_walk_selected"] = select(walk)

    temporal, stats = {}, {}
    for i, W in enumerate(WINDOWS):
        p = os.path.join(C.ROOT, "data", "ibm_aml_proc", f"{a.name}_tcycle_W{W:g}.npz")
        z = np.load(p)
        stats[f"W={W:g}h"] = json.load(open(os.path.join(C.ROOT, "results", a.name,
                                                         f"tcycle_W{W:g}_stats.json")))
        for j, cap in enumerate(CAPS):
            sc ={sp: z[f"{sp}_L{cap}"].astype(float) for sp in S}

            def f(s, sc=sc):
                sp = next(k for k in S if S[k] is s)
                return np.log1p(sc[sp][s["nodes"]])
            temporal[f"Temporal-cycle monitor, W={W:g}h, L<={cap}"] = {**ev(S, f), "_order": 10 * i + j}
    out["temporal"] = temporal
    out["temporal_selected"] = select(temporal)
    out["temporal_stream_stats"] = stats

    # coverage: share on a time-respecting cycle (score > 0) at the selected config
    W_sel = out["temporal_selected"].split("W=")[1].split("h")[0]
    cap_sel = int(out["temporal_selected"].split("L<=")[1])
    z = np.load(os.path.join(C.ROOT, "data", "ibm_aml_proc", f"{a.name}_tcycle_W{W_sel}.npz"))
    cov = {}
    for sp, s in S.items():
        on = z[f"{sp}_L{cap_sel}"][s["nodes"]] > 0
        y, pm = s["y"], s["pmask"]
        groups = {"illicit": y == 1, "licit": y == 0, "CYCLE": pm[:, 0] & (y == 1),
                  "illicit_non_cycle_pattern": pm[:, 1:].any(1) & ~pm[:, 0] & (y == 1),
                  "illicit_unattributed": (y == 1) & ~pm.any(1)}
        cov[sp] = {g: {"n": int(m.sum()), "share_on_temporal_cycle": float(on[m].mean()) if m.sum() else None}
                   for g, m in groups.items()}
    out["temporal_selected_coverage"] = cov

    json.dump(out, open(os.path.join(C.ROOT, "results", a.name, "improved_monitors.json"), "w"), indent=1)
    for fam in ("closed_walk", "temporal"):
        print(f"\n== {fam}: selected on validation -> {out[fam + '_selected']}")
        for nm, r in out[fam].items():
            v, te = r["val"], r["test"]
            mark = "*" if nm == out[fam + "_selected"] else " "
            print(f"{mark} {nm:<44} val CYC-AUC {v['by_pattern']['CYCLE']['AUC_vs_licit']:.3f} | test AUC "
                  f"{te['ROC-AUC']:.3f} PR {te['PR-AUC']:.4f} R@1% {te['Recall@1%FPR']:.3f} "
                  f"CYC R@1% {te['by_pattern']['CYCLE']['Recall@1%FPR']:.2f} CYC-AUC {te['by_pattern']['CYCLE']['AUC_vs_licit']:.3f}")
    print(json.dumps(cov, indent=1))


if __name__ == "__main__":
    main()
