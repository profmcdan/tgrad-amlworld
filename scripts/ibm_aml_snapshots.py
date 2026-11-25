"""Build the train / validation / test snapshots for an IBM AML set, evaluate the two
training-free monitors and the closed-walk coverage of illicit accounts.

    python scripts/ibm_aml_snapshots.py --name HI-Small

Writes data/ibm_aml_proc/<NAME>_snapshots.pkl (cache for ibm_aml_models.py) and
results/ibm_aml/<NAME>/monitors.json.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ibm_aml_common as C  # noqa: E402

TAUS = (6.0, 24.0, 72.0, 168.0)          # hours; chosen on the training snapshot


def coverage(snap):
    """Share of window accounts lying on a closed walk of length <= 6 in the observed
    (binary, self-loops removed) graph."""
    d = snap["diag_bin"][snap["nodes"]]
    y, pm = snap["y"], snap["pmask"]
    any26, any36 = (d > 0).any(1), (d[:, 1:] > 0).any(1)
    any212 = (snap["diag_bin12"][snap["nodes"]] > 0).any(1)
    scc = snap["in_scc"][snap["nodes"]]
    out = {}
    groups = {"illicit": y == 1, "licit": y == 0, "CYCLE": pm[:, 0] & (y == 1),
              "illicit_non_cycle_pattern": pm[:, 1:].any(1) & ~pm[:, 0] & (y == 1),
              "illicit_unattributed": (y == 1) & ~pm.any(1)}
    for g, m in groups.items():
        out[g] = {"n": int(m.sum()),
                  "share_closed_walk_L2to6": float(any26[m].mean()) if m.sum() else None,
                  "share_closed_walk_L3to6": float(any36[m].mean()) if m.sum() else None,
                  "share_closed_walk_L2to12 (diagnostic)": float(any212[m].mean()) if m.sum() else None,
                  "share_in_nontrivial_SCC (any cycle length)": float(scc[m].mean()) if m.sum() else None,
                  **{f"share_L{L}": float((d[m, L - 2] > 0).mean()) if m.sum() else None for L in range(2, 7)}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="HI-Small")
    ap.add_argument("--taus", default="", help="comma-separated hours; default: the TAUS grid")
    a = ap.parse_args()
    taus = tuple(float(x) for x in a.taus.split(",")) if a.taus else TAUS
    z = C.load(a.name)
    T1, T2 = C.cuts(z)
    Tend = int(z["t"][-1])
    t0 = time.time()
    snaps = {"train": C.snapshot(z, 0, T1, taus), "val": C.snapshot(z, T1, T2, taus),
             "test": C.snapshot(z, T2, Tend, taus, end=True)}
    build_s = time.time() - t0
    res = {"dataset": a.name, "cuts_minutes": [T1, T2, Tend], "t0": str(z["t0"]),
           "snapshot_build_sec": build_s, "taus_h": list(taus), "splits": {}}
    for k, s in snaps.items():
        res["splits"][k] = {"n_accounts": int(len(s["nodes"])), "n_illicit": int(s["y"].sum()),
                            "illicit_rate": float(s["y"].mean()), "n_transactions_observed": s["n_obs"],
                            "n_directed_pairs": s["A_nnz"],
                            "n_per_pattern": {p: int((s["pmask"][:, i] & (s["y"] == 1)).sum())
                                              for i, p in enumerate(C.PATTERNS)},
                            "n_unattributed": int(((s["y"] == 1) & ~s["pmask"].any(1)).sum()),
                            "coverage": coverage(s)}

    # tau for the decayed monitor / T-GRAD block: best training ROC-AUC (L = 2..6)
    tr = snaps["train"]
    tau_auc = {tau: float(roc_auc_score(tr["y"], C.monitor_from_diag(tr["diag_tau"][tau][tr["nodes"]])))
               for tau in taus}
    tau = max(tau_auc, key=tau_auc.get)
    res["tau_train_auc"] = {str(k): v for k, v in tau_auc.items()}
    res["tau_selected_h"] = tau

    # streaming per-transaction score (simulator StreamingCycleMonitor, same tau),
    # read at each snapshot time; part of the T-GRAD closed-walk block
    from types import SimpleNamespace as NS
    from tgrad.simulation.monitor import StreamingCycleMonitor
    t_stream = time.time()
    sm = StreamingCycleMonitor(len(z["accounts"]), tau_h=tau)
    th = z["t"] / 60.0
    order = [("train", T1, False), ("val", T2, False), ("test", Tend, True)]
    k = 0
    for i in range(len(th)):
        while k < len(order) and not order[k][2] and z["t"][i] >= order[k][1]:
            snaps[order[k][0]]["stream"] = sm.node_scores_at(order[k][1] / 60.0); k += 1
        sm.on_commit(NS(id=i, src=int(z["src"][i]), dst=int(z["dst"][i]), finish=float(th[i])))
    snaps["test"]["stream"] = sm.node_scores_at(Tend / 60.0)
    res["stream_monitor_sec"] = time.time() - t_stream
    res["stream_monitor_cost_us_per_tx"] = 1e6 * float(np.mean(sm.cost_s))

    scores = {
        "Monitor (binary, L=2..6)": lambda s: C.monitor_from_diag(s["diag_bin"][s["nodes"]], 2),
        "Monitor (binary, L=3..6)": lambda s: C.monitor_from_diag(s["diag_bin"][s["nodes"]], 3),
        "Monitor (decayed, L=2..6)": lambda s: C.monitor_from_diag(s["diag_tau"][tau][s["nodes"]], 2),
        "Monitor (decayed, L=3..6)": lambda s: C.monitor_from_diag(s["diag_tau"][tau][s["nodes"]], 3),
        "Streaming monitor score": lambda s: s["stream"][s["nodes"]],
        "Monitor (binary, L=2..12) [post-hoc diagnostic]": lambda s: C.monitor_from_diag(s["diag_bin12"][s["nodes"]], 2),
        "Total degree (reference)": lambda s: s["deg"][s["nodes"]].sum(1),
    }
    res["models"] = {}
    for nm, f in scores.items():
        st = f(tr)
        thr1, thr2 = C.best_f1_threshold(tr["y"], st), C.fpr_threshold(tr["y"], st)
        res["models"][nm] = {k: C.evaluate(s["y"], f(s), s["pmask"], thr1, thr2) for k, s in snaps.items()}
        print(f"{nm:<28} " + "  ".join(f"{k}: AUC={v['ROC-AUC']:.3f} PR={v['PR-AUC']:.4f} "
                                       f"CYC-AUC={v['by_pattern']['CYCLE'].get('AUC_vs_licit', float('nan')):.3f}"
                                       for k, v in res["models"][nm].items()), flush=True)

    out = os.path.join(C.ROOT, "results", a.name)
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "monitors.json"), "w"), indent=2)
    snaps["tau"] = tau
    pickle.dump(snaps, open(os.path.join(C.ROOT, "data", "ibm_aml_proc", f"{a.name}_snapshots.pkl"), "wb"),
                protocol=4)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "coverage"} for k, v in res["splits"].items()},
                     indent=1))
    print(json.dumps({k: v["coverage"] for k, v in res["splits"].items()}, indent=1))
    print("tau", tau_auc, "->", tau, "build", round(build_s, 1), "s")


if __name__ == "__main__":
    main()
