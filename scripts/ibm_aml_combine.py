"""Improvements 4-6 on HI-Small and LI-Small (development datasets; validation-only choices).

    python scripts/ibm_aml_combine.py models --name HI-Small --seed 0
    python scripts/ibm_aml_combine.py table  --name HI-Small

4. GraphSAGE with the rank-scaled T-GRAD block and the time-respecting cycle count together.
5. Graph Transformer with the T-GRAD features: ranked block, cycle count, and both
   (the matched-block Graph Transformer is re-run here as the reference).
6. Score fusion: rank average of a learned model's score with the training-free closed-walk
   monitor (binary, L = 2..12), fused = (1 - w) F(model) + w F(monitor), where F is the
   empirical CDF of that score on the training accounts (mid-rank for ties), applied to every
   split; w = 0 is then a monotone map of the model score with identical thresholds. w in {0, 0.05, 0.1, 0.2, 0.3, 0.5} is chosen per model and seed by
   validation ROC-AUC (w = 0 means no fusion). Thresholds come from the training snapshot.

Settings are those of scripts/ibm_aml_confirm.py (tau = 72 h, temporal cycles W = 168 h,
L <= 12). Pre-declared selection: among all learned rows (with and without fusion), the one
with the best mean validation ROC-AUC per backbone. Test numbers are reported for every row
but are not used for any choice.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ibm_aml_common as C  # noqa: E402
from ibm_aml_confirm import feature_sets, load  # noqa: E402
from ibm_aml_models import GNN  # noqa: E402

PLAN = [("GraphSAGE", "T-GRAD block, ranked"), ("GraphSAGE", "temporal cycles"),
        ("GraphSAGE", "ranked block + temporal cycles"),
        ("Graph Transformer", "matched block"), ("Graph Transformer", "T-GRAD block, ranked"),
        ("Graph Transformer", "temporal cycles"), ("Graph Transformer", "ranked block + temporal cycles")]
WEIGHTS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
SPLITS = ("train", "val", "test")


def feats(s, tau, tc, fs):
    F = feature_sets(s, tau, tc)
    if fs == "ranked block + temporal cycles":
        return np.hstack([F["T-GRAD block, ranked"], np.log1p(tc)[:, None]])
    return F[fs]


def cdf(ref):
    r = np.sort(np.asarray(ref, float))
    return lambda v: (np.searchsorted(r, v, "left") + np.searchsorted(r, v, "right")) / (2.0 * len(r))


def ev(S, sc):
    ytr = S["train"]["y"]
    thr1, thr2 = C.best_f1_threshold(ytr, sc["train"]), C.fpr_threshold(ytr, sc["train"])
    return {k: C.evaluate(S[k]["y"], sc[k], S[k]["pmask"], thr1, thr2) for k in SPLITS}


def models(a):
    torch.set_num_threads(a.threads)
    tau, S, tc = load(a.name)
    assert tau == 72.0, tau
    tr = S["train"]; ytr = tr["y"]
    pw = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
    base = os.path.join(C.ROOT, "results", a.name)
    out_path = os.path.join(base, f"combine_seed{a.seed}.json")
    sc_path = os.path.join(base, f"combine_scores_seed{a.seed}.npz")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    scores = dict(np.load(sc_path)) if os.path.exists(sc_path) else {}
    res["_meta"] = {"seed": a.seed, "epochs": a.epochs, "tau_h": tau, "W_h": 168, "cap": 12, "weights": WEIGHTS}
    mon = {k: C.monitor_from_diag(S[k]["diag_bin12"][S[k]["nodes"]], 2) for k in SPLITS}
    trn = torch.tensor(tr["nodes"]); yt = torch.tensor(ytr)
    EI = {k: torch.tensor(S[k]["ei"]) for k in SPLITS}
    for kind, fs in PLAN:
        name = f"{kind} + {fs}"
        if name not in res:
            F = {k: feats(S[k], tau, tc[k], fs) for k in SPLITS}
            mu = F["train"][tr["nodes"]].mean(0); sd = F["train"][tr["nodes"]].std(0) + 1e-8
            X = {k: torch.tensor((F[k] - mu) / sd, dtype=torch.float32) for k in SPLITS}
            del F
            torch.manual_seed(a.seed); t0 = time.time()
            net = GNN(kind, X["train"].shape[1]); o = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=5e-4)
            cw = torch.tensor([1.0, pw])
            for _ in range(a.epochs):
                net.train(); o.zero_grad()
                nn.functional.cross_entropy(net(X["train"], EI["train"])[trn], yt, weight=cw).backward(); o.step()
            net.eval()
            with torch.no_grad():
                sc = {k: torch.softmax(net(X[k], EI[k]), 1)[:, 1].numpy()[S[k]["nodes"]] for k in SPLITS}
            res[name] = ev(S, sc); res[name]["train_sec"] = time.time() - t0
            for k in SPLITS:
                scores[f"{name}|{k}"] = sc[k]
            np.savez_compressed(sc_path, **scores)
            te = res[name]["test"]
            print(f"[seed {a.seed}] {name:<52} val AUC={res[name]['val']['ROC-AUC']:.3f} | test AUC={te['ROC-AUC']:.3f} "
                  f"PR={te['PR-AUC']:.4f} F1={te['F1']:.3f} R@1%={te['Recall@1%FPR']:.3f} "
                  f"CYCLE R@1%={te['by_pattern']['CYCLE']['Recall@1%FPR']:.3f} ({res[name]['train_sec']:.0f}s)", flush=True)
            json.dump(res, open(out_path, "w"), indent=1)
        # fusion with the closed-walk monitor, w chosen on validation ROC-AUC
        fname = f"{name}, fused with monitor"
        if fname not in res:
            sc = {k: scores[f"{name}|{k}"] for k in SPLITS}
            Fm, Fo = cdf(sc["train"]), cdf(mon["train"])
            cand = {}
            for w in WEIGHTS:
                fz = {k: (1 - w) * Fm(sc[k]) + w * Fo(mon[k]) for k in SPLITS}
                cand[w] = (C.evaluate(S["val"]["y"], fz["val"], S["val"]["pmask"], 0.5, 0.5)["ROC-AUC"], fz)
            w_sel = max(WEIGHTS, key=lambda w: (cand[w][0], -w))
            res[fname] = ev(S, cand[w_sel][1])
            res[fname]["w_selected"] = w_sel
            res[fname]["val_auc_by_w"] = {str(w): cand[w][0] for w in WEIGHTS}
            te = res[fname]["test"]
            print(f"[seed {a.seed}] {fname:<52} w={w_sel} val AUC={res[fname]['val']['ROC-AUC']:.3f} | test AUC={te['ROC-AUC']:.3f} "
                  f"PR={te['PR-AUC']:.4f} F1={te['F1']:.3f} R@1%={te['Recall@1%FPR']:.3f} "
                  f"CYCLE R@1%={te['by_pattern']['CYCLE']['Recall@1%FPR']:.3f}", flush=True)
            json.dump(res, open(out_path, "w"), indent=1)


def table(a):
    base = os.path.join(C.ROOT, "results", a.name)
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(base, "combine_seed*.json")))]
    keys = [("val ROC-AUC", "val", lambda e: e["ROC-AUC"]), ("ROC-AUC", "test", lambda e: e["ROC-AUC"]),
            ("PR-AUC", "test", lambda e: e["PR-AUC"]), ("F1", "test", lambda e: e["F1"]),
            ("R@1%FPR", "test", lambda e: e["Recall@1%FPR"]),
            ("CYCLE R@1%FPR", "test", lambda e: e["by_pattern"]["CYCLE"]["Recall@1%FPR"])]
    names = [n for n in runs[0] if not n.startswith("_")]
    agg = {}
    for nm in names:
        rs = [r[nm] for r in runs if nm in r]
        agg[nm] = {"n_seeds": len(rs), **{k: {"mean": float(np.mean([f(r[sp]) for r in rs])),
                                                "std": float(np.std([f(r[sp]) for r in rs]))} for k, sp, f in keys}}
        if "w_selected" in rs[0]:
            agg[nm]["w_selected"] = [r["w_selected"] for r in rs]
    sel = {}
    for kind in ("GraphSAGE", "Graph Transformer"):
        c = [n for n in agg if n.startswith(kind + " +")]
        sel[kind] = max(c, key=lambda n: agg[n]["val ROC-AUC"]["mean"])
    L = [f"# IBM AML {a.name}: improvements 4-6 (generated by scripts/ibm_aml_combine.py table)\n",
         "tau = 72 h, temporal cycles W = 168 h, L <= 12. Mean ± std over seeds; test window unless marked. "
         "★ = best mean validation ROC-AUC per backbone (the pre-declared choice). Fusion weight w chosen per seed "
         "on validation ROC-AUC (w = 0 is no fusion).\n",
         "| Model | seeds | w (per seed) | " + " | ".join(k for k, _, _ in keys) + " |",
         "|---|---|---|" + "---|" * len(keys)]
    for nm, d in agg.items():
        star = " ★" if nm in sel.values() else ""
        w = ", ".join(f"{x:g}" for x in d["w_selected"]) if "w_selected" in d else "–"
        L.append(f"| {nm}{star} | {d['n_seeds']} | {w} | " + " | ".join(
            f"{d[k]['mean']:.{4 if k == 'PR-AUC' else 3}f} ± {d[k]['std']:.{4 if k == 'PR-AUC' else 3}f}" for k, _, _ in keys) + " |")
    open(os.path.join(base, "combine_tables.md"), "w").write("\n".join(L) + "\n")
    json.dump({"dataset": a.name, "selected": sel, "agg": agg}, open(os.path.join(base, "combine_summary.json"), "w"), indent=1)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["models", "table"])
    ap.add_argument("--name", default="HI-Small")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--threads", type=int, default=7)
    a = ap.parse_args()
    {"models": models, "table": table}[a.mode](a)


if __name__ == "__main__":
    main()
