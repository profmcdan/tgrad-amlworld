"""Improvement 3: why the T-GRAD closed-walk block hurts, and a corrected block.

    python scripts/ibm_aml_ablation.py --name HI-Small --seed 0

Same MLP and GraphSAGE protocol as scripts/ibm_aml_models.py (whose attr / matched /
full T-GRAD block runs are the reference). Feature variants, each = attributes + one block:
  decayed walks      log1p (A_tau^L)_ii, L = 2..6
  decayed strength   log1p decayed out/in strength
  streaming score    log1p streaming monitor score
  block, ranked      the whole T-GRAD block, each column rank-scaled to [0, 1] within the
                     snapshot over accounts seen so far (unlabelled, per snapshot)
  temporal cycles    log1p temporal-cycle count at the validation-selected (W, cap)
  matched + temporal matched block + temporal-cycle count
Selection (in ibm_aml_aggregate_improve.py): the variant with the best mean validation
ROC-AUC per model. Also writes, for seed 0, a train-vs-test shift diagnostic: the
Kolmogorov-Smirnov statistic of each feature between licit accounts in the training and
test snapshots.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import ks_2samp, rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ibm_aml_common as C  # noqa: E402
from ibm_aml_models import GNN, FraudHead  # noqa: E402


def ranked(M, seen):
    R = np.zeros_like(M, dtype=np.float32)
    for j in range(M.shape[1]):
        R[seen, j] = rankdata(M[seen, j]) / seen.sum()
    return R


def variants(s, tau, tc):
    l = np.log1p
    blk = np.hstack([l(s["diag_tau"][tau]), l(s["str_tau"][tau]), l(s["stream"])[:, None]])
    seen = (s["attr"][:, :3] > 0).any(1)
    t = l(tc)[:, None]
    match = np.hstack([l(s["deg"]), l(s["diag_bin"])])
    return {"decayed walks": np.hstack([s["attr"], l(s["diag_tau"][tau])]),
            "decayed strength": np.hstack([s["attr"], l(s["str_tau"][tau])]),
            "streaming score": np.hstack([s["attr"], l(s["stream"])[:, None]]),
            "T-GRAD block, ranked": np.hstack([s["attr"], ranked(blk, seen)]),
            "temporal cycles": np.hstack([s["attr"], t]),
            "matched + temporal cycles": np.hstack([s["attr"], match, t])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="HI-Small")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    snaps = pickle.load(open(os.path.join(C.ROOT, "data", "ibm_aml_proc", f"{a.name}_snapshots.pkl"), "rb"))
    tau = snaps["tau"]
    S = {k: snaps[k] for k in ("train", "val", "test")}
    base = os.path.join(C.ROOT, "results", a.name)
    imp = json.load(open(os.path.join(base, "improved_monitors.json")))
    sel = imp["temporal_selected"]
    W, cap = sel.split("W=")[1].split("h")[0], int(sel.split("L<=")[1])
    tcz = np.load(os.path.join(C.ROOT, "data", "ibm_aml_proc", f"{a.name}_tcycle_W{W}.npz"))
    V = {k: variants(s, tau, tcz[f"{k}_L{cap}"].astype(float)) for k, s in S.items()}
    tr = S["train"]; ytr = tr["y"]
    pw = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
    out_path = os.path.join(base, f"ablation_seed{a.seed}.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    res["_meta"] = {"seed": a.seed, "epochs": a.epochs, "tau_h": tau, "temporal_cycle_config": sel}

    if a.seed == 0 and "_shift_ks" not in res:
        from ibm_aml_models import blocks
        cols = ([f"attr:{n}" for n in C.ATTR_NAMES] + [f"decayed walk L={L}" for L in range(2, 7)]
                + ["decayed out-strength", "decayed in-strength", "streaming score"])
        Btr, Bte = blocks(S["train"], tau)["tgrad"], blocks(S["test"], tau)["tgrad"]
        ltr = S["train"]["nodes"][S["train"]["y"] == 0]; lte = S["test"]["nodes"][S["test"]["y"] == 0]
        rng = np.random.default_rng(0)
        ltr = rng.choice(ltr, min(50_000, len(ltr)), replace=False); lte = rng.choice(lte, min(50_000, len(lte)), replace=False)
        res["_shift_ks"] = {c: float(ks_2samp(Btr[ltr, j], Bte[lte, j]).statistic) for j, c in enumerate(cols)}
        Mtr, Mte = blocks(S["train"], tau)["matched"], blocks(S["test"], tau)["matched"]
        mcols = ["out-degree", "in-degree"] + [f"binary walk L={L}" for L in range(2, 7)]
        for j, c in enumerate(mcols):
            res["_shift_ks"][f"matched:{c}"] = float(ks_2samp(Mtr[ltr, len(C.ATTR_NAMES) + j],
                                                              Mte[lte, len(C.ATTR_NAMES) + j]).statistic)

    def finish(name, sc, secs):
        thr1 = C.best_f1_threshold(ytr, sc["train"]); thr2 = C.fpr_threshold(ytr, sc["train"])
        res[name] = {k: C.evaluate(S[k]["y"], sc[k], S[k]["pmask"], thr1, thr2) for k in S}
        res[name]["train_sec"] = secs
        v, te = res[name]["val"], res[name]["test"]
        print(f"[seed {a.seed}] {name:<44} val AUC={v['ROC-AUC']:.3f} | test AUC={te['ROC-AUC']:.3f} "
              f"PR={te['PR-AUC']:.4f} R@1%={te['Recall@1%FPR']:.3f} "
              f"CYCLE R@1%={te['by_pattern']['CYCLE']['Recall@1%FPR']:.3f} ({secs:.0f}s)", flush=True)
        json.dump(res, open(out_path, "w"), indent=1)

    trn = torch.tensor(tr["nodes"]); yt = torch.tensor(ytr)
    for vn in V["train"]:
        mu = V["train"][vn][tr["nodes"]].mean(0); sd = V["train"][vn][tr["nodes"]].std(0) + 1e-8
        X = {k: torch.tensor((V[k][vn] - mu) / sd, dtype=torch.float32) for k in S}
        EI = {k: torch.tensor(S[k]["ei"]) for k in S}
        name = f"MLP + {vn}"
        if name not in res:
            torch.manual_seed(a.seed); t0 = time.time()
            h = FraudHead(X["train"].shape[1]); o = torch.optim.Adam(h.parameters(), lr=1e-3, weight_decay=1e-4)
            for _ in range(a.epochs):
                h.train(); o.zero_grad()
                nn.functional.binary_cross_entropy_with_logits(h(X["train"][trn]), yt.float(),
                                                               pos_weight=torch.tensor([pw])).backward()
                o.step()
            h.eval()
            with torch.no_grad():
                sc = {k: torch.sigmoid(h(X[k][torch.tensor(S[k]["nodes"])])).numpy() for k in S}
            finish(name, sc, time.time() - t0)
        name = f"GraphSAGE + {vn}"
        if name not in res:
            torch.manual_seed(a.seed); t0 = time.time()
            net = GNN("GraphSAGE", X["train"].shape[1])
            o = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=5e-4)
            cw = torch.tensor([1.0, pw])
            for _ in range(a.epochs):
                net.train(); o.zero_grad()
                nn.functional.cross_entropy(net(X["train"], EI["train"])[trn], yt, weight=cw).backward(); o.step()
            net.eval()
            with torch.no_grad():
                sc = {k: torch.softmax(net(X[k], EI[k]), 1)[:, 1].numpy()[S[k]["nodes"]] for k in S}
            finish(name, sc, time.time() - t0)


if __name__ == "__main__":
    main()
