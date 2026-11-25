"""Learned detectors on the IBM AML snapshots (run ibm_aml_snapshots.py first).

    python scripts/ibm_aml_models.py --name HI-Small --seed 0

Same GNN protocol as the closed-walk monitor paper, profmcdan/tgrad (two-layer, hidden 64,
undirected graph, class-weighted cross-entropy, Adam lr 1e-2, wd 5e-4, 300 full-batch
epochs, best-F1 threshold on training scores), fitted on the training snapshot and
applied inductively to the validation and test snapshots. Three feature sets:
  attr     account attributes from observed transactions
  matched  attr + log1p[out-degree, in-degree, (A^L)_ii L=2..6]  (binary adjacency)
  tgrad    attr + T-GRAD closed-walk block as in the simulator:
           log1p[(A_tau^L)_ii L=2..6, decayed out/in strength, streaming score]
"GraphSAGE + T-GRAD block" is the T-GRAD + GraphSAGE model.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ibm_aml_common as C  # noqa: E402
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv, TransformerConv  # noqa: E402

KINDS = ["GCN", "GraphSAGE", "GAT", "GIN", "Graph Transformer"]


class GNN(nn.Module):
    def __init__(self, kind, d, h=64):
        super().__init__(); self.act = nn.ReLU(); self.drop = nn.Dropout(0.3)
        if kind == "GCN":
            self.c1, self.c2 = GCNConv(d, h), GCNConv(h, 2)
        elif kind == "GraphSAGE":
            self.c1, self.c2 = SAGEConv(d, h), SAGEConv(h, 2)
        elif kind == "GAT":
            self.c1 = GATConv(d, h, heads=4, concat=True, dropout=0.3)
            self.c2 = GATConv(h * 4, 2, heads=1, concat=False, dropout=0.3)
        elif kind == "GIN":
            self.c1 = GINConv(nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h)))
            self.c2 = GINConv(nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2)))
        elif kind == "Graph Transformer":
            self.c1 = TransformerConv(d, h, heads=4, concat=True, dropout=0.3)
            self.c2 = TransformerConv(h * 4, 2, heads=1, concat=False, dropout=0.3)

    def forward(self, x, ei):
        return self.c2(self.drop(self.act(self.c1(x, ei))), ei)


class FraudHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def blocks(s, tau):
    l = np.log1p
    return {"attr": s["attr"],
            "matched": np.hstack([s["attr"], l(s["deg"]), l(s["diag_bin"])]),
            "tgrad": np.hstack([s["attr"], l(s["diag_tau"][tau]), l(s["str_tau"][tau]), l(s["stream"])[:, None]])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="HI-Small")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    snaps = pickle.load(open(os.path.join(C.ROOT, "data", "ibm_aml_proc", f"{a.name}_snapshots.pkl"), "rb"))
    tau = snaps["tau"]
    S = {k: snaps[k] for k in ("train", "val", "test")}
    B = {k: blocks(s, tau) for k, s in S.items()}
    tr = S["train"]; ytr = tr["y"]
    pw = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
    out_dir = os.path.join(C.ROOT, "results", a.name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"models_seed{a.seed}.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    res.update({"_meta": {"seed": a.seed, "epochs": a.epochs, "tau_h": tau, "pos_weight": pw}})

    def finish(name, sc, secs):
        thr1 = C.best_f1_threshold(ytr, sc["train"]); thr2 = C.fpr_threshold(ytr, sc["train"])
        res[name] = {k: C.evaluate(S[k]["y"], sc[k], S[k]["pmask"], thr1, thr2) for k in S}
        res[name]["train_sec"] = secs
        te = res[name]["test"]
        print(f"[seed {a.seed}] {name:<34} test AUC={te['ROC-AUC']:.3f} PR={te['PR-AUC']:.4f} F1={te['F1']:.3f} "
              f"R@1%={te['Recall@1%FPR']:.3f} CYCLE R@1%={te['by_pattern']['CYCLE']['Recall@1%FPR']:.3f} "
              f"AUC={te['by_pattern']['CYCLE']['AUC_vs_licit']:.3f} ({secs:.0f}s)", flush=True)
        json.dump(res, open(out_path, "w"), indent=1)

    for fs in ("attr", "matched", "tgrad"):
        mu = B["train"][fs][tr["nodes"]].mean(0); sd = B["train"][fs][tr["nodes"]].std(0) + 1e-8
        X = {k: torch.tensor((B[k][fs] - mu) / sd, dtype=torch.float32) for k in S}
        EI = {k: torch.tensor(S[k]["ei"]) for k in S}
        trn = torch.tensor(tr["nodes"]); yt = torch.tensor(ytr)

        # MLP head (no graph), same head as the T-GRAD experiments
        name = {"attr": "MLP (attributes)", "matched": "MLP + matched block", "tgrad": "MLP + T-GRAD block"}[fs]
        if (not a.only or a.only in name) and name not in res:
            torch.manual_seed(a.seed); t0 = time.time()
            h = FraudHead(X["train"].shape[1]); o = torch.optim.Adam(h.parameters(), lr=1e-3, weight_decay=1e-4)
            yf = yt.float()
            for _ in range(a.epochs):
                h.train(); o.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(h(X["train"][trn]), yf,
                                                                      pos_weight=torch.tensor([pw]))
                loss.backward(); o.step()
            h.eval()
            with torch.no_grad():
                sc = {k: torch.sigmoid(h(X[k][torch.tensor(S[k]["nodes"])])).numpy() for k in S}
            finish(name, sc, time.time() - t0)

        for kind in KINDS:
            name = kind + {"attr": "", "matched": " (matched)", "tgrad": " + T-GRAD block"}[fs]
            if (a.only and a.only not in name) or name in res:
                continue                      # resume: models already in the JSON are kept
            torch.manual_seed(a.seed); t0 = time.time()
            net = GNN(kind, X["train"].shape[1])
            o = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=5e-4)
            cw = torch.tensor([1.0, pw])
            for ep in range(a.epochs):
                net.train(); o.zero_grad()
                loss = nn.functional.cross_entropy(net(X["train"], EI["train"])[trn], yt, weight=cw)
                loss.backward(); o.step()
            net.eval()
            with torch.no_grad():
                sc = {k: torch.softmax(net(X[k], EI[k]), 1)[:, 1].numpy()[S[k]["nodes"]] for k in S}
            finish(name, sc, time.time() - t0)


if __name__ == "__main__":
    main()
