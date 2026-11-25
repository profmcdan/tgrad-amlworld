"""Confirmatory run on fresh data (HI-Medium), with every setting fixed in advance from
HI-Small / LI-Small. Nothing is selected on HI-Medium.

    python scripts/ibm_aml_confirm.py monitors --name HI-Medium
    python scripts/ibm_aml_confirm.py models   --name HI-Medium --seed 0
    python scripts/ibm_aml_confirm.py table    --name HI-Medium

Fixed settings (chosen on HI-Small and LI-Small validation windows): tau = 72 h; the
closed-walk monitor with L = 2..12; the time-respecting cycle monitor with W = 168 h, L <= 12.
Thresholds come from the HI-Medium training snapshot, as before. Same model protocol as
scripts/ibm_aml_models.py.

Pre-registered claims (written before the HI-Medium run):
  C1  The L <= 12 closed-walk monitor has higher test CYCLE recall at 1% FPR than every
      learned model here.
  C2  The original T-GRAD block lowers GraphSAGE test ROC-AUC relative to attributes only,
      and the rank-scaled block does not (its AUC is >= attributes only - 0.005).
  C3  Adding the time-respecting cycle count raises GraphSAGE test CYCLE recall at 1% FPR
      over attributes only, without lowering test ROC-AUC by more than 0.005.

Second confirmatory run (LI-Medium), claims written before LI-Medium was downloaded, from
the HI-Small / LI-Small results of scripts/ibm_aml_combine.py. Same fixed settings. 3-seed
test means; "GT-TGRAD" = Graph Transformer + ranked block + temporal cycles, "GT-ref" =
Graph Transformer + matched block:
  D1  GT-TGRAD has higher test PR-AUC than GT-ref.
  D2  GT-TGRAD test ROC-AUC >= GT-ref test ROC-AUC - 0.005.
  D3  GT-TGRAD has higher test F1 than GT-ref.
  Exploratory (no claim): test CYCLE recall at 1% FPR of GT-TGRAD vs GT-ref.
Select this plan with --plan D (the default for LI-Medium).
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
from ibm_aml_ablation import ranked  # noqa: E402
from ibm_aml_models import GNN, FraudHead, blocks  # noqa: E402

W_FIXED, CAP_FIXED = 168, 12


def load(name):
    snaps = pickle.load(open(os.path.join(C.ROOT, "data", "ibm_aml_proc", f"{name}_snapshots.pkl"), "rb"))
    S = {k: snaps[k] for k in ("train", "val", "test")}
    tcz = np.load(os.path.join(C.ROOT, "data", "ibm_aml_proc", f"{name}_tcycle_W{W_FIXED}.npz"))
    return snaps["tau"], S, {k: tcz[f"{k}_L{CAP_FIXED}"].astype(float) for k in S}


def ev(S, f):
    tr = S["train"]; st = f("train")
    thr1, thr2 = C.best_f1_threshold(tr["y"], st), C.fpr_threshold(tr["y"], st)
    return {k: C.evaluate(S[k]["y"], f(k), S[k]["pmask"], thr1, thr2) for k in S}


def monitors(a):
    tau, S, tc = load(a.name)
    assert tau == 72.0, tau
    fs = {"Closed-walk monitor, binary, L=2..6 (original)": lambda k: C.monitor_from_diag(S[k]["diag_bin"][S[k]["nodes"]], 2),
          "Closed-walk monitor, binary, L=2..12 (fixed)": lambda k: C.monitor_from_diag(S[k]["diag_bin12"][S[k]["nodes"]], 2),
          f"Temporal-cycle monitor, W={W_FIXED}h, L<={CAP_FIXED} (fixed)": lambda k: np.log1p(tc[k][S[k]["nodes"]]),
          "Total degree (reference)": lambda k: S[k]["deg"][S[k]["nodes"]].sum(1)}
    res = {nm: ev(S, f) for nm, f in fs.items()}
    cov = {}
    for k, s in S.items():
        y, pm, nd = s["y"], s["pmask"], s["nodes"]
        g = {"illicit": y == 1, "licit": y == 0, "CYCLE": pm[:, 0] & (y == 1)}
        on = {"closed walk L<=6": (s["diag_bin"][nd] > 0).any(1), "closed walk L<=12": (s["diag_bin12"][nd] > 0).any(1),
              "temporal cycle": tc[k][nd] > 0, "non-trivial SCC": s["in_scc"][nd]}
        cov[k] = {gn: {"n": int(m.sum()), **{on_n: float(o[m].mean()) for on_n, o in on.items()}} for gn, m in g.items()}
    out = {"dataset": a.name, "fixed": {"tau_h": tau, "W_h": W_FIXED, "cap": CAP_FIXED},
           "splits": {k: {"n_accounts": int(len(s["nodes"])), "n_illicit": int(s["y"].sum()),
                          "n_per_pattern": {p: int((s["pmask"][:, i] & (s["y"] == 1)).sum()) for i, p in enumerate(C.PATTERNS)},
                          "n_unattributed": int(((s["y"] == 1) & ~s["pmask"].any(1)).sum())} for k, s in S.items()},
           "coverage": cov, "models": res}
    json.dump(out, open(os.path.join(C.ROOT, "results", a.name, "confirm_monitors.json"), "w"), indent=1)
    for nm, r in res.items():
        te = r["test"]
        print(f"{nm:<52} test AUC {te['ROC-AUC']:.3f} PR {te['PR-AUC']:.4f} R@1% {te['Recall@1%FPR']:.3f} "
              f"CYC R@1% {te['by_pattern']['CYCLE']['Recall@1%FPR']:.3f} CYC-AUC {te['by_pattern']['CYCLE']['AUC_vs_licit']:.3f}")
    print(json.dumps(cov["test"], indent=1))


def feature_sets(s, tau, tc):
    l = np.log1p
    b = blocks(s, tau)
    blk = b["tgrad"][:, s["attr"].shape[1]:]
    seen = (s["attr"][:, :3] > 0).any(1)
    return {"attributes only": b["attr"], "matched block": b["matched"], "T-GRAD block": b["tgrad"],
            "T-GRAD block, ranked": np.hstack([s["attr"], ranked(blk, seen)]),
            "temporal cycles": np.hstack([s["attr"], l(tc)[:, None]]),
            "ranked block + temporal cycles": np.hstack([s["attr"], ranked(blk, seen), l(tc)[:, None]])}


PLAN = [("MLP", fs) for fs in ("attributes only", "matched block", "T-GRAD block", "T-GRAD block, ranked", "temporal cycles")] \
    + [("GraphSAGE", fs) for fs in ("attributes only", "matched block", "T-GRAD block", "T-GRAD block, ranked", "temporal cycles")] \
    + [("Graph Transformer", "matched block")]
PLAN_D = [("GraphSAGE", fs) for fs in ("attributes only", "T-GRAD block, ranked", "temporal cycles")] \
    + [("Graph Transformer", "matched block"), ("Graph Transformer", "ranked block + temporal cycles")]


def plan_of(a):
    p = a.plan or ("D" if a.name == "LI-Medium" else "C")
    return PLAN_D if p == "D" else PLAN


def models(a):
    torch.set_num_threads(a.threads)
    tau, S, tc = load(a.name)
    tr = S["train"]; ytr = tr["y"]
    pw = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
    out_path = os.path.join(C.ROOT, "results", a.name, f"confirm_seed{a.seed}.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    res["_meta"] = {"seed": a.seed, "epochs": a.epochs, "tau_h": tau, "pos_weight": pw}
    trn = torch.tensor(tr["nodes"]); yt = torch.tensor(ytr)
    cache = {}
    for kind, fs in plan_of(a):
        name = f"{kind} + {fs}"
        if name in res or (a.only and a.only not in name) or (a.skip and a.skip in name):
            continue
        if fs not in cache:
            cache.clear()
            F = {k: feature_sets(s, tau, tc[k])[fs] for k, s in S.items()}
            mu = F["train"][tr["nodes"]].mean(0); sd = F["train"][tr["nodes"]].std(0) + 1e-8
            cache[fs] = {k: torch.tensor((F[k] - mu) / sd, dtype=torch.float32) for k in S}
            del F
        X = cache[fs]
        torch.manual_seed(a.seed); t0 = time.time()
        if kind == "MLP":
            h = FraudHead(X["train"].shape[1]); o = torch.optim.Adam(h.parameters(), lr=1e-3, weight_decay=1e-4)
            for _ in range(a.epochs):
                h.train(); o.zero_grad()
                nn.functional.binary_cross_entropy_with_logits(h(X["train"][trn]), yt.float(),
                                                               pos_weight=torch.tensor([pw])).backward(); o.step()
            h.eval()
            with torch.no_grad():
                sc = {k: torch.sigmoid(h(X[k][torch.tensor(S[k]["nodes"])])).numpy() for k in S}
        else:
            EI = {k: torch.tensor(S[k]["ei"]) for k in S}
            net = GNN(kind, X["train"].shape[1]); o = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=5e-4)
            cw = torch.tensor([1.0, pw])
            for _ in range(a.epochs):
                net.train(); o.zero_grad()
                nn.functional.cross_entropy(net(X["train"], EI["train"])[trn], yt, weight=cw).backward(); o.step()
            net.eval()
            with torch.no_grad():
                sc = {k: torch.softmax(net(X[k], EI[k]), 1)[:, 1].numpy()[S[k]["nodes"]] for k in S}
            del EI
        thr1, thr2 = C.best_f1_threshold(ytr, sc["train"]), C.fpr_threshold(ytr, sc["train"])
        res[name] = {k: C.evaluate(S[k]["y"], sc[k], S[k]["pmask"], thr1, thr2) for k in S}
        res[name]["train_sec"] = time.time() - t0
        te = res[name]["test"]
        print(f"[seed {a.seed}] {name:<40} test AUC={te['ROC-AUC']:.3f} PR={te['PR-AUC']:.4f} F1={te['F1']:.3f} "
              f"R@1%={te['Recall@1%FPR']:.3f} CYCLE R@1%={te['by_pattern']['CYCLE']['Recall@1%FPR']:.3f} "
              f"({res[name]['train_sec']:.0f}s)", flush=True)
        json.dump(res, open(out_path, "w"), indent=1)


def table(a):
    import glob
    base = os.path.join(C.ROOT, "results", a.name)
    mon = json.load(open(os.path.join(base, "confirm_monitors.json")))
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(base, "confirm_seed*.json")))]
    keys = [("ROC-AUC", lambda e: e["ROC-AUC"]), ("PR-AUC", lambda e: e["PR-AUC"]), ("F1", lambda e: e["F1"]),
            ("R@1%FPR", lambda e: e["Recall@1%FPR"]), ("CYCLE R@1%FPR", lambda e: e["by_pattern"]["CYCLE"]["Recall@1%FPR"]),
            ("CYCLE AUC", lambda e: e["by_pattern"]["CYCLE"]["AUC_vs_licit"])]
    agg = {}
    for nm, r in mon["models"].items():
        agg[nm] = {k: (f(r["test"]), 0.0, 1) for k, f in keys}
    for kind, fs in plan_of(a):
        nm = f"{kind} + {fs}"
        rs = [r[nm]["test"] for r in runs if nm in r]
        if rs:
            agg[nm] = {k: (float(np.mean([f(e) for e in rs])), float(np.std([f(e) for e in rs])), len(rs)) for k, f in keys}
    L = [f"# IBM AML {a.name}: confirmatory run (generated by scripts/ibm_aml_confirm.py table)\n",
         f"Settings fixed in advance: {mon['fixed']}. Test window: {mon['splits']['test']['n_accounts']} accounts, "
         f"{mon['splits']['test']['n_illicit']} illicit, {mon['splits']['test']['n_per_pattern']['CYCLE']} CYCLE.\n",
         "| Model | seeds | " + " | ".join(k for k, _ in keys) + " |", "|---|---|" + "---|" * len(keys)]
    for nm, d in agg.items():
        n = next(iter(d.values()))[2]
        L.append(f"| {nm} | {n} | " + " | ".join(f"{m:.{4 if k == 'PR-AUC' else 3}f}" + (f" ± {s:.3f}" if n > 1 else "")
                                                for k, (m, s, _) in d.items()) + " |")
    g = lambda nm, k: agg[nm][k][0] if nm in agg else float("nan")
    learned = [nm for nm in agg if nm.split(" + ")[0] in ("MLP", "GraphSAGE", "Graph Transformer")]
    mon12 = "Closed-walk monitor, binary, L=2..12 (fixed)"
    if plan_of(a) is PLAN_D:
        ref, tg = "Graph Transformer + matched block", "Graph Transformer + ranked block + temporal cycles"
        claims = {"D1": g(tg, "PR-AUC") > g(ref, "PR-AUC"),
                  "D2": g(tg, "ROC-AUC") >= g(ref, "ROC-AUC") - 0.005,
                  "D3": g(tg, "F1") > g(ref, "F1")}
        hs = lambda c: "holds" if claims[c] else "fails"
        L += ["\n## Pre-registered claims\n",
              f"- D1 (GT + ranked block + temporal cycles has higher PR-AUC than GT + matched): **{hs('D1')}**",
              f"- D2 (its ROC-AUC >= GT + matched - 0.005): **{hs('D2')}**",
              f"- D3 (its F1 is higher than GT + matched): **{hs('D3')}**",
              f"- Exploratory: CYCLE R@1%FPR {g(tg, 'CYCLE R@1%FPR'):.3f} vs {g(ref, 'CYCLE R@1%FPR'):.3f} (reference)"]
    else:
        c1 = all(g(mon12, "CYCLE R@1%FPR") > g(nm, "CYCLE R@1%FPR") for nm in learned)
        c2 = (g("GraphSAGE + T-GRAD block", "ROC-AUC") < g("GraphSAGE + attributes only", "ROC-AUC")
              and g("GraphSAGE + T-GRAD block, ranked", "ROC-AUC") >= g("GraphSAGE + attributes only", "ROC-AUC") - 0.005)
        c3 = (g("GraphSAGE + temporal cycles", "CYCLE R@1%FPR") > g("GraphSAGE + attributes only", "CYCLE R@1%FPR")
              and g("GraphSAGE + temporal cycles", "ROC-AUC") >= g("GraphSAGE + attributes only", "ROC-AUC") - 0.005)
        claims = {"C1": c1, "C2": c2, "C3": c3}
        L += ["\n## Pre-registered claims\n", f"- C1 (L<=12 monitor beats every learned model on CYCLE recall): **{'holds' if c1 else 'fails'}**",
              f"- C2 (original block hurts GraphSAGE, ranked block does not): **{'holds' if c2 else 'fails'}**",
              f"- C3 (temporal-cycle count raises GraphSAGE CYCLE recall at no AUC cost): **{'holds' if c3 else 'fails'}**"]
    L += [
          "\n## Coverage (test)\n", "```", json.dumps(mon["coverage"]["test"], indent=1), "```"]
    open(os.path.join(base, "confirm_tables.md"), "w").write("\n".join(L) + "\n")
    json.dump({"claims": claims, "test": agg}, open(os.path.join(base, "confirm_summary.json"), "w"), indent=1)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["monitors", "models", "table"])
    ap.add_argument("--name", default="HI-Medium")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--plan", choices=["C", "D"], default=None, help="C = HI-Medium plan, D = LI-Medium plan")
    a = ap.parse_args()
    {"monitors": monitors, "models": models, "table": table}[a.mode](a)


if __name__ == "__main__":
    main()
