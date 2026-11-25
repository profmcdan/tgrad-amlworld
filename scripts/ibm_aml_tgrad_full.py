"""Full T-GRAD detector on IBM AML, using the simulator's TGradNeural (encoder +
VP-SDE + energy guidance, streamed) and the T-GRAD head on
    zeta_i = [ a_i || m_i (T-GRAD closed-walk block) || z_i || rho_i || -E(z_i) ].

    python scripts/ibm_aml_tgrad_full.py --name HI-Small --seed 0 [--enc-epochs 2]

The encoder is trained on training-window transactions with training-window labels
only, then run once over the whole stream; embeddings are read at each snapshot time.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ibm_aml_common as C  # noqa: E402
import tgrad.simulation.detectors as D  # noqa: E402  (src/tgrad, as in profmcdan/tgrad-event-simulator)
from ibm_aml_models import blocks  # noqa: E402

MSG_DIM = 11


def ibm_messages(a, lo, hi):
    h = 2 * np.pi * ((a["t_commit"][lo:hi]) % 24.0) / 24.0
    fmt = a["fmt"][lo:hi]
    return np.column_stack([np.log1p(a["usd"][lo:hi]) / 10.0] + [fmt == k for k in range(7)]
                           + [a["xcur"][lo:hi], np.sin(h), np.cos(h)]).astype(np.float32)


D.ledger_messages = ibm_messages
D.EDGE_DIM = MSG_DIM

# Memory fix, numerically neutral: NodeMemoryV2.get_memory already detaches every read,
# so no gradient ever flows through the memory buffer; but set_memory writes tensors that
# still carry autograd history into the buffer, which then keeps a CopySlices chain that
# grows with every event (~110 GB per process after <1 epoch of 3M events here).
# Storing detached values gives identical forward values and identical gradients.
from tgrad.models.temporal_encoder_v2 import NodeMemoryV2  # noqa: E402


def _set_memory_detached(self, node_ids, values, timestamps):
    self.memory[node_ids] = values.detach()
    self.last_update[node_ids] = timestamps.detach()


NodeMemoryV2.set_memory = _set_memory_detached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="HI-Small")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enc-epochs", type=int, default=2)
    ap.add_argument("--head-epochs", type=int, default=300)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--max-train-tx", type=int, default=0, help="timing runs only")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    z = C.load(a.name)
    snaps = pickle.load(open(os.path.join(C.ROOT, "data", "ibm_aml_proc", f"{a.name}_snapshots.pkl"), "rb"))
    tau = snaps["tau"]
    S = {k: snaps[k] for k in ("train", "val", "test")}
    T1, T2 = C.cuts(z)
    n_train = int(np.searchsorted(z["t"], T1, side="left"))
    if a.max_train_tx:
        n_train = min(n_train, a.max_train_tx)
    N = len(z["accounts"])
    led = {"src": z["src"].astype(np.int64), "dst": z["dst"].astype(np.int64),
           "t_commit": z["t"] / 60.0, "usd": z["usd"], "fmt": z["fmt"], "xcur": z["xcur"]}
    node_label = np.zeros(N, np.int64)
    node_label[S["train"]["nodes"]] = S["train"]["y"]          # training-window labels only

    t0 = time.time()
    neu = D.TGradNeural(N, seed=a.seed).fit(led, n_train, node_label, epochs=a.enc_epochs)
    fit_s = time.time() - t0
    print(f"encoder fit {fit_s:.0f}s on {n_train} tx, loss {neu.loss_hist}", flush=True)
    if a.max_train_tx:
        return

    feats = {}
    t0 = time.time()
    n_val = int(np.searchsorted(z["t"], T2, side="left"))
    for k, upto in (("train", n_train), ("val", n_val), ("test", len(z["t"]))):
        neu.advance(led, upto)
        feats[k] = neu.features(S[k]["nodes"])
    stream_s = time.time() - t0
    B = {k: blocks(s, tau)["tgrad"][s["nodes"]] for k, s in S.items()}
    X = {k: np.hstack([B[k], feats[k]]) for k in S}
    ytr = S["train"]["y"]
    head = D.HeadModel(a.head_epochs, a.seed).fit(X["train"], ytr)
    sc = {k: head.score(X[k]) for k in S}
    thr1, thr2 = C.best_f1_threshold(ytr, sc["train"]), C.fpr_threshold(ytr, sc["train"])
    res = {"_meta": {"seed": a.seed, "enc_epochs": a.enc_epochs, "encoder_fit_sec": fit_s,
                     "stream_sec": stream_s, "loss_hist": neu.loss_hist, "tau_h": tau},
           "T-GRAD (full detector)": {k: C.evaluate(S[k]["y"], sc[k], S[k]["pmask"], thr1, thr2) for k in S}}
    # the neural part alone (energy posterior, no monitor block, no attributes)
    e = {k: feats[k][:, -1] for k in S}
    t1, t2 = C.best_f1_threshold(ytr, e["train"]), C.fpr_threshold(ytr, e["train"])
    res["T-GRAD energy logit only"] = {k: C.evaluate(S[k]["y"], e[k], S[k]["pmask"], t1, t2) for k in S}
    out = os.path.join(C.ROOT, "results", a.name, f"tgrad_full_seed{a.seed}.json")
    json.dump(res, open(out, "w"), indent=1)
    for nm in ("T-GRAD (full detector)", "T-GRAD energy logit only"):
        te = res[nm]["test"]
        print(f"[seed {a.seed}] {nm:<26} test AUC={te['ROC-AUC']:.3f} PR={te['PR-AUC']:.4f} F1={te['F1']:.3f} "
              f"R@1%={te['Recall@1%FPR']:.3f} CYCLE R@1%={te['by_pattern']['CYCLE']['Recall@1%FPR']:.3f}", flush=True)


if __name__ == "__main__":
    main()
