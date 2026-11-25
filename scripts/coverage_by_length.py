"""Coverage of closed walks by length cap, used for the coverage figure (read from the cached snapshots).

    python scripts/coverage_by_length.py

For every IBM AML set and the test snapshot, the share of illicit, licit and CYCLE accounts
that lie on at least one closed walk of length <= k in the binary observed graph, for
k = 2..12 (columns of diag_bin12 are L = 2..12). Writes results/ibm_aml/<NAME>/coverage_by_L.json.
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ibm_aml_common as C  # noqa: E402

for name in ("HI-Small", "LI-Small", "HI-Medium", "LI-Medium"):
    out = os.path.join(C.ROOT, "results", name, "coverage_by_L.json")
    if os.path.exists(out):
        continue
    snaps = pickle.load(open(os.path.join(C.ROOT, "data", "ibm_aml_proc", f"{name}_snapshots.pkl"), "rb"))
    s = snaps["test"]
    d = s["diag_bin12"][s["nodes"]] > 0
    y, pm = s["y"], s["pmask"]
    groups = {"illicit": y == 1, "licit": y == 0, "CYCLE": pm[:, 0] & (y == 1)}
    res = {"dataset": name, "split": "test", "k": list(range(2, 13)),
           "n": {g: int(m.sum()) for g, m in groups.items()}, "share": {},
           "n_unattributed_illicit": int(((y == 1) & ~pm.any(1)).sum()),
           "n_transactions_observed": int(s["n_obs"]), "n_accounts": int(len(s["attr"]))}
    cum = np.cumsum(d, axis=1) > 0
    for g, m in groups.items():
        res["share"][g] = [float(cum[m, j].mean()) for j in range(cum.shape[1])]
    json.dump(res, open(out, "w"), indent=1)
    print(name, {g: [round(v, 3) for v in res["share"][g]] for g in groups}, flush=True)
    del snaps, s, d
