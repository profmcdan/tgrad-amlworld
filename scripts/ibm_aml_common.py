"""Shared code for the IBM AML (Altman et al., 2023) evaluation of T-GRAD.

Protocol (temporal, inductive):
  * transactions are sorted by time and cut at the 60% and 80% quantiles of the
    transaction count: train window [0, T1), validation [T1, T2), test [T2, end];
  * a snapshot at the end of each window uses every transaction observed so far
    (no future edges) for the graph, the account attributes and the monitors;
  * the accounts scored in a snapshot are those active in that window, labelled
    illicit iff they take part in an `Is Laundering` transaction in that window;
  * models are fitted on the training snapshot only; standardisation statistics,
    the best-F1 threshold, the 1%-FPR threshold and the decay constant tau are all
    chosen on the training snapshot.
"""
import os
import sys

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from tgrad.simulation.monitor import closed_walk_diagonals, decayed_adjacency  # noqa: E402

PATTERNS = ["CYCLE", "FAN-IN", "FAN-OUT", "GATHER-SCATTER", "SCATTER-GATHER",
            "STACK", "BIPARTITE", "RANDOM"]
L_MAX = 6
ATTR_NAMES = (["out_n", "in_n", "self_n", "log_out_usd", "log_in_usd", "log_out_max", "log_in_max",
               "log_out_mean", "log_in_mean", "out_cp", "in_cp"]
              + [f"out_fmt_{i}" for i in range(7)] + [f"in_fmt_{i}" for i in range(7)]
              + ["xcur_share", "n_pay_cur", "pass_through", "log_span_h", "log_recency_h"])


def load(name):
    z = np.load(os.path.join(ROOT, "data", "ibm_aml_proc", f"{name}.npz"))
    return {k: z[k] for k in z.files}


def cuts(z, q=(0.6, 0.8)):
    t = z["t"]
    return [int(t[int(qq * len(t))]) for qq in q]


def binary_adjacency(src, dst, N):
    m = src != dst                                  # self-transfers removed
    A = sp.csr_matrix((np.ones(int(m.sum())), (src[m], dst[m])), shape=(N, N))
    A.data[:] = 1.0
    return A


def monitor_from_diag(diag, L_min=2):
    """M_i = sum_{L=L_min..L_max} 2^-L (A^L)_ii; diag columns are L = 2..L_max."""
    w = np.array([2.0 ** -L if L >= L_min else 0.0 for L in range(2, diag.shape[1] + 2)])
    return diag @ w


def attributes(src, dst, t, usd, fmt, cur, xcur, N, t_hi):
    l = np.log1p
    ns = src != dst
    s, d, u = src[ns], dst[ns], usd[ns]
    out_n = np.bincount(s, minlength=N).astype(float)
    in_n = np.bincount(d, minlength=N).astype(float)
    self_n = np.bincount(src[~ns], minlength=N).astype(float)
    out_a = np.bincount(s, weights=u, minlength=N)
    in_a = np.bincount(d, weights=u, minlength=N)
    out_mx = np.zeros(N); np.maximum.at(out_mx, s, u)
    in_mx = np.zeros(N); np.maximum.at(in_mx, d, u)
    pr = np.unique(np.stack([s, d], 1), axis=0)
    out_cp = np.bincount(pr[:, 0], minlength=N).astype(float)
    in_cp = np.bincount(pr[:, 1], minlength=N).astype(float)
    of = np.stack([np.bincount(src[fmt == k], minlength=N) for k in range(7)], 1) / np.maximum(
        np.bincount(src, minlength=N), 1)[:, None]
    inf = np.stack([np.bincount(dst[fmt == k], minlength=N) for k in range(7)], 1) / np.maximum(
        np.bincount(dst, minlength=N), 1)[:, None]
    xc = np.bincount(src, weights=xcur.astype(float), minlength=N) / np.maximum(np.bincount(src, minlength=N), 1)
    pc = np.unique(np.stack([src, cur.astype(np.int32)], 1), axis=0)
    n_cur = np.bincount(pc[:, 0], minlength=N).astype(float)
    pas = np.minimum(out_a, in_a) / (np.maximum(out_a, in_a) + 1e-9)
    first = np.full(N, np.inf); last = np.full(N, -np.inf)
    for a in (src, dst):
        np.minimum.at(first, a, t); np.maximum.at(last, a, t)
    seen = np.isfinite(first)
    span = np.where(seen, (last - first) / 60.0, 0.0)
    rec = np.where(seen, (t_hi - last) / 60.0, 0.0)
    X = np.column_stack([l(out_n), l(in_n), l(self_n), l(out_a), l(in_a), l(out_mx), l(in_mx),
                         l(out_a / np.maximum(out_n, 1)), l(in_a / np.maximum(in_n, 1)), l(out_cp), l(in_cp),
                         of, inf, xc, n_cur, pas, l(span), l(rec)])
    return X.astype(np.float32)


def snapshot(z, lo, hi, taus, end=False):
    """Snapshot at the end of window [lo, hi) (minutes); `end` includes t == hi."""
    t = z["t"]
    n_obs = np.searchsorted(t, hi, side="right" if end else "left")
    n_lo = np.searchsorted(t, lo, side="left")
    N = len(z["accounts"])
    src, dst = z["src"][:n_obs], z["dst"][:n_obs]
    tt = t[:n_obs]
    out = {"lo": lo, "hi": hi, "n_obs": int(n_obs)}
    out["attr"] = attributes(src, dst, tt, z["usd"][:n_obs], z["fmt"][:n_obs], z["cur"][:n_obs],
                             z["xcur"][:n_obs], N, hi)
    A = binary_adjacency(src, dst, N)
    out["A_nnz"] = int(A.nnz)
    out["diag_bin"] = closed_walk_diagonals(A, L_MAX)                     # (N, 5), L = 2..6
    # post-hoc diagnostics (not the thesis monitor): closed walks up to L = 12, and
    # membership of a non-trivial strongly connected component (on a cycle of any length)
    out["diag_bin12"] = closed_walk_diagonals(A, 12)                      # (N, 11), L = 2..12
    from scipy.sparse.csgraph import connected_components
    _, lab = connected_components(A, directed=True, connection="strong")
    out["in_scc"] = np.bincount(lab)[lab] > 1
    out["deg"] = np.column_stack([np.asarray(A.sum(1)).ravel(), np.asarray(A.sum(0)).ravel()])
    U = (A + A.T).tocoo()
    out["ei"] = np.vstack([U.row, U.col]).astype(np.int64)                # undirected, unique, no self loops
    t_h = tt / 60.0
    out["diag_tau"], out["str_tau"] = {}, {}
    for tau in taus:
        Ad = decayed_adjacency(src, dst, t_h, hi / 60.0, N, tau)
        out["diag_tau"][tau] = closed_walk_diagonals(Ad, L_MAX)
        out["str_tau"][tau] = np.column_stack([np.asarray(Ad.sum(1)).ravel(), np.asarray(Ad.sum(0)).ravel()])
    # accounts active in the window and their window labels
    ws, wd, wy = z["src"][n_lo:n_obs], z["dst"][n_lo:n_obs], z["y"][n_lo:n_obs]
    active = np.zeros(N, bool); active[ws] = True; active[wd] = True
    y = np.zeros(N, np.int8); y[ws[wy == 1]] = 1; y[wd[wy == 1]] = 1
    pt = z["p_t"]
    pm = (pt >= lo) & ((pt <= hi) if end else (pt < hi))
    pmask = np.zeros((N, len(PATTERNS)), bool)
    for a in (z["p_src"][pm], z["p_dst"][pm]):
        pmask[a, z["p_type"][pm]] = True
    nodes = np.where(active)[0]
    out.update(nodes=nodes, y=y[nodes].astype(int), pmask=pmask[nodes])
    return out


# ------------------------------------------------------------------ metrics
def best_f1_threshold(y, s):
    from sklearn.metrics import precision_recall_curve
    p, r, th = precision_recall_curve(y, s)
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(th[int(np.argmax(f1[:-1]))]) if len(th) else 0.5


def fpr_threshold(y, s, fpr=0.01):
    """Score above which `fpr` of the training licit accounts fall."""
    return float(np.quantile(s[y == 0], 1.0 - fpr))


def evaluate(y, s, pmask, thr_f1, thr_fpr):
    s = np.asarray(s, float)
    yp = s >= thr_f1
    yq = s > thr_fpr
    lic = y == 0
    r = {"n": int(len(y)), "n_illicit": int(y.sum()),
         "ROC-AUC": float(roc_auc_score(y, s)), "PR-AUC": float(average_precision_score(y, s)),
         "F1": float(f1_score(y, yp, zero_division=0)),
         "Precision": float(precision_score(y, yp, zero_division=0)),
         "Recall": float(recall_score(y, yp, zero_division=0)),
         "Recall@1%FPR": float(yq[y == 1].mean()) if y.sum() else float("nan"),
         "FPR@1%thr (realised)": float(yq[lic].mean()),
         "by_pattern": {}}
    groups = {p: pmask[:, k] & (y == 1) for k, p in enumerate(PATTERNS)}
    groups["UNATTRIBUTED"] = (y == 1) & ~pmask.any(1)
    for g, m in groups.items():
        if m.sum() == 0:
            r["by_pattern"][g] = {"n": 0}
            continue
        sel = m | lic
        r["by_pattern"][g] = {"n": int(m.sum()),
                              "Recall@1%FPR": float(yq[m].mean()),
                              "Recall@F1thr": float(yp[m].mean()),
                              "AUC_vs_licit": float(roc_auc_score(m[sel], s[sel]))}
    return r
