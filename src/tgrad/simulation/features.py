"""
Causal per-epoch account snapshots for the Objective-6 simulator.

At every scoring epoch t the detector sees the accounts that transacted in the
last ``active_h`` hours and, for each, features computed ONLY from transactions
committed at or before t:

  activity  (attribute view; what a conventional monitoring system has)
      counts / log-amounts in and out over 24 h and 7 d, distinct counterparties,
      near-threshold and >= threshold transfers, channel mix, pass-through ratio,
      account type and tenure;
  monitor   (Objective-2 view)
      log(1 + (A_tau^L)_ii) for L = 2..L_max on the decayed adjacency, decayed
      in/out strength, and the streaming per-node monitor score.

Ground-truth labels (used for training and scoring only, never as inputs):
an account-epoch is positive if the account sent or received a fraud
transaction committed in (t - active_h, t]; ``kind`` records the typology.
"""
from __future__ import annotations

import numpy as np

from .monitor import closed_walk_diagonals, decayed_adjacency
from .system import THRESHOLD
from .world import MERCHANT, BUSINESS

ACT_NAMES = ["out_n_24h", "in_n_24h", "out_amt_24h", "in_amt_24h", "out_max_24h",
             "out_cp_24h", "in_cp_24h", "near_thr_24h", "over_thr_24h",
             "transfer_share_24h", "mobile_share_24h", "pass_through_24h",
             "out_n_7d", "in_n_7d", "out_amt_7d", "in_amt_7d", "out_cp_7d", "in_cp_7d",
             "is_merchant", "is_business", "log_tenure"]


def _mon_names(L_max):
    return [f"log_A{L}_ii" for L in range(2, L_max + 1)] + \
           ["log_out_strength", "log_in_strength", "stream_score"]


class SnapshotBuilder:
    def __init__(self, pop, ledger, stream_monitor=None, tau_h=24.0, L_max=6,
                 active_h=24.0):
        self.pop, self.ledger, self.stream = pop, ledger, stream_monitor
        self.tau, self.L_max, self.active_h = tau_h, L_max, active_h
        self.act_names = ACT_NAMES
        self.mon_names = _mon_names(L_max)

    @staticmethod
    def _window_stats(N, src, dst, amt, ch):
        out_n = np.bincount(src, minlength=N).astype(float)
        in_n = np.bincount(dst, minlength=N).astype(float)
        out_a = np.bincount(src, weights=amt, minlength=N)
        in_a = np.bincount(dst, weights=amt, minlength=N)
        out_max = np.zeros(N); np.maximum.at(out_max, src, amt)
        pairs = np.unique(np.stack([src, dst], 1), axis=0) if len(src) else np.zeros((0, 2), int)
        out_cp = np.bincount(pairs[:, 0], minlength=N).astype(float)
        in_cp = np.bincount(pairs[:, 1], minlength=N).astype(float)
        near = np.bincount(src, weights=((amt >= 0.9 * THRESHOLD) & (amt < THRESHOLD)).astype(float), minlength=N)
        over = np.bincount(src, weights=(amt >= THRESHOLD).astype(float), minlength=N)
        tr = np.bincount(src, weights=(ch == 1).astype(float), minlength=N)
        mo = np.bincount(src, weights=(ch == 2).astype(float), minlength=N)
        return out_n, in_n, out_a, in_a, out_max, out_cp, in_cp, near, over, tr, mo

    def snapshot(self, t: float) -> dict:
        a = self.ledger.arrays()
        N = self.pop.N
        tc = a["t_commit"]
        hi = np.searchsorted(tc, t, side="right")
        lo24 = np.searchsorted(tc, t - 24.0, side="right")
        lo7 = np.searchsorted(tc, t - 168.0, side="right")
        loA = np.searchsorted(tc, t - self.active_h, side="right")
        s24 = slice(lo24, hi); s7 = slice(lo7, hi)

        w24 = self._window_stats(N, a["src"][s24], a["dst"][s24], a["amount"][s24], a["channel"][s24])
        w7 = self._window_stats(N, a["src"][s7], a["dst"][s7], a["amount"][s7], a["channel"][s7])
        out_n, in_n, out_a, in_a, out_max, out_cp, in_cp, near, over, tr, mo = w24
        l = np.log1p
        pass_thr = np.minimum(out_a, in_a) / (np.maximum(out_a, in_a) + 1e-9)
        act = np.vstack([
            out_n, in_n, l(out_a), l(in_a), l(out_max), out_cp, in_cp, near, over,
            tr / np.maximum(out_n, 1), mo / np.maximum(out_n, 1), pass_thr,
            w7[0], w7[1], l(w7[2]), l(w7[3]), w7[5], w7[6],
            (self.pop.kind == MERCHANT).astype(float), (self.pop.kind == BUSINESS).astype(float),
            l(self.pop.tenure),
        ]).T

        A = decayed_adjacency(a["src"][:hi], a["dst"][:hi], tc[:hi], t, N, self.tau)
        diag = closed_walk_diagonals(A, self.L_max)
        stream = self.stream.node_scores_at(t) if self.stream is not None else np.zeros(N)
        mon = np.hstack([l(diag), l(np.asarray(A.sum(1)).ravel())[:, None],
                         l(np.asarray(A.sum(0)).ravel())[:, None], l(stream)[:, None]])

        sl = slice(loA, hi)
        active = np.zeros(N, bool)
        active[a["src"][sl]] = True; active[a["dst"][sl]] = True
        nodes = np.where(active)[0]
        # labels (evaluation only)
        fr = a["fraud"][sl]
        y = np.zeros(N, np.int64); kind = np.full(N, -1)
        fs, fd, fk = a["src"][sl][fr], a["dst"][sl][fr], a["kind"][sl][fr]
        y[fs] = 1; y[fd] = 1
        kind[fs] = fk; kind[fd] = fk
        # recent-transaction graph (7 d, undirected, unique) for GNN baselines
        pr = np.unique(np.stack([a["src"][s7], a["dst"][s7]], 0), axis=1)
        ei = np.hstack([pr, pr[::-1]])
        return {"t": t, "nodes": nodes, "act": act[nodes], "mon": mon[nodes],
                "act_all": act, "mon_all": mon, "ei": ei,
                "y": y[nodes], "kind": kind[nodes],
                "cycle_intensity": {L: float(diag[:, L - 2].sum()) for L in range(2, self.L_max + 1)}}
