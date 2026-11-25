"""
Objective-2 cyclic monitor, deployed inside the discrete-event simulator.

Two readouts of the same statistic, both on the continuous-time, exponentially
decayed adjacency

    A_tau(t)[u, v] = sum_{e: u->v, t_e <= t} exp(-(t - t_e) / tau)      (1)

so an edge's weight fades with age; a directed cycle whose edges all occurred
within ~tau of each other keeps a large product of weights, whereas a slow
economic cycle (salary -> spend -> supplier, spread over weeks) is suppressed.

(a) STREAMING, per transaction (event-driven, at commit time).  When u->v
    commits at time t, the closed walks it closes are the walks v ~> u:

        s(u->v, t) = sum_{m=1}^{L_max-1} beta^{m+1} [A_tau(t)^m]_{v u}        (2)

    (beta = 0.5 reproduces the thesis weights w_L = 0.5^L, L = m + 1.)

    computed exactly by a bidirectional sparse frontier search (forward from v,
    backward from u, meet in the middle). The score is available the instant
    the closing transaction commits, so a ring can be alerted on with zero
    batching delay.

(b) BATCH, per scoring epoch.  Per-node closed-walk counts (A_tau^L)_ii for
    L = 2..L_max, via the exact identity

        diag(A^L) = rowsum( A^a  ∘  (A^b)^T ),   a + b = L,               (3)

    which needs only A^2 and A^3 for L <= 6. These are the monitor features
    consumed by the T-GRAD detector head (Objective 4 augmented state).
"""
from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np
import scipy.sparse as sp


class StreamingCycleMonitor:
    """Per-transaction continuous-time closed-walk score, eq. (2)."""

    def __init__(self, num_nodes: int, tau_h: float = 24.0, L_max: int = 6,
                 beta: float = 0.5, prune: float = 1e-4, horizon_factor: float = 5.0):
        self.N = num_nodes
        self.tau = tau_h
        self.L_max = L_max
        self.beta = beta
        self.prune = prune
        self.keep_h = horizon_factor * tau_h
        # weights stored relative to a base time T0: w(t) = S * exp(-(t - T0)/tau)
        self.T0 = 0.0
        self.out = defaultdict(dict)
        self.inn = defaultdict(dict)
        # per-node running score (decayed max of incident transaction scores)
        self.node_score = np.zeros(num_nodes)
        self.node_score_t = np.zeros(num_nodes)
        self.tx_scores = []           # (tx_id, score) in commit order
        self.tx_walks = []            # per-tx closed walks by cycle length L = 2..L_max
        self.cost_s = []              # wall-clock cost per transaction

    # ------------------------------------------------------------ maintenance
    def _rebase(self, t):
        f = math.exp(-(t - self.T0) / self.tau)
        thr = self.prune
        for adj in (self.out, self.inn):
            dead = []
            for u, nb in adj.items():
                for v in list(nb):
                    nb[v] *= f
                    if nb[v] < thr:
                        del nb[v]
                if not nb:
                    dead.append(u)
            for u in dead:
                del adj[u]
        self.T0 = t

    def _g(self, t):
        return math.exp(-(t - self.T0) / self.tau)

    @staticmethod
    def _step(front, adj, g, cap=1e-8):
        nxt = {}
        for i, fi in front.items():
            nb = adj.get(i)
            if not nb:
                continue
            for j, w in nb.items():
                val = fi * w * g
                if val > cap:
                    nxt[j] = nxt.get(j, 0.0) + val
        return nxt

    def walks_between(self, v, u, t):
        """[A_tau(t)^m]_{v,u} for m = 1..L_max-1 (bidirectional search)."""
        g = self._g(t)
        M = self.L_max - 1
        fwd = [{v: 1.0}]
        bwd = [{u: 1.0}]
        out = np.zeros(M + 1)
        # grow the cheaper frontier until depths sum to M
        while len(fwd) - 1 + len(bwd) - 1 < M:
            if len(fwd[-1]) <= len(bwd[-1]):
                fwd.append(self._step(fwd[-1], self.out, g))
            else:
                bwd.append(self._step(bwd[-1], self.inn, g))
        # e_v^T A^a A^b e_u = [A^m]_{vu} for any split a + b = m; use one split
        for m in range(1, M + 1):
            a = min(m, len(fwd) - 1)
            fa, gb = fwd[a], bwd[m - a]
            if not fa or not gb:
                continue
            small, big = (fa, gb) if len(fa) <= len(gb) else (gb, fa)
            out[m] = sum(x * big.get(k, 0.0) for k, x in small.items())
        return out[1:]

    # ------------------------------------------------------------ commit hook
    def on_commit(self, tx):
        t0 = time.perf_counter()
        t = tx.finish
        if t - self.T0 > self.tau:
            self._rebase(t)
        u, v = tx.src, tx.dst
        if u != v:
            w = self.walks_between(v, u, t)          # walks v ~> u, length 1..L_max-1
            s = float(sum(self.beta ** (m + 2) * w[m] for m in range(len(w))))
        else:
            w = np.zeros(self.L_max - 1)
            s = 0.0
        self.tx_walks.append(w)
        # insert the edge u->v with weight 1 at time t  (S = exp((t - T0)/tau))
        S = math.exp((t - self.T0) / self.tau)
        self.out[u][v] = self.out[u].get(v, 0.0) + S
        self.inn[v][u] = self.inn[v].get(u, 0.0) + S
        # per-node decayed max
        for n in (u, v):
            dec = math.exp(-(t - self.node_score_t[n]) / self.tau)
            self.node_score[n] = max(self.node_score[n] * dec, s)
            self.node_score_t[n] = t
        self.tx_scores.append((tx.id, s))
        self.cost_s.append(time.perf_counter() - t0)
        return s

    def node_scores_at(self, t):
        return self.node_score * np.exp(-(t - self.node_score_t) / self.tau)


def decayed_adjacency(src, dst, t_commit, t, N, tau_h, keep_h=None):
    """A_tau(t) from ledger arrays, eq. (1) (edges older than keep_h dropped)."""
    keep_h = keep_h if keep_h is not None else 5.0 * tau_h
    m = (t_commit <= t) & (t_commit > t - keep_h) & (src != dst)
    w = np.exp(-(t - t_commit[m]) / tau_h)
    return sp.csr_matrix((w, (src[m], dst[m])), shape=(N, N))


def closed_walk_diagonals(A: sp.csr_matrix, L_max: int = 6) -> np.ndarray:
    """(A^L)_ii for L = 2..L_max via eq. (3). Returns array (N, L_max-1)."""
    P = {1: A}
    AT = {1: A.T.tocsr()}
    half = (L_max + 1) // 2
    for k in range(2, half + 1):
        P[k] = (P[k - 1] @ A).tocsr()
        AT[k] = P[k].T.tocsr()
    cols = []
    for L in range(2, L_max + 1):
        a = (L + 1) // 2
        b = L - a
        # diag(A^a A^b) = rowsum(A^a ∘ (A^b)^T)
        cols.append(np.asarray(P[a].multiply(AT[b]).sum(axis=1)).ravel())
    return np.vstack(cols).T
