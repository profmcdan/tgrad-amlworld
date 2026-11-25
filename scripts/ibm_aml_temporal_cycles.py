"""Time-respecting cycle monitor on IBM AML (improvement 2), streamed over the ledger.

    python scripts/ibm_aml_temporal_cycles.py --name HI-Small --window 72

When a transfer a->b commits at time t, it closes a temporal cycle if there is a walk
b -> ... -> a whose transfers are non-decreasing in time, all inside [t - W, t], with at
most L_max - 1 hops (so the cycle including a->b has length <= L_max). The search is
bidirectional and layered: forward from b by earliest arrival, backward from a by latest
departure, per hop count, which decides existence exactly. For every node where the two
searches meet, the shortest such walk through it is rebuilt from parent pointers and all
of its accounts are credited. An account's score at a snapshot is the number of closing
transfers (up to that time) whose credited walk of length <= cap contains it. This is
the cumulative graph, as for the binary monitor.

Selection protocol: W in {6, 24, 72, 168} h and cap in {6, 8, 10, 12} are chosen by
validation-window ROC-AUC of CYCLE accounts against licit accounts (ties go to the
smaller W, then the smaller cap). Thresholds (best F1, 1% FPR) are set on the training
snapshot. The test window is not used for any choice.
"""
import argparse
import json
import math
import os
import sys
import time
from array import array
from bisect import bisect_left, bisect_right

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ibm_aml_common as C  # noqa: E402

CAPS = (6, 8, 10, 12)
WINDOWS = (6.0, 24.0, 72.0, 168.0)


class TemporalCycleMonitor:
    def __init__(self, N, window_h, L_max=12, budget=200_000, max_meet=64, caps=CAPS):
        self.N, self.W, self.M = N, window_h, L_max - 1
        self.budget, self.max_meet, self.caps = budget, max_meet, caps
        self.ot, self.on, self.it, self.inb = {}, {}, {}, {}
        self.counts = {c: np.zeros(N, np.int64) for c in caps}
        self.n_closing = {c: 0 for c in caps}
        self.n_truncated = 0
        self.work = []

    def _add(self, a, b, t):
        if a not in self.ot:
            self.ot[a], self.on[a] = array("d"), array("l")
        self.ot[a].append(t); self.on[a].append(b)
        if b not in self.it:
            self.it[b], self.inb[b] = array("d"), array("l")
        self.it[b].append(t); self.inb[b].append(a)

    def search(self, a, b, t):
        """Credited node sets per cap for the transfer a->b at time t (before inserting it)."""
        lo = t - self.W
        F = [{b: (lo, -1)}]; B = [{a: (t, -1)}]
        fbest = {b: lo}; bbest = {a: t}
        work = 0; trunc = False
        while (len(F) - 1) + (len(B) - 1) < self.M:
            fwd = len(F[-1]) <= len(B[-1]) if F[-1] and B[-1] else bool(F[-1])
            if not F[-1] and not B[-1]:
                break
            new = {}
            if fwd:
                for n, (r, _) in F[-1].items():
                    ts = self.ot.get(n)
                    if ts is None:
                        continue
                    nb = self.on[n]
                    for k in range(bisect_left(ts, r), len(ts)):
                        tm = ts[k]; m = nb[k]; work += 1
                        if tm < fbest.get(m, math.inf):
                            fbest[m] = tm; new[m] = (tm, n)
                F.append(new)
            else:
                for n, (d, _) in B[-1].items():
                    ts = self.it.get(n)
                    if ts is None:
                        continue
                    nb = self.inb[n]
                    for k in range(bisect_left(ts, lo), bisect_right(ts, d)):
                        tm = ts[k]; m = nb[k]; work += 1
                        if tm > bbest.get(m, -math.inf):
                            bbest[m] = tm; new[m] = (tm, n)
                B.append(new)
            if work > self.budget:
                trunc = True
                break
        self.work.append(work)
        if trunc:
            self.n_truncated += 1
        # meeting nodes: forward arrival <= backward departure, shortest hop total
        meet = []
        for n in fbest.keys() & bbest.keys():
            if fbest[n] > bbest[n]:
                continue
            fl = [(h, L[n][0]) for h, L in enumerate(F) if n in L]
            bl = [(h, L[n][0]) for h, L in enumerate(B) if n in L]
            best = None
            for h1, tf in fl:
                for h2, tb in bl:
                    if tf <= tb and h1 + h2 >= 1 and (best is None or h1 + h2 < best[0] + best[1]):
                        best = (h1, h2)
            if best is not None:
                meet.append((best[0] + best[1], best[0], best[1], n))
        if not meet:
            return None
        meet.sort()
        cred = {c: set() for c in self.caps}
        for hops, h1, h2, n in meet[: self.max_meet]:
            L = hops + 1
            nodes = {a, b, n}
            x, h = n, h1
            while h > 0:
                x = F[h][x][1]; h -= 1; nodes.add(x)
            x, h = n, h2
            while h > 0:
                x = B[h][x][1]; h -= 1; nodes.add(x)
            for c in self.caps:
                if L <= c:
                    cred[c] |= nodes
        return cred

    def on_commit(self, a, b, t):
        if a != b:
            cred = self.search(a, b, t)
            if cred:
                for c, s in cred.items():
                    if s:
                        self.counts[c][list(s)] += 1; self.n_closing[c] += 1
            self._add(a, b, t)


def run_stream(z, window_h, budget):
    T1, T2 = C.cuts(z)
    Tend = int(z["t"][-1])
    N = len(z["accounts"])
    mon = TemporalCycleMonitor(N, window_h, budget=budget)
    src, dst, tm = z["src"].tolist(), z["dst"].tolist(), z["t"].tolist()
    th = [x / 60.0 for x in tm]
    snaps, k = {}, 0
    order = [("train", T1), ("val", T2)]
    t0 = time.time()
    for i in range(len(tm)):
        while k < len(order) and tm[i] >= order[k][1]:
            snaps[order[k][0]] = {c: v.copy() for c, v in mon.counts.items()}; k += 1
        mon.on_commit(src[i], dst[i], th[i])
        if i % 500_000 == 0 and i:
            print(f"  W={window_h:g}h {i}/{len(tm)} {time.time() - t0:.0f}s trunc={mon.n_truncated}", flush=True)
    snaps["test"] = {c: v.copy() for c, v in mon.counts.items()}
    w = np.array(mon.work)
    stats = {"stream_sec": time.time() - t0, "n_truncated": mon.n_truncated,
             "share_truncated": mon.n_truncated / len(tm), "work_mean": float(w.mean()),
             "work_p99": float(np.quantile(w, 0.99)), "n_closing_tx": mon.n_closing}
    return snaps, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="HI-Small")
    ap.add_argument("--window", type=float, required=True)
    ap.add_argument("--budget", type=int, default=200_000)
    ap.add_argument("--max-tx", type=int, default=0, help="timing runs only")
    a = ap.parse_args()
    z = C.load(a.name)
    if a.max_tx:
        z = {k: (v[: a.max_tx] if k in ("src", "dst", "t") else v) for k, v in z.items()}
        z["t"] = z["t"]
    snaps, stats = run_stream(z, a.window, a.budget)
    print(json.dumps(stats), flush=True)
    if a.max_tx:
        return
    out = os.path.join(C.ROOT, "data", "ibm_aml_proc", f"{a.name}_tcycle_W{a.window:g}.npz")
    np.savez_compressed(out, **{f"{sp}_L{c}": v for sp, d in snaps.items() for c, v in d.items()})
    json.dump(stats, open(os.path.join(C.ROOT, "results", a.name, f"tcycle_W{a.window:g}_stats.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
