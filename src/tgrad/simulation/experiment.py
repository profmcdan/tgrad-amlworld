"""
Objective 6 experiment protocol: train on simulated history, deploy online in
the DES, and evaluate WITH fraud (S1) and on the matched WITHOUT-fraud twin (S0).

Timeline of one run (hours):

  0 ........ burn-in ........ T_fit ........ T_train | ........ deployment ........ T_end
  |<-- history: stream monitor on, snapshots -->|     |<- detectors score every epoch ->|
                 fit rows          calib rows   train  (online, causal, alerts raised)

  * History [0, T_train]: the simulation runs with the label-free streaming
    monitor attached; account snapshots are taken every epoch.
  * Training (the simulation is paused at T_train): the neural T-GRAD is trained
    on the historical ledger; heads are fitted on snapshots in [burn-in, T_fit]
    and alert thresholds are calibrated on the benign rows of the last
    ``calib_frac`` of the history (target FPR), never on the deployment period.
  * Deployment (T_train, T_end]: the simulation resumes; at each epoch every
    detector scores the currently active accounts using only data committed so
    far and raises alerts.
  * Twin S0: the same world (identical benign traffic, common random numbers)
    with fraud disabled, scored by the SAME deployed detectors and thresholds.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .detectors import (HeadModel, SAGEDetector, TGradNeural, monitor_score, rule_score,
                        walk_weights)
from .engine import Server, Simulator
from .features import SnapshotBuilder
from .fraud import FraudConfig
from .monitor import StreamingCycleMonitor
from .system import PaymentSystem
from .world import KIND, WorldConfig

TYPOLOGIES = {"cycle": KIND["cycle"], "fanout": KIND["fanout"], "ato": KIND["ato"]}

MODELS = ["Rules", "Attribute MLP", "GraphSAGE", "T-GRAD monitor (untrained)",
          "T-GRAD monitor L>=3 (untrained)",
          "T-GRAD w/o monitor", "T-GRAD w/o diffusion", "T-GRAD (full)",
          "GraphSAGE + monitor"]          # hybrid: GraphSAGE given the closed-walk block
# optional GNN baselines (ExperimentConfig.extra_gnns), each with and without the closed-walk block
EXTRA_GNNS = {"GCN": "gcn", "GAT": "gat"}
ALL_MODELS = MODELS + [n for g in EXTRA_GNNS for n in (g, g + " + monitor")]
STREAM_VARIANTS = {"thesis weights, L=2..6": 2, "L=3..6": 3}


@dataclass
class ExperimentConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    fraud: FraudConfig = field(default_factory=FraudConfig)
    train_days: float = 14.0
    burn_in_days: float = 2.0
    calib_frac: float = 0.25
    epoch_h: float = 6.0
    tau_h: float = 24.0
    L_max: int = 6
    target_fpr: float = 0.01
    tx_alert_q: float = 0.999
    enc_epochs: int = 4
    head_epochs: int = 300
    sage_epochs: int = 40
    max_delay_h: float = 48.0
    extra_gnns: tuple = ()
    run_twin: bool = True
    seed: int = 0


def _features(name, s):
    if name == "Attribute MLP":
        return s["act"]
    if name == "T-GRAD w/o monitor":           # [a || z || rho || E]
        return np.hstack([s["act"], s["neu"]])
    if name == "T-GRAD w/o diffusion":         # [a || m]  (no encoder / diffusion / energy)
        return np.hstack([s["act"], s["mon"]])
    if name == "T-GRAD (full)":                # [a || z || m || rho || E]
        return np.hstack([s["act"], s["neu"], s["mon"]])
    raise KeyError(name)


HEADS = ["Attribute MLP", "T-GRAD w/o monitor", "T-GRAD w/o diffusion", "T-GRAD (full)"]


class Deployed:
    """The trained detector bundle; scores one snapshot."""

    def __init__(self, heads, sage, neu, L_max, sage_mon=None, extra=None):
        self.heads, self.sage, self.neu, self.L_max = heads, sage, neu, L_max
        self.sage_mon, self.extra = sage_mon, extra or {}

    def score(self, s):
        """Scores for every model + model inference cost (s). Feature costs
        (snapshot incl. monitor, and the neural stream) are recorded separately."""
        out, cost = {}, {}

        def timed(name, fn):
            t0 = time.perf_counter(); out[name] = fn(); cost[name] = time.perf_counter() - t0
        timed("Rules", lambda: rule_score(s["act"]))
        timed("T-GRAD monitor (untrained)", lambda: monitor_score(s["mon"], self.L_max))
        timed("T-GRAD monitor L>=3 (untrained)", lambda: monitor_score(s["mon"], self.L_max, 3))
        timed("GraphSAGE", lambda: self.sage.score(s["act_all"], s["ei"], s["nodes"]))
        if self.sage_mon is not None:
            timed("GraphSAGE + monitor", lambda: self.sage_mon.score(
                np.hstack([s["act_all"], s["mon_all"]]), s["ei"], s["nodes"]))
        for h in HEADS:
            timed(h, lambda h=h: self.heads[h].score(_features(h, s)))
        for name, (det, with_mon) in self.extra.items():
            X = np.hstack([s["act_all"], s["mon_all"]]) if with_mon else s["act_all"]
            timed(name, lambda det=det, X=X: det.score(X, s["ei"], s["nodes"]))
        return out, cost


def _snap(sb, t, n_ledger):
    t0 = time.perf_counter()
    s = sb.snapshot(t)
    s["cost_snapshot"] = time.perf_counter() - t0
    s["n_ledger"] = n_ledger
    return s


def _add_neural(s, neu, a):
    t0 = time.perf_counter()
    neu.advance(a, s["n_ledger"])
    s["neu"] = neu.features(s["nodes"])
    s["cost_neural"] = time.perf_counter() - t0


def _deploy(ps, sb, deployed, record_from, log):
    """Attach the online scoring hook to a payment system."""
    recs, intensity = [], []

    def hook(t):
        a_len = len(ps.ledger)
        s = _snap(sb, t, a_len)
        intensity.append((t, s["cycle_intensity"]))
        if t <= record_from + 1e-9:
            return
        _add_neural(s, deployed.neu, ps.ledger.arrays())
        scores, cost = deployed.score(s)
        recs.append({"t": t, "nodes": s["nodes"], "y": s["y"], "kind": s["kind"],
                     "scores": scores, "cost": cost, "cost_snapshot": s["cost_snapshot"],
                     "cost_neural": s["cost_neural"]})
        if log and len(recs) % 8 == 0:
            print(f"    t={t / 24:5.1f} d  active={len(s['nodes'])}  positives={int(s['y'].sum())}")
    ps.epoch_hooks.append(hook)
    return recs, intensity


def _instances(fraud, a):
    """Fraud instances with first/last commit time from the ledger."""
    out = []
    inst, tc = a["inst"], a["t_commit"]
    for iid, d in fraud.instances.items():
        idx = np.where(inst == iid)[0]
        if len(idx) == 0:
            continue
        out.append({"id": iid, "kind": d["kind"], "accounts": d["accounts"],
                    "t_first": float(tc[idx].min()), "t_last": float(tc[idx].max()),
                    "tx": idx})
    return out


# ================================================================== evaluation
def _pool(recs, name):
    y = np.concatenate([r["y"] for r in recs])
    k = np.concatenate([r["kind"] for r in recs])
    s = np.concatenate([r["scores"][name] for r in recs])
    return y, k, s


def evaluate_rows(recs, thr):
    res = {}
    for m in thr:
        y, k, s = _pool(recs, m)
        yp = s >= thr[m]
        tp = int((yp & (y == 1)).sum()); fp = int((yp & (y == 0)).sum())
        fn = int((~yp & (y == 1)).sum()); tn = int((~yp & (y == 0)).sum())
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        r = {"ROC-AUC": float(roc_auc_score(y, s)), "PR-AUC": float(average_precision_score(y, s)),
             "Precision": prec, "Recall": rec, "F1": 2 * prec * rec / max(1e-12, prec + rec),
             "FPR": fp / max(1, fp + tn), "prevalence": float(y.mean()), "rows": int(len(y))}
        for tn_, kv in TYPOLOGIES.items():
            sel = (y == 0) | (k == kv)
            if (k == kv).sum() > 0:
                r[f"AUC[{tn_}]"] = float(roc_auc_score(k[sel] == kv, s[sel]))
                r[f"Recall[{tn_}]"] = float(yp[k == kv].mean())
        res[m] = r
    return res


def evaluate_instances(recs, thr, insts, t_from, t_end, max_delay):
    """Instance-level detection rate and delay (epoch alerts)."""
    res = {}
    elig = [d for d in insts if d["t_first"] > t_from and d["t_last"] + max_delay <= t_end]
    for m in thr:
        alerts = [(r["t"], set(r["nodes"][r["scores"][m] >= thr[m]].tolist())) for r in recs]
        per = {}
        for d in elig:
            acc = set(d["accounts"]); det = None
            for t, al in alerts:
                if t < d["t_first"] or t > d["t_last"] + max_delay:
                    continue
                if acc & al:
                    det = t; break
            per.setdefault(d["kind"], []).append(None if det is None else det - d["t_last"])
        res[m] = {k: {"n": len(v), "detected": float(np.mean([x is not None for x in v])),
                      "median_delay_h": float(np.median([x for x in v if x is not None]))
                      if any(x is not None for x in v) else float("nan")}
                  for k, v in per.items()}
    return res


def stream_scores(mon, L_max, L_min):
    return np.vstack(mon.tx_walks) @ walk_weights(L_max, L_min)


def evaluate_streaming(s, a, insts, tx_thr, t_from, t_end, max_delay):
    """Per-transaction alerting by the streaming monitor (zero batching delay).

    ``s`` are the per-transaction scores in commit order; an instance is
    detected when any of ITS OWN transactions scores >= tx_thr."""
    tc = a["t_commit"]
    test = tc > t_from
    fr = a["fraud"]
    per = {}
    for d in insts:
        if not (d["t_first"] > t_from and d["t_last"] + max_delay <= t_end):
            continue
        hit = d["tx"][s[d["tx"]] >= tx_thr]
        per.setdefault(d["kind"], []).append(None if len(hit) == 0 else float(tc[hit].min() - d["t_last"]))
    out = {k: {"n": len(v), "detected": float(np.mean([x is not None for x in v])),
               "median_delay_h": float(np.median([x for x in v if x is not None]))
               if any(x is not None for x in v) else float("nan")} for k, v in per.items()}
    for tn_, kv in TYPOLOGIES.items():
        sel = test & ((~fr) | (a["kind"] == kv))
        if (a["kind"][sel] == kv).any():
            out[f"tx_AUC[{tn_}]"] = float(roc_auc_score(a["kind"][sel] == kv, s[sel]))
    out["benign_tx_alert_rate"] = float((s[test & ~fr] >= tx_thr).mean())
    out["benign_alerts_per_day"] = float((s[test & ~fr] >= tx_thr).sum() / ((t_end - t_from) / 24))
    return out


def load_test(cost_s, rates_tps, servers=4, base_s=0.4, n_jobs=20000, seed=0):
    """M/G/c switch with the monitor inline: sojourn percentiles vs arrival rate.

    Service time = base switch time (exponential, mean ``base_s``) + an inline
    scoring cost resampled from the MEASURED per-transaction costs ``cost_s``.
    """
    rng = np.random.default_rng(seed)
    cost_s = np.asarray(cost_s)
    out = []
    for lam in rates_tps:
        sim = Simulator()
        srv = Server(sim, servers, lambda j: j.svc, lambda j: None)

        class J:
            __slots__ = ("svc", "arrive", "start", "finish")
        t = 0.0
        for _ in range(n_jobs):
            t += rng.exponential(1.0 / lam)
            j = J(); j.svc = (rng.exponential(base_s) + rng.choice(cost_s)) / 3600.0
            sim.at(t / 3600.0, srv.submit, j)
        sim.run(t / 3600.0 + 10.0)
        st = srv.stats(t / 3600.0)
        out.append({"rate_tps": lam, "p50_ms": st["p50_sojourn_h"] * 3.6e6,
                    "p99_ms": st["p99_sojourn_h"] * 3.6e6, "utilisation": st["utilisation"]})
    return out


# ================================================================== driver
def run_experiment(cfg: ExperimentConfig, verbose: bool = True) -> dict:
    T_train = cfg.train_days * 24.0
    T_end = cfg.world.days * 24.0
    wall = {}
    log = print if verbose else (lambda *a, **k: None)

    # ---------------- history (S1) ----------------
    t0 = time.perf_counter()
    ps = PaymentSystem(cfg.world, cfg.fraud, epoch_h=cfg.epoch_h)
    mon = StreamingCycleMonitor(ps.pop.N, tau_h=cfg.tau_h, L_max=cfg.L_max)
    ps.commit_hooks.append(mon.on_commit)
    sb = SnapshotBuilder(ps.pop, ps.ledger, mon, cfg.tau_h, cfg.L_max)
    hist, hist_int = [], []

    def hookA(t):
        s = _snap(sb, t, len(ps.ledger))
        hist.append(s); hist_int.append((t, s["cycle_intensity"]))
    ps.epoch_hooks.append(hookA)
    ps.run_until(T_train)
    ps.epoch_hooks.clear()
    wall["history_s"] = time.perf_counter() - t0
    a = ps.ledger.arrays(); n_train = len(a["src"]); N = ps.pop.N
    log(f"  history: {n_train} tx, {int(a['fraud'].sum())} fraud tx, {len(hist)} epochs "
        f"({wall['history_s']:.1f}s)")

    # ---------------- training ----------------
    t0 = time.perf_counter()
    node_label = np.zeros(N, np.int64)
    f = a["fraud"]; node_label[a["src"][f]] = 1; node_label[a["dst"][f]] = 1
    neu = TGradNeural(N, seed=cfg.seed).fit(a, n_train, node_label, epochs=cfg.enc_epochs)
    for s in hist:
        _add_neural(s, neu, a)
    wall["neural_train_s"] = time.perf_counter() - t0
    usable = [s for s in hist if s["t"] >= cfg.burn_in_days * 24.0]
    k = int(round(len(usable) * (1 - cfg.calib_frac)))
    fit_s, cal_s = usable[:k], usable[k:]
    yfit = np.concatenate([s["y"] for s in fit_s])
    t0 = time.perf_counter()
    heads = {h: HeadModel(cfg.head_epochs, cfg.seed).fit(np.vstack([_features(h, s) for s in fit_s]), yfit)
             for h in HEADS}
    sage = SAGEDetector(epochs=cfg.sage_epochs, seed=cfg.seed).fit(
        [(s["act_all"], s["ei"], s["nodes"], s["y"]) for s in fit_s])
    sage_mon = SAGEDetector(epochs=cfg.sage_epochs, seed=cfg.seed).fit(
        [(np.hstack([s["act_all"], s["mon_all"]]), s["ei"], s["nodes"], s["y"]) for s in fit_s])
    extra = {}
    for g in cfg.extra_gnns:
        for with_mon in (False, True):
            X = (lambda s: np.hstack([s["act_all"], s["mon_all"]])) if with_mon else (lambda s: s["act_all"])
            det = SAGEDetector(epochs=cfg.sage_epochs, seed=cfg.seed, conv=EXTRA_GNNS[g]).fit(
                [(X(s), s["ei"], s["nodes"], s["y"]) for s in fit_s])
            extra[g + " + monitor" if with_mon else g] = (det, with_mon)
    wall["heads_train_s"] = time.perf_counter() - t0
    deployed = Deployed(heads, sage, neu, cfg.L_max, sage_mon, extra)
    # thresholds: (1 - target_fpr) quantile of calibration-period benign scores
    cal_scores = [deployed.score(s)[0] for s in cal_s]
    thr = {}
    for m in cal_scores[0]:
        sc = np.concatenate([cs[m][s["y"] == 0] for cs, s in zip(cal_scores, cal_s)])
        thr[m] = float(np.quantile(sc, 1 - cfg.target_fpr))
    # streaming thresholds: tx_alert_q quantile of KNOWN-BENIGN historical tx
    hist_benign = ~a["fraud"][:n_train]
    tx_thr = {v: float(np.quantile(stream_scores(mon, cfg.L_max, lm)[:n_train][hist_benign],
                                   cfg.tx_alert_q)) for v, lm in STREAM_VARIANTS.items()}
    log(f"  trained: fit rows={len(yfit)} (pos {int(yfit.sum())}), "
        f"encoder loss {neu.loss_hist[0]:.3f}->{neu.loss_hist[-1]:.3f} "
        f"({wall['neural_train_s'] + wall['heads_train_s']:.1f}s)")

    # ---------------- deployment (S1) ----------------
    t0 = time.perf_counter()
    recs1, int1 = _deploy(ps, sb, deployed, T_train, verbose)
    ps.run_until(T_end)
    wall["deploy_s"] = time.perf_counter() - t0
    a = ps.ledger.arrays()
    insts = _instances(ps.fraud, a)
    res = {
        "config": {"world": vars(cfg.world), "fraud": vars(cfg.fraud),
                   **{k_: v for k_, v in vars(cfg).items() if k_ not in ("world", "fraud")}},
        "thresholds": thr, "tx_threshold": tx_thr,
        "S1_rows": evaluate_rows(recs1, thr),
        "S1_instances": evaluate_instances(recs1, thr, insts, T_train, T_end, cfg.max_delay_h),
        "n_instances": {k_: sum(1 for d in insts if d["kind"] == k_ and d["t_first"] > T_train)
                        for k_ in TYPOLOGIES},
        "switch_S1": ps.switch.stats(T_end),
        "n_tx_S1": int(len(a["src"])), "n_fraud_tx_S1": int(a["fraud"].sum()),
    }
    tx_scores = {v: stream_scores(mon, cfg.L_max, lm) for v, lm in STREAM_VARIANTS.items()}
    res["S1_streaming"] = {v: evaluate_streaming(tx_scores[v], a, insts, tx_thr[v], T_train, T_end,
                                                 cfg.max_delay_h) for v in STREAM_VARIANTS}
    res["monitor_cost_us"] = {"mean": float(np.mean(mon.cost_s) * 1e6),
                              "p50": float(np.percentile(mon.cost_s, 50) * 1e6),
                              "p99": float(np.percentile(mon.cost_s, 99) * 1e6),
                              "max": float(np.max(mon.cost_s) * 1e6)}
    res["epoch_cost_ms"] = {m: float(np.mean([r["cost"][m] for r in recs1]) * 1e3) for m in thr}
    res["epoch_cost_ms"]["features: activity + monitor snapshot"] = float(
        np.mean([r["cost_snapshot"] for r in recs1]) * 1e3)
    res["epoch_cost_ms"]["features: neural stream + guided recon"] = float(
        np.mean([r["cost_neural"] for r in recs1]) * 1e3)
    log(f"  deployed S1 ({wall['deploy_s']:.1f}s)")

    # ---------------- twin without fraud (S0) ----------------
    int0 = None
    if cfg.run_twin:
        t0 = time.perf_counter()
        ps0 = PaymentSystem(cfg.world, replace(cfg.fraud, enabled=False), epoch_h=cfg.epoch_h)
        mon0 = StreamingCycleMonitor(ps0.pop.N, tau_h=cfg.tau_h, L_max=cfg.L_max)
        ps0.commit_hooks.append(mon0.on_commit)
        sb0 = SnapshotBuilder(ps0.pop, ps0.ledger, mon0, cfg.tau_h, cfg.L_max)
        neu.reset_stream()
        recs0, int0 = _deploy(ps0, sb0, deployed, T_train, False)
        ps0.run_until(T_end)
        a0 = ps0.ledger.arrays()
        days = (T_end - T_train) / 24.0
        s0 = {}
        for m in thr:
            _, _, sc = _pool(recs0, m)
            s0[m] = {"FPR": float((sc >= thr[m]).mean()),
                     "alerts_per_day": float((sc >= thr[m]).sum() / days),
                     "rows": int(len(sc))}
        te0 = a0["t_commit"] > T_train
        s0["streaming"] = {}
        for v, lm in STREAM_VARIANTS.items():
            tx0 = stream_scores(mon0, cfg.L_max, lm)
            s0["streaming"][v] = {"tx_alert_rate": float((tx0[te0] >= tx_thr[v]).mean()),
                                  "alerts_per_day": float((tx0[te0] >= tx_thr[v]).sum() / days)}
        res["S0"] = s0
        # benign rows of S1 for the with/without comparison
        res["S1_benign_alerts_per_day"] = {
            m: float(sum(((r["scores"][m] >= thr[m]) & (r["y"] == 0)).sum() for r in recs1) / days)
            for m in thr}
        res["n_tx_S0"] = int(len(a0["src"]))
        wall["twin_s"] = time.perf_counter() - t0
        log(f"  twin S0 ({wall['twin_s']:.1f}s)")
    res["wall_s"] = wall

    # series for figures (kept out of the JSON summary by the driver)
    series = {"intensity_S1": hist_int + int1, "intensity_S0": int0,
              "insts": insts,
              "recs1": recs1, "tx_scores": tx_scores, "tx_kind": a["kind"],
              "tx_t": a["t_commit"], "monitor_cost_s": np.asarray(mon.cost_s),
              "encoder_loss": neu.loss_hist, "thr": thr, "tx_thr": tx_thr,
              "recs0": recs0 if cfg.run_twin else None}
    return res, series
