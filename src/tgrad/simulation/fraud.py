"""
Fraud typologies injected into the Objective-6 discrete-event simulator.

All fraud processes draw from their OWN random stream (separate from the benign
world), so enabling/disabling fraud leaves the benign traffic untouched.

Typologies
----------
  cycle  (target of T-GRAD) -- round-tripping / layering ring: funds leave a
         controller, pass through L-1 mules and return to the controller within a
         short window W, each hop skimming a small fee. A directed cycle with
         bounded temporal ordering (the thesis' Eqs. 17-19).
  fanout (control)          -- smurfing: a source splits funds across k mules
         which forward them to a collector. Acyclic (scatter-gather DAG).
  ato    (control)          -- account takeover: a compromised ordinary
         customer is drained to several unfamiliar payees within ~2 h. Acyclic.

Two regimes: OVERT (default) -- fraud moves larger sums over bank transfer, so
attributes carry signal (cf. the feature-saturated Elliptic regime); and
CAMOUFLAGED (``camouflage=True``) -- every fraud amount and channel is drawn
from the benign P2P law, so what remains is timing and topology (cf. the
structure-only AML benchmark).

The two control typologies are deliberately *outside* the cyclic scope of the
model; they measure whether the detector generalises or merely specialises.
Mules are recruited from ordinary retail accounts and keep their normal
behaviour, so they are attribute-camouflaged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FraudConfig:
    enabled: bool = True
    n_mules: int = 80                  # recruited retail accounts
    cycles_per_day: float = 3.0
    cycle_len: tuple = (3, 6)
    cycle_window_h: tuple = (0.5, 6.0)  # total time for the funds to go round
    cycle_amount_log: tuple = (7.6, 0.6)  # lognormal (mu, sigma) of the principal
    cycle_fee: tuple = (0.01, 0.03)
    fanout_per_day: float = 1.0
    fanout_k: tuple = (4, 8)
    ato_per_day: float = 1.0
    ato_payees: tuple = (3, 6)
    start_h: float = 24.0              # fraud begins after a 1-day quiet period
    camouflage: bool = False           # True: amounts/channels mimic benign P2P
    seed: int = 1


class FraudInjector:
    """Registers fraud processes; keeps a registry of fraud instances.

    ``instances`` maps instance id -> dict(kind, accounts, t_start, n_tx).
    Completion time / detection are filled in by the evaluation from the ledger.
    """

    def __init__(self, sim, pop, emit, cfg: FraudConfig, horizon_h: float):
        self.sim, self.pop, self.emit, self.cfg = sim, pop, emit, cfg
        self.rng = np.random.default_rng(cfg.seed + 10_007)
        self.horizon = horizon_h
        self.instances = {}
        self._next_id = 0
        self.mules = self.rng.choice(pop.retail, cfg.n_mules, replace=False).astype(int)
        mule_set = set(self.mules.tolist())
        self.non_mules = np.array([i for i in pop.retail if int(i) not in mule_set])
        # external "drop" accounts for ATO are mules too (unfamiliar payees)

    def start(self):
        if not self.cfg.enabled:
            return
        c, rng = self.cfg, self.rng
        if c.cycles_per_day > 0:
            self.sim.schedule(c.start_h + rng.exponential(24.0 / c.cycles_per_day), self._cycle)
        if c.fanout_per_day > 0:
            self.sim.schedule(c.start_h + rng.exponential(24.0 / c.fanout_per_day), self._fanout)
        if c.ato_per_day > 0:
            self.sim.schedule(c.start_h + rng.exponential(24.0 / c.ato_per_day), self._ato)

    # camouflaged fraud copies the benign P2P amount law and channel mix
    def _amount(self, mu, sigma):
        if self.cfg.camouflage:
            return self.rng.lognormal(4.3, 1.0)
        return self.rng.lognormal(mu, sigma)

    def _channel(self, default):
        if self.cfg.camouflage:
            return "mobile" if self.rng.random() < 0.6 else "transfer"
        return default

    def _new(self, kind, accounts):
        iid = self._next_id
        self._next_id += 1
        self.instances[iid] = {"kind": kind, "accounts": [int(a) for a in accounts],
                               "t_start": self.sim.now}
        return iid

    # ------------------------------------------------------------------ cycle
    def _cycle(self):
        c, rng = self.cfg, self.rng
        L = int(rng.integers(c.cycle_len[0], c.cycle_len[1] + 1))
        ring = rng.choice(self.mules, L, replace=False)
        iid = self._new("cycle", ring)
        W = rng.uniform(*c.cycle_window_h)
        gaps = rng.dirichlet(np.ones(L)) * W
        amt = self._amount(*c.cycle_amount_log)
        t = 0.0
        for k in range(L):
            t += gaps[k]
            self.sim.schedule(t, self.emit, int(ring[k]), int(ring[(k + 1) % L]), amt,
                              self._channel("transfer"), "cycle", rng, iid)
            amt *= 1.0 - rng.uniform(*c.cycle_fee)
        self.sim.schedule(rng.exponential(24.0 / c.cycles_per_day), self._cycle)

    # ------------------------------------------------------------------ fan-out
    def _fanout(self):
        c, rng = self.cfg, self.rng
        k = int(rng.integers(c.fanout_k[0], c.fanout_k[1] + 1))
        parts = rng.choice(self.mules, k + 2, replace=False)
        src, col, mids = int(parts[0]), int(parts[1]), parts[2:]
        iid = self._new("fanout", parts)
        total = self._amount(8.3, 0.5) * (k if c.camouflage else 1)
        for m in mids:
            t1 = rng.uniform(0.0, 3.0)
            a = total / k * rng.uniform(0.85, 1.15)
            self.sim.schedule(t1, self.emit, src, int(m), a, self._channel("transfer"),
                              "fanout", rng, iid)
            self.sim.schedule(t1 + rng.uniform(1.0, 24.0), self.emit, int(m), col,
                              a * rng.uniform(0.95, 0.99), self._channel("transfer"),
                              "fanout", rng, iid)
        self.sim.schedule(rng.exponential(24.0 / c.fanout_per_day), self._fanout)

    # ------------------------------------------------------------------ ATO
    def _ato(self):
        c, rng = self.cfg, self.rng
        victim = int(rng.choice(self.non_mules))
        n = int(rng.integers(c.ato_payees[0], c.ato_payees[1] + 1))
        payees = rng.choice(self.mules, n, replace=False)
        iid = self._new("ato", [victim, *payees])
        for p in payees:
            self.sim.schedule(rng.uniform(0.0, 2.0), self.emit, victim, int(p),
                              self._amount(6.5, 0.5), self._channel("mobile"), "ato", rng, iid)
        self.sim.schedule(rng.exponential(24.0 / c.ato_per_day), self._ato)
