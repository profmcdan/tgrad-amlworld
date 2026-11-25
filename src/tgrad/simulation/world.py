"""
Benign financial world for the Objective-6 discrete-event simulator.

Population
----------
  * retail accounts   -- individuals: card purchases, P2P transfers, repayments,
                         group settle-ups, salary receipts, owner draws;
  * merchant accounts -- receive purchases, pay suppliers, pay staff, pay the owner;
  * business accounts -- suppliers / employers: pay salaries, trade B2B.

Every behaviour is a self-rescheduling DES process driven by the BENIGN random
stream only. Fraud (``fraud.py``) draws from a separate stream, so a run with
fraud and the matching run without fraud share *identical* benign traffic
(common random numbers): any difference between the two is caused by the fraud
alone.

Money circulates: employee -> merchant -> supplier -> employee (salary) forms
genuine *benign* directed cycles over days/weeks, and friends' settle-ups and
repayments form short benign cycles within hours/days. These are the hard
negatives a cycle-based detector must not flag.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

RETAIL, MERCHANT, BUSINESS = 0, 1, 2

# benign transaction kinds (kept on every record for analysis only;
# detectors never see them)
KIND = {"purchase": 0, "p2p": 1, "repay": 2, "settle": 3, "salary": 4,
        "supplier": 5, "owner": 6, "b2b": 7,
        # fraud typologies
        "cycle": 10, "fanout": 11, "ato": 12}
FRAUD_KINDS = {10, 11, 12}
CHANNELS = {"card": 0, "transfer": 1, "mobile": 2}


@dataclass
class WorldConfig:
    n_retail: int = 1800
    n_merchant: int = 150
    n_business: int = 50
    days: float = 28.0
    purchase_rate: float = 1.5          # mean card purchases / retail / day
    p2p_rate: float = 0.25              # mean P2P transfers / retail / day
    repay_prob: float = 0.4             # P(recipient repays)  -> benign 2-cycles
    settle_per_day: float = 1.0         # benign fast friend cycles / day (hard negatives)
    settle_len: tuple = (3, 4)
    settle_window_h: tuple = (1.0, 12.0)
    supplier_every_h: float = 84.0      # merchant -> supplier mean inter-payment
    salary_every_h: float = 336.0       # biweekly payroll
    owner_every_h: float = 168.0        # weekly owner draw
    b2b_per_day: float = 0.5
    friends_k: int = 6                  # Watts-Strogatz friendship graph
    friends_p: float = 0.1
    service_mean_s: float = 0.4         # payment-switch service time (seconds)
    switch_servers: int = 4
    seed: int = 0


def diurnal(hour: float) -> float:
    """Relative activity by hour of day (peak ~13:00-19:00, trough ~04:00), max 1."""
    h = hour % 24.0
    return 0.15 + 0.85 * math.exp(-((h - 15.0) ** 2) / (2 * 4.5 ** 2))


class Population:
    """Accounts, their static relationships and attributes."""

    def __init__(self, cfg: WorldConfig, rng: np.random.Generator):
        self.cfg = cfg
        nr, nm, nb = cfg.n_retail, cfg.n_merchant, cfg.n_business
        self.N = nr + nm + nb
        self.retail = np.arange(nr)
        self.merchant = np.arange(nr, nr + nm)
        self.business = np.arange(nr + nm, self.N)
        self.kind = np.zeros(self.N, dtype=np.int64)
        self.kind[self.merchant] = MERCHANT
        self.kind[self.business] = BUSINESS
        # tenure (days the account has existed) -- a static attribute
        self.tenure = rng.gamma(2.0, 400.0, self.N)

        # per-retail heterogeneous behaviour rates
        self.purchase_rate = cfg.purchase_rate * rng.lognormal(0, 0.5, nr) / math.exp(0.125)
        self.p2p_rate = cfg.p2p_rate * rng.lognormal(0, 0.6, nr) / math.exp(0.18)
        self.spend_scale = rng.lognormal(0, 0.4, nr)

        # merchant popularity (Zipf) and each retail's favourite merchants
        pop = 1.0 / np.arange(1, nm + 1) ** 0.9
        self.merchant_p = pop / pop.sum()
        self.favourites = [rng.choice(self.merchant, 8, replace=False, p=self.merchant_p)
                           for _ in range(nr)]

        # friendship graph among retail (Watts-Strogatz small world)
        k, p = cfg.friends_k, cfg.friends_p
        friends = [set() for _ in range(nr)]
        for i in range(nr):
            for j in range(1, k // 2 + 1):
                a, b = i, (i + j) % nr
                if rng.random() < p:
                    b = int(rng.integers(nr))
                    if b == a:
                        continue
                friends[a].add(b); friends[b].add(a)
        self.friends = [np.array(sorted(f)) for f in friends]

        # employment: 70% business, 30% merchant employers
        employers = np.where(rng.random(nr) < 0.7,
                             rng.choice(self.business, nr),
                             rng.choice(self.merchant, nr))
        self.employer = employers
        self.staff = {e: [] for e in np.concatenate([self.merchant, self.business])}
        for i, e in enumerate(employers):
            self.staff[int(e)].append(i)
        self.payroll_phase = {e: float(rng.uniform(0, cfg.salary_every_h))
                              for e in self.staff}
        # merchants' suppliers and owners
        self.suppliers = {int(m): rng.choice(self.business, int(rng.integers(1, 4)), replace=False)
                          for m in self.merchant}
        self.owner = {int(m): int(rng.integers(nr)) for m in self.merchant}


class BenignWorld:
    """Registers all benign behaviour processes on a simulator.

    ``emit(src, dst, amount, channel, kind, rng, inst=-1)`` is supplied by the
    payment system and submits a transaction for processing.
    """

    def __init__(self, sim, pop: Population, emit, rng: np.random.Generator):
        self.sim, self.pop, self.emit, self.rng = sim, pop, emit, rng
        self.cfg = pop.cfg
        self.horizon = self.cfg.days * 24.0

    # -------------------------------------------------------------- helpers
    def _thinned_next(self, rate_per_h: float) -> float:
        """Next arrival of a diurnally modulated Poisson process (thinning)."""
        t = self.sim.now
        rng = self.rng
        while True:
            t += rng.exponential(1.0 / rate_per_h)
            if rng.random() < diurnal(t):
                return t - self.sim.now

    # -------------------------------------------------------------- start
    def start(self):
        pop, rng = self.pop, self.rng
        for i in pop.retail:
            i = int(i)
            self.sim.schedule(self._thinned_next(pop.purchase_rate[i] / 24.0), self._purchase, i)
            self.sim.schedule(self._thinned_next(pop.p2p_rate[i] / 24.0), self._p2p, i)
        for e in pop.staff:
            if pop.staff[e]:
                self.sim.schedule(pop.payroll_phase[e], self._payroll, e)
        for m in pop.merchant:
            m = int(m)
            self.sim.schedule(rng.exponential(self.cfg.supplier_every_h), self._supplier, m)
            self.sim.schedule(rng.uniform(0, self.cfg.owner_every_h), self._owner, m)
        if self.cfg.b2b_per_day > 0:
            self.sim.schedule(rng.exponential(24.0 / self.cfg.b2b_per_day), self._b2b)
        if self.cfg.settle_per_day > 0:
            self.sim.schedule(rng.exponential(24.0 / self.cfg.settle_per_day), self._settle)

    # -------------------------------------------------------------- processes
    def _purchase(self, i):
        pop, rng = self.pop, self.rng
        m = rng.choice(pop.favourites[i]) if rng.random() < 0.85 else \
            rng.choice(pop.merchant, p=pop.merchant_p)
        amt = rng.lognormal(3.4, 0.9) * pop.spend_scale[i]
        ch = "card" if rng.random() < 0.8 else "mobile"
        self.emit(i, int(m), amt, ch, "purchase", rng)
        self.sim.schedule(self._thinned_next(pop.purchase_rate[i] / 24.0), self._purchase, i)

    def _p2p(self, i):
        pop, rng = self.pop, self.rng
        fr = pop.friends[i]
        if len(fr):
            j = int(rng.choice(fr))
            amt = rng.lognormal(4.3, 1.0)
            self.emit(i, j, amt, "mobile" if rng.random() < 0.6 else "transfer", "p2p", rng)
            if rng.random() < self.cfg.repay_prob:
                delay = rng.lognormal(math.log(24.0), 1.0)
                self.sim.schedule(delay, self._repay, j, i, amt * rng.uniform(0.9, 1.0))
        self.sim.schedule(self._thinned_next(pop.p2p_rate[i] / 24.0), self._p2p, i)

    def _repay(self, j, i, amt):
        self.emit(j, i, amt, "mobile", "repay", self.rng)

    def _settle(self):
        """Benign fast cycle among friends (shared-expense settle-up)."""
        pop, rng = self.pop, self.rng
        lo, hi = self.cfg.settle_len
        L = int(rng.integers(lo, hi + 1))
        a = int(rng.choice(pop.retail))
        ring = [a]
        while len(ring) < L:
            fr = pop.friends[ring[-1]]
            cand = [int(x) for x in fr if int(x) not in ring]
            if not cand:
                break
            ring.append(int(rng.choice(cand)))
        if len(ring) >= 3:
            W = rng.uniform(*self.cfg.settle_window_h)
            gaps = rng.dirichlet(np.ones(len(ring))) * W
            amt = rng.lognormal(4.0, 0.7)
            t = 0.0
            for k in range(len(ring)):
                t += gaps[k]
                self.sim.schedule(t, self.emit, ring[k], ring[(k + 1) % len(ring)],
                                  amt * rng.uniform(0.8, 1.2), "mobile", "settle", rng)
        self.sim.schedule(rng.exponential(24.0 / self.cfg.settle_per_day), self._settle)

    def _payroll(self, e):
        pop, rng = self.pop, self.rng
        for k, i in enumerate(pop.staff[e]):
            self.sim.schedule(k * 0.01, self.emit, e, int(i),
                              rng.lognormal(7.4, 0.3), "transfer", "salary", rng)
        self.sim.schedule(self.cfg.salary_every_h, self._payroll, e)

    def _supplier(self, m):
        pop, rng = self.pop, self.rng
        s = int(rng.choice(pop.suppliers[m]))
        self.emit(m, s, rng.lognormal(7.0, 0.8), "transfer", "supplier", rng)
        self.sim.schedule(rng.exponential(self.cfg.supplier_every_h), self._supplier, m)

    def _owner(self, m):
        rng = self.rng
        self.emit(m, self.pop.owner[m], rng.lognormal(7.2, 0.5), "transfer", "owner", rng)
        self.sim.schedule(self.cfg.owner_every_h, self._owner, m)

    def _b2b(self):
        pop, rng = self.pop, self.rng
        a, b = rng.choice(pop.business, 2, replace=False)
        self.emit(int(a), int(b), rng.lognormal(8.0, 0.9), "transfer", "b2b", rng)
        self.sim.schedule(rng.exponential(24.0 / self.cfg.b2b_per_day), self._b2b)
