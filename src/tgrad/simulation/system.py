"""
Payment system: the processing switch, the committed-transaction ledger and the
hooks through which monitoring / detection components observe the stream.

Flow of one transaction in the DES
----------------------------------
    initiate (benign or fraud process)
        -> submit to the payment switch  (M/G/c queue, :class:`engine.Server`)
        -> service completes             -> COMMIT to the ledger
        -> every registered commit hook sees the transaction (streaming monitor)
    periodic SCORING EPOCH events (every ``epoch_h``) call the epoch hooks
    (batch node scoring, alerting).

The ledger is append-only and columnar (python lists, frozen to numpy by
:meth:`Ledger.arrays`) so feature extraction can slice time windows quickly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .engine import Server, Simulator
from .fraud import FraudConfig, FraudInjector
from .world import CHANNELS, FRAUD_KINDS, KIND, BenignWorld, Population, WorldConfig

EDGE_DIM = 8
THRESHOLD = 10_000.0   # reporting threshold (structuring reference)


class Tx:
    __slots__ = ("id", "src", "dst", "amount", "channel", "kind", "inst",
                 "t_init", "service", "arrive", "start", "finish")

    def __init__(self, src, dst, amount, channel, kind, inst, t_init, service):
        self.src, self.dst, self.amount = src, dst, amount
        self.channel, self.kind, self.inst = channel, kind, inst
        self.t_init, self.service = t_init, service
        self.id = -1
        self.arrive = self.start = self.finish = t_init


def tx_message(amount, channel, t) -> np.ndarray:
    """Edge message x_e fed to the temporal encoder (no label information)."""
    h = 2 * math.pi * (t % 24.0) / 24.0
    return np.array([
        math.log1p(amount) / 10.0,
        channel == 0, channel == 1, channel == 2,
        math.sin(h), math.cos(h),
        0.9 * THRESHOLD <= amount < THRESHOLD,
        amount >= THRESHOLD,
    ], dtype=np.float32)


class Ledger:
    """Append-only columnar record of committed transactions."""

    def __init__(self):
        self.cols = {k: [] for k in ("src", "dst", "amount", "channel", "kind",
                                     "inst", "t_init", "t_commit", "wait")}
        self._frozen = None
        self._frozen_n = -1

    def append(self, tx: Tx):
        c = self.cols
        tx.id = len(c["src"])
        c["src"].append(tx.src); c["dst"].append(tx.dst); c["amount"].append(tx.amount)
        c["channel"].append(tx.channel); c["kind"].append(tx.kind); c["inst"].append(tx.inst)
        c["t_init"].append(tx.t_init); c["t_commit"].append(tx.finish)
        c["wait"].append(tx.start - tx.arrive)

    def __len__(self):
        return len(self.cols["src"])

    def arrays(self) -> dict:
        """Numpy view of the ledger (cached until the next append)."""
        n = len(self)
        if self._frozen_n != n:
            a = {k: np.asarray(v) for k, v in self.cols.items()}
            a["fraud"] = np.isin(a["kind"], list(FRAUD_KINDS)) if n else np.zeros(0, bool)
            self._frozen, self._frozen_n = a, n
        return self._frozen


class PaymentSystem:
    """Owns the simulator, the world, the fraud injector and the switch."""

    def __init__(self, world_cfg: WorldConfig, fraud_cfg: FraudConfig,
                 epoch_h: float = 6.0):
        self.wcfg, self.fcfg = world_cfg, fraud_cfg
        self.sim = Simulator()
        self.horizon = world_cfg.days * 24.0
        rng_pop = np.random.default_rng(world_cfg.seed)
        self.rng_benign = np.random.default_rng(world_cfg.seed + 1)
        self.pop = Population(world_cfg, rng_pop)
        self.ledger = Ledger()
        self.commit_hooks: List[Callable] = []
        self.epoch_hooks: List[Callable] = []
        self.epoch_h = epoch_h
        self.extra_service: Optional[Callable] = None   # inline-scoring cost (hours)
        self.switch = Server(self.sim, world_cfg.switch_servers,
                             self._service_time, self._commit)
        self.world = BenignWorld(self.sim, self.pop, self.emit, self.rng_benign)
        self.fraud = FraudInjector(self.sim, self.pop, self.emit, fraud_cfg, self.horizon)
        self._started = False

    # ---------------------------------------------------------------- tx flow
    def emit(self, src, dst, amount, channel, kind, rng, inst=-1):
        svc = rng.exponential(self.wcfg.service_mean_s) / 3600.0
        tx = Tx(int(src), int(dst), float(amount), CHANNELS[channel], KIND[kind],
                int(inst), self.sim.now, svc)
        self.switch.submit(tx)

    def _service_time(self, tx: Tx) -> float:
        extra = self.extra_service(tx) if self.extra_service else 0.0
        return tx.service + extra

    def _commit(self, tx: Tx):
        self.ledger.append(tx)
        for h in self.commit_hooks:
            h(tx)

    def _epoch(self):
        for h in self.epoch_hooks:
            h(self.sim.now)
        if self.sim.now + self.epoch_h <= self.horizon + 1e-9:
            self.sim.schedule(self.epoch_h, self._epoch)

    # ---------------------------------------------------------------- control
    def start(self):
        if not self._started:
            self.world.start()
            self.fraud.start()
            self.sim.schedule(self.epoch_h, self._epoch)
            self._started = True

    def run_until(self, t_h: float):
        """Advance the simulation (can be called repeatedly: train, then deploy)."""
        self.start()
        self.sim.run(min(t_h, self.horizon))
        return self
