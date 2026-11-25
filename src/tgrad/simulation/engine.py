"""
Discrete-event simulation (DES) kernel.

A minimal, dependency-free next-event-time-advance simulator:

  * the future-event list is a binary heap ordered by (time, sequence number),
    so simultaneous events fire in the order they were scheduled (deterministic);
  * the clock jumps from one event to the next -- nothing happens between events;
  * an event is just a callback ``fn(*args)`` that may schedule further events.

On top of the kernel, :class:`Server` models an M/G/c FIFO service station
(the payment switch). It records waiting time, sojourn time and time-weighted
queue length, which Objective 6 uses to measure the operational cost of running
the detector inline with transaction processing.

Time unit throughout the simulation package is the HOUR.
"""
from __future__ import annotations

import heapq
import itertools
from collections import deque
from typing import Callable


class Simulator:
    """Next-event-time-advance DES kernel."""

    def __init__(self, start: float = 0.0):
        self.now = float(start)
        self._fel = []                      # future-event list (heap)
        self._seq = itertools.count()
        self.n_events = 0

    def schedule(self, delay: float, fn: Callable, *args):
        """Schedule ``fn(*args)`` to fire ``delay`` hours from now."""
        if delay < 0:
            raise ValueError("cannot schedule into the past")
        heapq.heappush(self._fel, (self.now + delay, next(self._seq), fn, args))

    def at(self, t: float, fn: Callable, *args):
        """Schedule ``fn(*args)`` at absolute time ``t`` (>= now)."""
        self.schedule(max(0.0, t - self.now), fn, *args)

    def peek(self) -> float:
        return self._fel[0][0] if self._fel else float("inf")

    def run(self, until: float):
        """Advance the clock, firing events with time <= ``until``."""
        fel = self._fel
        while fel and fel[0][0] <= until:
            t, _, fn, args = heapq.heappop(fel)
            self.now = t
            fn(*args)
            self.n_events += 1
        self.now = until


class Server:
    """M/G/c FIFO service station on a :class:`Simulator`.

    Parameters
    ----------
    sim : Simulator
    capacity : int
        Number of parallel servers c.
    service_time : callable(job) -> float
        Service-time sampler (hours).
    on_complete : callable(job)
        Called when a job finishes service (e.g. commit to the ledger).
    """

    def __init__(self, sim: Simulator, capacity: int,
                 service_time: Callable, on_complete: Callable):
        self.sim = sim
        self.capacity = int(capacity)
        self.service_time = service_time
        self.on_complete = on_complete
        self.busy = 0
        self.queue = deque()
        # statistics
        self.n_done = 0
        self.wait = []
        self.sojourn = []
        self._area_q = 0.0
        self._area_busy = 0.0
        self._last = sim.now

    def _account(self):
        dt = self.sim.now - self._last
        self._area_q += dt * len(self.queue)
        self._area_busy += dt * self.busy
        self._last = self.sim.now

    def submit(self, job):
        self._account()
        job.arrive = self.sim.now
        if self.busy < self.capacity:
            self._start(job)
        else:
            self.queue.append(job)

    def _start(self, job):
        self.busy += 1
        job.start = self.sim.now
        self.sim.schedule(self.service_time(job), self._finish, job)

    def _finish(self, job):
        self._account()
        self.busy -= 1
        job.finish = self.sim.now
        self.n_done += 1
        self.wait.append(job.start - job.arrive)
        self.sojourn.append(job.finish - job.arrive)
        self.on_complete(job)
        if self.queue:
            self._start(self.queue.popleft())

    def stats(self, horizon: float) -> dict:
        self._account()
        import numpy as np
        w = np.asarray(self.wait) if self.wait else np.zeros(1)
        s = np.asarray(self.sojourn) if self.sojourn else np.zeros(1)
        return {
            "jobs": self.n_done,
            "mean_wait_h": float(w.mean()),
            "mean_sojourn_h": float(s.mean()),
            "p50_sojourn_h": float(np.percentile(s, 50)),
            "p99_sojourn_h": float(np.percentile(s, 99)),
            "mean_queue_len": self._area_q / max(horizon, 1e-12),
            "utilisation": self._area_busy / max(horizon * self.capacity, 1e-12),
        }
