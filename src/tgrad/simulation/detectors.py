"""
Detectors deployed in the Objective-6 simulator.

T-GRAD (the developed model, Objectives 1-4) is assembled from the SAME
components and training recipe as the static T-GRAD experiments
(https://github.com/profmcdan/tgrad-detector):

  * TemporalEncoderV2 (continuous-time memory encoder)  -> z_i
  * VP-SDE with ScoreNetworkGNN, trained on licit states  (Objective 3)
  * EnergyCycleGuidance, class-balanced BCE on -E(z)      (Objective 1 nodal
    distribution p(y=1|z) = sigmoid(-E(z)))
  * guided reconstruction anomaly rho_i (lambda = 10)     (Objective 3)
  * closed-walk monitor features (A_tau^L)_ii            (Objective 2)
  * FraudHead on the augmented state
        zeta_i = [ a_i || z_i || m_i || rho_i || -E(z_i) ]   (Objective 4)

The encoder is run STREAMING inside the simulation: at each scoring epoch it
consumes the transactions committed since the previous epoch and updates each
touched account's embedding (exponential moving average of its encoder output,
the streaming analogue of the mean pooling used offline).

Baselines: a rule engine of standard AML scenarios, an attribute-only MLP, and
GraphSAGE, GCN and GAT on the recent-transaction graph, each also given the
closed-walk block (the hybrids).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from ..models.cycle_guidance import EnergyCycleGuidance
from ..models.diffusion import ScoreNetworkGNN, VariancePreservingSDE
from ..models.temporal_encoder_v2 import TemporalEncoderV2
from .features import ACT_NAMES
from .system import EDGE_DIM, THRESHOLD


# ------------------------------------------------------------------ helpers
def ledger_messages(a: dict, lo: int, hi: int) -> np.ndarray:
    """Vectorised edge messages (same definition as system.tx_message)."""
    amt, ch, t = a["amount"][lo:hi], a["channel"][lo:hi], a["t_commit"][lo:hi]
    h = 2 * np.pi * (t % 24.0) / 24.0
    return np.stack([np.log1p(amt) / 10.0, ch == 0, ch == 1, ch == 2,
                     np.sin(h), np.cos(h),
                     (amt >= 0.9 * THRESHOLD) & (amt < THRESHOLD),
                     amt >= THRESHOLD], 1).astype(np.float32)


class FraudHead(nn.Module):
    """Same head as the static T-GRAD experiments."""

    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class HeadModel:
    """Standardise -> FraudHead -> class-balanced BCE (full batch Adam)."""

    def __init__(self, epochs=300, seed=0):
        self.epochs, self.seed = epochs, seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-8
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        pw = torch.tensor([(y == 0).sum() / max(1, (y == 1).sum())], dtype=torch.float32)
        self.net = FraudHead(X.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.epochs):
            self.net.train(); opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(self.net(Xt), yt, pos_weight=pw)
            loss.backward(); opt.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def score(self, X):
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        return torch.sigmoid(self.net(Xt)).numpy()


# ------------------------------------------------------------------ rules
_I = {n: i for i, n in enumerate(ACT_NAMES)}


def rule_score(act: np.ndarray) -> np.ndarray:
    """Count of fired rules from a typical AML/fraud rulebook (no training)."""
    r = [
        act[:, _I["out_n_24h"]] >= 6,                                   # velocity
        act[:, _I["near_thr_24h"]] >= 1,                                # structuring
        act[:, _I["over_thr_24h"]] >= 1,                                # large transfer
        (act[:, _I["pass_through_24h"]] >= 0.8)
        & (act[:, _I["in_amt_24h"]] >= math.log1p(1000.0)),             # pass-through / mule
        (act[:, _I["out_cp_24h"]] >= 4) & (act[:, _I["transfer_share_24h"]] >= 0.5),  # fan-out
    ]
    s = np.sum(r, axis=0).astype(float)
    # tie-break by out-flow so the score is a ranking, not only 0..5
    return s + 1e-3 * act[:, _I["out_amt_24h"]]


def monitor_score(mon: np.ndarray, L_max: int = 6, L_min: int = 2) -> np.ndarray:
    """Untrained Objective-2 monitor: M_i = sum_{L=L_min}^{L_max} 0.5^L (A_tau^L)_ii.

    L_min = 2 is the thesis default; L_min = 3 drops reciprocal (A<->B) pairs,
    which in a payments system are dominated by ordinary repayments."""
    diag = np.expm1(mon[:, :L_max - 1])
    w = np.array([0.5 ** L if L >= L_min else 0.0 for L in range(2, L_max + 1)])
    return diag @ w


def walk_weights(L_max: int = 6, L_min: int = 2) -> np.ndarray:
    """Weights over the per-transaction walk vector (index m-1 <-> cycle length m+1)."""
    return np.array([0.5 ** (m + 1) if m + 1 >= L_min else 0.0 for m in range(1, L_max)])


# ------------------------------------------------------------------ GraphSAGE
class SAGEDetector:
    """Two-layer GNN detector; ``conv`` selects GraphSAGE (default), GCN or GAT."""

    def __init__(self, hidden=64, epochs=150, seed=0, conv="sage"):
        self.hidden, self.epochs, self.seed, self.conv = hidden, epochs, seed, conv

    def _net(self, d):
        from torch_geometric.nn import GATConv, GCNConv, SAGEConv
        Conv = {"sage": SAGEConv, "gcn": GCNConv, "gat": GATConv}[self.conv]

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.c1, s.c2 = Conv(d, self.hidden), Conv(self.hidden, 1)
                s.drop = nn.Dropout(0.3)

            def forward(s, x, ei):
                return s.c2(s.drop(torch.relu(s.c1(x, ei))), ei).squeeze(-1)
        return Net()

    def fit(self, graphs):
        """graphs: list of (X_all (N,d), edge_index (2,E), nodes, y)."""
        torch.manual_seed(self.seed)
        Xcat = np.vstack([g[0][g[2]] for g in graphs])
        self.mu = Xcat.mean(0); self.sd = Xcat.std(0) + 1e-8
        ycat = np.concatenate([g[3] for g in graphs])
        pw = torch.tensor([(ycat == 0).sum() / max(1, (ycat == 1).sum())], dtype=torch.float32)
        self.net = self._net(Xcat.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=5e-3, weight_decay=5e-4)
        rng = np.random.default_rng(self.seed)
        data = [self._tensors(g) for g in graphs]
        for _ in range(self.epochs):
            self.net.train()
            for k in rng.permutation(len(data)):
                x, ei, nodes, y = data[k]
                opt.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(
                    self.net(x, ei)[nodes], y, pos_weight=pw)
                loss.backward(); opt.step()
        self.net.eval()
        return self

    def _tensors(self, g):
        X, ei, nodes, y = g
        return (torch.tensor((X - self.mu) / self.sd, dtype=torch.float32),
                torch.tensor(ei, dtype=torch.long), torch.tensor(nodes),
                torch.tensor(y, dtype=torch.float32))

    @torch.no_grad()
    def score(self, X, ei, nodes):
        x = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        return torch.sigmoid(self.net(x, torch.tensor(ei, dtype=torch.long))[nodes]).numpy()


# ------------------------------------------------------------------ T-GRAD neural
class TGradNeural:
    """Encoder + guided diffusion + energy (Objectives 1 and 3), streamed online."""

    def __init__(self, num_nodes, node_dim=64, time_dim=16, seed=0, ema=0.3,
                 guide_lambda=10.0, recon_steps=20):
        torch.manual_seed(seed)
        self.N, self.D, self.seed, self.ema = num_nodes, node_dim, seed, ema
        self.lam, self.steps = guide_lambda, recon_steps
        self.encoder = TemporalEncoderV2(num_nodes, node_dim, EDGE_DIM, time_dim)
        self.sde = VariancePreservingSDE(ScoreNetworkGNN(node_dim))
        self.guidance = EnergyCycleGuidance(node_dim)
        self.loss_hist = []
        self.reset_stream()

    # ---------------- training (offline, on the historical ledger) ----------------
    def fit(self, a, n_train, node_label, epochs=4, batch=512):
        enc, sde, gd = self.encoder, self.sde, self.guidance
        opt = torch.optim.Adam(list(enc.parameters()) + list(sde.parameters())
                               + list(gd.parameters()), lr=1e-3)
        lab = torch.tensor(node_label)
        pw = torch.tensor([(node_label == 0).sum() / max(1, (node_label == 1).sum())],
                          dtype=torch.float32).clamp(max=200.0)
        src_all = torch.tensor(a["src"][:n_train]); dst_all = torch.tensor(a["dst"][:n_train])
        t_all = torch.tensor(a["t_commit"][:n_train], dtype=torch.float32)
        msg_all = torch.tensor(ledger_messages(a, 0, n_train))
        for ep in range(epochs):
            enc.memory.reset_memory(); el = []
            for s in range(0, n_train, batch):
                e = min(s + batch, n_train)
                src, dst = src_all[s:e], dst_all[s:e]
                sz, dz = enc(src, dst, t_all[s:e], msg_all[s:e])
                z = torch.cat([sz, dz]); lbl = torch.cat([lab[src], lab[dst]])
                licit = lbl == 0
                ld = sde.compute_loss(z[licit], torch.arange(int(licit.sum())).unsqueeze(0).repeat(2, 1)) \
                    if licit.sum() > 4 else torch.tensor(0.0)
                lg = gd.compute_bce_loss(z, lbl, pos_weight=pw)
                loss = ld + lg
                opt.zero_grad(); loss.backward(); opt.step(); el.append(loss.item())
            self.loss_hist.append(float(np.mean(el)))
        enc.eval()
        self.reset_stream()
        return self

    # ---------------- streaming inference ----------------
    def reset_stream(self):
        self.encoder.memory.reset_memory()
        self.emb = torch.zeros(self.N, self.D)
        self.seen = torch.zeros(self.N, dtype=torch.bool)
        self.pos = 0

    @torch.no_grad()
    def advance(self, a, upto, batch=1024):
        """Consume ledger rows [pos, upto) -- the transactions committed since
        the previous scoring epoch -- and update account embeddings."""
        while self.pos < upto:
            e = min(self.pos + batch, upto)
            src = torch.tensor(a["src"][self.pos:e]); dst = torch.tensor(a["dst"][self.pos:e])
            t = torch.tensor(a["t_commit"][self.pos:e], dtype=torch.float32)
            msg = torch.tensor(ledger_messages(a, self.pos, e))
            sz, dz = self.encoder(src, dst, t, msg)
            for ids, z in ((src, sz), (dst, dz)):
                for i, zi in zip(ids.tolist(), z):
                    if self.seen[i]:
                        self.emb[i] = (1 - self.ema) * self.emb[i] + self.ema * zi
                    else:
                        self.emb[i] = zi; self.seen[i] = True
            self.pos = e

    def recon(self, z, lam):
        """Guided reconstruction anomaly rho (same sampler as the static experiments)."""
        g = torch.Generator().manual_seed(self.seed)
        n = z.shape[0]
        e = torch.arange(n).unsqueeze(0).repeat(2, 1)
        mean, std = self.sde.marginal_prob(z, torch.ones(n))
        x = mean + std[:, None] * torch.randn(z.shape, generator=g)
        ts = torch.linspace(1.0, 1e-3, self.steps); dt = 1.0 / self.steps
        for k, tv in enumerate(ts):
            beta = self.sde.beta_t(tv)
            with torch.no_grad():
                score = self.sde.score_net(x, e, torch.ones(n) * tv)
            if lam > 0:
                score = score + lam * self.guidance.compute_cycle_gradient(x).clamp(-5, 5)
            drift = -0.5 * beta * x - beta * score
            noise = torch.randn(z.shape, generator=g) if k < self.steps - 1 else 0.0
            x = (x - drift * dt + math.sqrt(beta) * math.sqrt(dt) * noise).detach()
        return torch.mean((z - x) ** 2, 1).numpy()

    def features(self, nodes):
        """[z || rho_guided || -E(z)] for the given accounts."""
        z = self.emb[torch.as_tensor(nodes)]
        with torch.no_grad():
            energy = self.guidance.fraud_logit(z).numpy()
        rho = self.recon(z, self.lam)
        return np.hstack([z.numpy(), rho[:, None], energy[:, None]])
