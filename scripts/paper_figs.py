"""Figures of the paper, drawn from the result files.

Called by paper_tables.py; every plotted value is read from results/<NAME>/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
DS = ["HI-Small", "LI-Small", "HI-Medium", "LI-Medium"]
COL = {"HI-Small": "#1f5fa8", "LI-Small": "#6aa6e0", "HI-Medium": "#b4232a", "LI-Medium": "#ef8a8f"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def J(*p):
    return json.load(open(os.path.join(R, *p)))


def fig_pipeline(path):
    fig, ax = plt.subplots(figsize=(9.2, 2.7)); ax.set_xlim(0, 10.4); ax.set_ylim(0, 2.9); ax.axis("off")

    def box(x, y, w, h, head, sub, fc, tc="#1b1b2f"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="none"))
        ax.text(x + w / 2, y + h * 0.68, head, ha="center", va="center", fontsize=9, weight="bold", color=tc)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=7, color=tc)

    def arr(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color="#8d99ae", lw=1.3))
    box(0.0, 1.0, 1.55, 0.95, "Ledger", "transfers up to\nsnapshot time", "#f1f3f6")
    box(1.85, 1.0, 1.55, 0.95, "Observed graph", "self-transfers\nremoved", "#f1f3f6")
    box(3.75, 1.75, 2.1, 0.95, "Closed walks", "$(A^L)_{ii}$, $L\\leq12$; decayed\n$A_\\tau$, strengths, streaming", "#e3ecf7")
    box(3.75, 0.2, 2.1, 0.95, "Temporal cycles", "time-ordered loops\nwithin $W$ = 168 h", "#e3ecf7")
    box(6.2, 1.0, 1.55, 0.95, "Rank-scaling", "within each\nsnapshot", "#e3ecf7")
    box(8.1, 1.0, 2.2, 0.95, "T-GRAD + GNN", "attributes $\\|$ ranked block\n$\\|$ log(1 + cycles)", "#b4232a", "white")
    arr(1.55, 1.47, 1.85, 1.47); arr(3.4, 1.6, 3.75, 2.1); arr(3.4, 1.35, 3.75, 0.75)
    arr(5.85, 2.2, 6.2, 1.65); arr(5.85, 0.7, 8.1, 1.15); arr(7.75, 1.47, 8.1, 1.47)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig_coverage(path):
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.0), sharex=True)
    for d in DS:
        c = J(d, "coverage_by_L.json"); k = c["k"]
        axs[0].plot(k, c["share"]["CYCLE"], marker="o", ms=3, color=COL[d], label=f"{d} (n = {c['n']['CYCLE']})")
        axs[1].plot(k, c["share"]["illicit"], marker="o", ms=3, color=COL[d], label=f"{d} illicit")
        axs[1].plot(k, c["share"]["licit"], ls="--", color=COL[d], lw=1, label=f"{d} licit")
    axs[0].set_ylabel("share of CYCLE accounts"); axs[1].set_ylabel("share of accounts")
    for a in axs:
        a.set_xlabel("longest closed walk counted, $L_{\\max}$"); a.set_xticks(range(2, 13))
        a.spines[["top", "right"]].set_visible(False); a.set_ylim(0, 1.02 if a is axs[0] else 0.25)
    axs[0].axvline(6, color="#888", lw=0.8, ls=":")
    axs[1].axvline(6, color="#888", lw=0.8, ls=":")
    axs[0].legend(fontsize=7, frameon=False, loc="upper left")
    h, l = axs[1].get_legend_handles_labels()
    axs[1].legend(h, l, fontsize=6, frameon=False, ncol=2, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig_headline(path, rows):
    """rows[d] = (label, tgrad metrics, baseline metrics), metrics m -> (mean, std)."""
    fig, axs = plt.subplots(1, 3, figsize=(9.2, 2.6))
    x = range(len(DS)); w = 0.38
    for ax, m, lab in zip(axs, ("PR-AUC", "F1", "CYCLE R@1%FPR"),
                          ("PR-AUC", "F1", "CYCLE recall at 1% FPR")):
        a = [rows[d][1][m] for d in DS]; b = [rows[d][2][m] for d in DS]
        ax.bar([i - w / 2 for i in x], [v[0] for v in a], w, yerr=[v[1] for v in a], capsize=2,
               color="#b4232a", label="T-GRAD features")
        ax.bar([i + w / 2 for i in x], [v[0] for v in b], w, yerr=[v[1] for v in b], capsize=2,
               color="#a0aec0", label="Graph Transformer, matched")
        ax.set_xticks(list(x)); ax.set_xticklabels([d.replace("-", "-\n") for d in DS], fontsize=7)
        ax.set_ylabel(lab, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(path, dpi=300); plt.close(fig)
