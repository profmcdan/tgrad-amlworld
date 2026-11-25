# Drift-invariant cycle features for money-laundering detection on AMLworld

Code and results for the paper

> D. T. Ale, E. O. Ogunti, S. A. Oyetunji, A. A. Ponnle and K. B. Adedeji,
> *Drift-Invariant Cycle Features for Account-Level Money-Laundering Detection: A Pre-Specified
> Evaluation of T-GRAD on AMLworld*, manuscript, 2026.

Graph neural networks (GNNs) cannot count cycles, yet cyclic layering is a core money-laundering
pattern. This repository evaluates whether per-account cycle statistics from the T-GRAD detector,
given to a standard GNN as input features, improve account-level detection on the AMLworld
benchmark (Altman et al., NeurIPS 2023). It contains:

- the **closed-walk monitor** $M_i=\sum_{L=2}^{L_{\max}}2^{-L}(A^L)_{ii}$ on the observed transaction
  graph (binary or exponentially decayed), with a cap of $L_{\max}=12$, computed through
  $\mathrm{diag}(A^L)=\mathrm{rowsum}(A^a\circ(A^{L-a})^\top)$ without forming high powers;
- an exact **time-respecting cycle count**: for every transfer, a layered earliest-arrival /
  latest-departure search decides whether it closes a loop whose transfers are in time order within
  a window $W$, and credits the accounts on it;
- **within-snapshot rank-scaling** of the T-GRAD feature block, which removes train-to-test drift
  that preserves the order of accounts;
- the **evaluation**: six model families (MLP, GCN, GraphSAGE, GAT, GIN, Graph Transformer) with
  attributes only, a matched structural block and the T-GRAD block; development on HI-Small and
  LI-Small; confirmation on HI-Medium and LI-Medium with every setting and six claims fixed before
  those runs.

## Main results

Test window, mean ± std over seeds 0, 1, 2 (from `results/paper_tables.md`). "T-GRAD + Graph
Transformer" is the Graph Transformer given the rank-scaled T-GRAD block and the temporal-cycle count.

| Set | Model | ROC-AUC | PR-AUC | F1 | Recall @1% FPR | CYCLE recall @1% FPR |
|---|---|---|---|---|---|---|
| LI-Medium | T-GRAD + Graph Transformer | 0.913 ± 0.001 | 0.1362 ± 0.0087 | 0.121 ± 0.022 | 0.341 ± 0.019 | 0.915 ± 0.014 |
| LI-Medium | Graph Transformer, matched block | 0.907 ± 0.001 | 0.1175 ± 0.0075 | 0.080 ± 0.004 | 0.356 ± 0.019 | 0.789 ± 0.018 |

Of the six claims fixed before the confirmation runs, five hold (C2, C3, D1, D2, D3) and one fails
(C1: the training-free monitor does not beat every learned model on CYCLE recall). On all four sets
a GNN given the T-GRAD features has a higher PR-AUC than the matched Graph Transformer (+8% to
+40%), with ROC-AUC within 0.003 or higher; recall at a 1% false-positive rate does not improve.

## Layout

```
scripts/ibm_aml_common.py           protocol shared by all scripts: temporal split, snapshots, attributes, metrics
scripts/ibm_aml_prepare.py          parse <NAME>_Trans.csv and <NAME>_Patterns.txt into integer arrays
scripts/ibm_aml_snapshots.py        train/validation/test snapshots, decay constant, monitors, coverage
scripts/ibm_aml_models.py           MLP and five GNNs with attributes / matched block / raw T-GRAD block
scripts/ibm_aml_tgrad_full.py       the full T-GRAD detector (encoder, guided diffusion, energy head) on HI-Small
scripts/ibm_aml_aggregate.py        results/<NAME>/tables.md and summary.json
scripts/ibm_aml_temporal_cycles.py  time-respecting cycle counts, streamed over the ledger
scripts/ibm_aml_improve_monitors.py validation selection of the closed-walk cap and of (W, cap)
scripts/ibm_aml_ablation.py         one feature group at a time, rank-scaled block, drift (KS) diagnostic
scripts/ibm_aml_aggregate_improve.py results/<NAME>/improve_tables.md
scripts/ibm_aml_combine.py          Graph Transformer with T-GRAD features, and fusion with the monitor
scripts/ibm_aml_confirm.py          confirmation runs on HI-Medium (claims C1-C3) and LI-Medium (D1-D3)
scripts/coverage_by_length.py       share of accounts on a closed walk of length <= L, L = 2..12
scripts/paper_tables.py             every table and figure of the paper, from results/ (no training)
src/tgrad/                          monitor, detectors and models, copied unchanged from
                                    profmcdan/tgrad-event-simulator (commit 8d0d803)
results/<NAME>/                     per-seed results (JSON) and the generated tables
figures/                            figures of the paper
```

## Setup

Python 3.13 on CPU. The reported results were produced with torch 2.15.0, torch-geometric 2.8.1,
numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.1.1 and matplotlib 3.11.2 on an Apple M5 Max
(18 cores, 36 GB).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Data

AMLworld is not redistributed here. Download the transaction and pattern files from the authors'
release on Kaggle (*IBM Transactions for Anti Money Laundering (AML)*, by E. Altman) and place them in
`data/ibm_aml/`:

```
data/ibm_aml/HI-Small_Trans.csv     data/ibm_aml/HI-Small_Patterns.txt
data/ibm_aml/LI-Small_Trans.csv     data/ibm_aml/LI-Small_Patterns.txt
data/ibm_aml/HI-Medium_Trans.csv    data/ibm_aml/HI-Medium_Patterns.txt
data/ibm_aml/LI-Medium_Trans.csv    data/ibm_aml/LI-Medium_Patterns.txt
```

`data/` is git-ignored; intermediate arrays and snapshot caches are written to `data/ibm_aml_proc/`.

## Reproducing the paper

The tables and figures can be rebuilt from the saved results in under a minute:

```bash
python scripts/paper_tables.py          # results/paper_tables.md and figures/paper_*.png
```

To rerun everything from the raw data, run the stages below in order. Times are for the machine
above; the medium sets need about 27 GB of memory for the Graph Transformer, so run their seeds one
after another.

**Development sets** (`NAME` = `HI-Small`, then `LI-Small`; several hours per set, of which the 18 models of one seed take about 40 minutes on HI-Small):

```bash
python scripts/ibm_aml_prepare.py   --name $NAME
python scripts/ibm_aml_snapshots.py --name $NAME            # tau chosen on the training snapshot from {6, 24, 72, 168} h
for s in 0 1 2; do python scripts/ibm_aml_models.py --name $NAME --seed $s; done
python scripts/ibm_aml_aggregate.py --name $NAME
for W in 6 24 72 168; do python scripts/ibm_aml_temporal_cycles.py --name $NAME --window $W; done
python scripts/ibm_aml_improve_monitors.py --name $NAME     # validation selection of the caps and W
for s in 0 1 2; do python scripts/ibm_aml_ablation.py --name $NAME --seed $s; done
python scripts/ibm_aml_aggregate_improve.py --name $NAME
for s in 0 1 2; do python scripts/ibm_aml_combine.py models --name $NAME --seed $s; done
python scripts/ibm_aml_combine.py table --name $NAME
```

The full T-GRAD detector (HI-Small only; about one hour per seed):

```bash
for s in 0 1 2; do python scripts/ibm_aml_tgrad_full.py --name HI-Small --seed $s; done
python scripts/ibm_aml_aggregate.py --name HI-Small
```

**Confirmation sets** (`NAME` = `HI-Medium`, then `LI-Medium`; every setting is fixed in
`scripts/ibm_aml_confirm.py`; about 15 hours per set, mostly the Graph Transformer):

```bash
python scripts/ibm_aml_prepare.py   --name $NAME
python scripts/ibm_aml_snapshots.py --name $NAME --taus 72
python scripts/ibm_aml_temporal_cycles.py --name $NAME --window 168
python scripts/ibm_aml_confirm.py monitors --name $NAME
for s in 0 1 2; do python scripts/ibm_aml_confirm.py models --name $NAME --seed $s; done
python scripts/ibm_aml_confirm.py table --name $NAME       # plan C for HI-Medium, D for LI-Medium
```

Finally `python scripts/coverage_by_length.py` and `python scripts/paper_tables.py`.

## Protocol in brief

Transfers are sorted by time and cut at the 60% and 80% quantiles of their count. Each snapshot
uses every transfer observed so far and none after; the accounts scored are those active in the
window, labelled illicit if they take part in a laundering transfer in it. Models, standardisation,
the decay constant, the best-F1 threshold and the 1%-false-positive threshold use the training
snapshot only; validation is used only for the stated selections; the test window is never used for
a choice. Every GNN has two layers and hidden width 64 and is trained with class-weighted
cross-entropy, Adam (learning rate 0.01, weight decay 5e-4), 300 full-batch epochs, no early
stopping, seeds 0, 1, 2.

## Verification

- `scripts/paper_tables.py` reproduces every number of the paper's tables from `results/`.
- The aggregation scripts (`ibm_aml_aggregate.py`, `ibm_aml_aggregate_improve.py`,
  `ibm_aml_combine.py table`, `ibm_aml_confirm.py table`) regenerate the shipped summaries from the
  per-seed files byte for byte.
- A fresh copy containing only this repository's code and the raw HI-Small files was run from
  scratch (prepare, snapshots, `ibm_aml_models.py --seed 0`). The decay-constant selection, the
  training-free monitors, the coverage statistics and 16 of the 18 models reproduced
  `results/HI-Small/` exactly, every metric on every split. The two MLPs given a structural block
  differed by at most 0.002 on ROC-AUC, PR-AUC, F1 and recall at 1% FPR (at most 0.01 on a
  per-typology recall); their large dense layers are sensitive to the order of floating-point
  sums across CPU threads. Run times: prepare 21 s, snapshots 87 s, the 18 models of one seed
  39 min.

## Licence

MIT. AMLworld is subject to the terms of its original release.
