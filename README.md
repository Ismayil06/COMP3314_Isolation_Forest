# Credit Card Fraud Detection with Isolation Forest

> COMP3314 Machine Learning — Course Project
>
> **Task:** Unsupervised credit card fraud detection
> **Method:** *Isolation Forest* (Liu, Ting & Zhou, ICDM 2008), implemented from scratch
>
> **Status:** 🚧 Workspace set up — implementation not started.

---

## 1. Project Overview

Credit card fraud is a textbook "few and different" anomaly problem: fraudulent transactions are
extremely rare (0.17% in our data set) and tend to have unusual feature values. This is exactly
the assumption behind Isolation Forest (iForest), which isolates anomalies with random binary
trees instead of modelling normal behaviour.

**Goal:** Implement iForest from scratch, apply it to the Credit Card Fraud Detection data set,
evaluate it as an unsupervised detector, compare it with baselines, and analyse its behaviour
(sub-sampling size, number of trees, training on normal transactions only, feature selection).

The experiments follow the evaluation methodology of the original paper (Sec. 5: baseline
comparison, number of trees, sub-sampling size, high-dimensional feature selection, normal-only
training), but are run on the fraud data. Our results are compared with the trends and claims
reported in the paper, and discrepancies are discussed.

---

## 2. Paper Details

| Field     | Value |
|-----------|-------|
| Title     | Isolation Forest |
| Authors   | Fei Tony Liu, Kai Ming Ting (Monash University), Zhi-Hua Zhou (Nanjing University) |
| Venue     | 2008 Eighth IEEE International Conference on Data Mining (ICDM 2008), pp. 413–422 |
| DOI       | [10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17) |
| Local PDF | `paper/Isolation_Forest.pdf` (git-ignored, not redistributed) |

**Summary:** iForest builds an ensemble of isolation trees (iTrees) on small random sub-samples.
Anomalies have short average path lengths. Anomaly score:
`s(x, n) = 2^(-E[h(x)] / c(n))`, with `c(n) = 2H(n-1) - 2(n-1)/n`.

**Paper defaults:** `t = 100` trees, sub-sampling size `ψ = 256`, height limit `l = ceil(log2 ψ)`,
sub-sampling without replacement.

**Paper claims we test on fraud data:**
1. iForest outperforms distance/density-based detectors (ORCA, LOF) in AUC and time, especially on large data (Table 3).
2. AUC converges at a small number of trees, t ≤ ~100 (Fig. 5).
3. A small sub-sample (ψ ≈ 128–512) is near-optimal; larger ψ adds time but no accuracy (Fig. 6, Sec. 6).
4. With many irrelevant attributes, Kurtosis-based subspace selection helps (Fig. 7).
5. Training on normal instances only causes only a small AUC drop, recoverable with larger ψ (Sec. 5.4).
6. Swamping/masking is reduced by sub-sampling (Sec. 3, Fig. 4).

- [ ] TODO: Check whether the authors released any official code (e.g. an R package) and note it here.
      State clearly that **this repository implements the algorithm from scratch**.

---

## 3. Data Set

| Property | Value |
|----------|-------|
| Name | Credit Card Fraud Detection (ULB Machine Learning Group / Worldline) |
| Source | [Kaggle: mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Local file | `data/raw/creditcard.csv` (≈144 MB, git-ignored) |
| Transactions | 284,807 (European cardholders, 2 days in September 2013) |
| Frauds | 492 (**0.172%**) |
| Features | 30: `Time`, `V1`–`V28` (PCA-transformed, anonymised), `Amount` |
| Label | `Class` (1 = fraud, 0 = genuine), used **only for evaluation** |

- [ ] TODO: Add download instructions for a fresh clone (Kaggle CLI or manual download → `data/raw/`).
- [ ] TODO: EDA notebook in `notebooks/`: class balance, `Amount`/`Time` distributions, per-feature
      kurtosis, fraud vs genuine feature distributions, duplicate rows.
- [ ] TODO: Decide preprocessing: drop or transform `Time` (e.g. hour-of-day), scale/log `Amount`
      (iForest is scale-invariant per feature, but baselines like LOF are not).
- [ ] TODO: Decide evaluation protocol:
  - (a) Unsupervised: fit and score on the full data (paper setting, Table 3).
  - (b) Train/test split: stratified random split **and** a time-based split (first ~70% of `Time` → train), which is more realistic.
  - (c) Normal-only training (paper Sec. 5.4): train only on genuine transactions.
- [ ] TODO: Cite the data set as its license requires (Dal Pozzolo et al., 2015; see §10).

---

## 4. Scope of Work

| # | Experiment | Paper reference | Priority | Output | Status |
|---|-----------|-----------------|----------|--------|--------|
| C0 | From-scratch iForest: iTree, iForest, PathLength, score (+ unit tests) | Alg. 1–3, Eq. 1–2 | P1 | `src/iforest/` | ☐ |
| C1 | iForest (ours) on creditcard with paper defaults (t=100, ψ=256) | Table 3 | P1 | `results/tables/cc_main_*` | ☐ |
| C2 | Baselines: sklearn iForest, LOF (k=10), ORCA-style k-NN (k=5, N=n/8), (opt.) One-Class SVM / autoencoder | Table 3 | P1 | `results/tables/cc_baselines_*` | ☐ |
| C3 | AUC & runtime vs sub-sampling size ψ = 2 … 32768 | Fig. 6, Sec. 6 | P1 | `results/figures/cc_psi*` | ☑ |
| C4 | AUC vs number of trees t | Fig. 5 | P1 | `results/figures/cc_trees*` | ☑ |
| C5 | Train on genuine transactions only vs contaminated training data (10 random splits, larger ψ to recover AUC) | Sec. 5.4 | P2 | `results/tables/cc_normal_only_*` | ☐ |
| C6 | Kurtosis-based feature subspace selection (+ optionally 506 added uniform-noise features, as in the paper) | Fig. 7, Sec. 5.3 | P2 | `results/figures/cc_kurtosis_*` | ☐ |
| C7 | Anomaly-score threshold analysis: precision / recall / F1, alerts per 1,000 transactions | — | P2 | `results/figures/cc_threshold_*` | ☐ |
| C8 | Sub-sampling vs swamping/masking: AUC on full data vs sub-sample; score distributions of fraud vs genuine; 2-D score contour on top features | Sec. 3, Fig. 3–4 | P3 | `results/figures/cc_scores_*` | ☐ |
| C9 | Path-length convergence illustration: a genuine vs a fraud transaction as t grows | Fig. 1 | P3 | `results/figures/cc_pathlen_*` | ☐ |
| C10 | (Optional) Supervised reference (Logistic Regression / Random Forest) to show the label-free trade-off | — | P3 | `results/tables/cc_supervised_ref_*` | ☐ |

**Metrics.** ROC-AUC is the metric the paper uses, but with 0.17% fraud it can look misleadingly
good. Also report:
- [ ] Average Precision (area under the precision–recall curve)
- [ ] Precision@k / Recall@k (e.g. k = 492, top 0.1%, top 1%)
- [ ] Recall at a fixed false-positive rate (e.g. 0.1%, 1%)
- [ ] Train / scoring time (compare relative speed, not absolute times, with the paper)
- [ ] Mean ± std over several random seeds

---

## 5. Repository Structure

```
.
├── README.md               # This file: overview, setup, results, TODOs
├── .gitignore
├── paper/                  # Local copy of the paper PDF (git-ignored)
├── data/
│   ├── raw/                # creditcard.csv (git-ignored)
│   └── processed/          # Cleaned features + labels, train/test splits (git-ignored)
├── src/
│   ├── iforest/            # ★ From-scratch iTree / iForest / path length / anomaly score
│   ├── baselines/          # ORCA-style k-NN, LOF, (opt.) OCSVM / autoencoder / supervised ref
│   ├── datasets/           # creditcard loader, preprocessing, splits
│   └── utils/              # Metrics (ROC-AUC, AP, precision@k), timing, seeding, plotting
├── experiments/            # One entry-point script per experiment (cc_*)
├── tests/                  # Unit tests (c(n), score bounds, tree properties, sklearn sanity check)
├── notebooks/              # EDA of creditcard.csv, figure prototyping
├── results/
│   ├── tables/             # Final CSV / Markdown tables
│   ├── figures/            # Result figures
│   └── logs/               # Raw run logs (git-ignored)
└── report/                 # Write-up: scope, results, comparison with paper, discussion
```

---

## 6. Setup

- [ ] TODO: Choose Python version (e.g. 3.10+) and document it here.
- [ ] TODO: Create `requirements.txt` (expected: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `pytest`, possibly `kaggle`).
- [ ] TODO: Write setup commands, e.g.
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- [ ] TODO: Document data download steps.

---

## 7. How to Run

C3 and C4 (run from the repo root, needs `data/raw/creditcard.csv`):
```bash
python experiments/cc_trees.py   # C4, about 2 min
python experiments/cc_psi.py     # C3
```

- [ ] TODO: Fill in once the other experiment scripts exist. Planned interface:
  ```bash
  python experiments/<experiment_name>.py   # writes to results/tables and results/figures
  ```
- [ ] TODO: Fix and document random seeds; report mean ± std over multiple runs.

---

## 8. Results (to be filled)

### 8.1 Method comparison on credit card fraud

| Method | ROC-AUC | Avg. Precision | Precision@492 | Recall @ 1% FPR | Train time (s) | Score time (s) |
|--------|--------:|---------------:|--------------:|----------------:|---------------:|---------------:|
| iForest (ours, t=100, ψ=256) | | | | | | |
| iForest (sklearn, same params) | | | | | | |
| LOF (k=10) | | | | | | |
| ORCA-style k-NN (k=5) | | | | | | |
| (opt.) One-Class SVM | | | | | | |
| (opt.) Supervised reference | | | | | | |

### 8.2 Comparison with the paper's claims

Paper numbers are on other data sets, so the comparison is about **trends**, not exact values.

| Paper claim | Paper value | Our finding on creditcard | Agrees? |
|-------------|-------------|---------------------------|---------|
| AUC converges by t ≈ 100 (Fig. 5) | converged well before t=100 | AUC 0.848 (t=1), 0.930 (t=10), 0.947 (t=50), 0.949 (t=100), 0.950 (t=300) | Yes: flat after ~50–100 trees |
| ψ near-optimal at 128–512 (Fig. 6) | Http ψ=128, ForestCover ψ=512 | AUC plateaus at ~0.95 from ψ≈32 up to 32768; but average precision keeps rising until ψ≈2048 (0.12 at ψ=128, 0.24 at ψ=2048) | AUC: yes. AP: partly, larger ψ still helps |
| Time grows slowly with ψ (Sec. 6) | 64× ψ → 1.6× time | ψ=256 → 16384 (64×): train+score 5.2 s → 15.0 s (2.9×); training alone 0.26 s → 7.7 s (~29×) | Partly: total time grows slowly because scoring 284k rows dominates, but training time grows almost linearly in ψ |
| iForest faster than LOF / ORCA (Table 3) | orders of magnitude on large data | | |
| Normal-only training small AUC drop (Sec. 5.4) | Http 0.9997→0.9919; ForestCover 0.8817→0.8802 | | |
| Larger ψ restores normal-only AUC (Sec. 5.4) | Http 0.9997 at ψ=8192 | | |
| Kurtosis selection helps with noise features (Fig. 7) | AUC peaks near original attribute count | | |
| Sub-sampling reduces masking/swamping (Fig. 4) | Mulcross AUC 0.67 → 0.91 | | |

### 8.3 Sensitivity plots, normal-only training, Kurtosis, thresholds

- [ ] TODO: Embed figures (ψ, t, Kurtosis, threshold, score distributions) with short commentary.
- [ ] TODO: Discuss what a fraud analyst would see: how many frauds are caught in the top-N alerts.

---

## 9. Discrepancies & Challenges (to be filled)

Fraud data:
- [ ] Extreme class imbalance (0.17%): ROC-AUC alone is not enough, so justify the extra metrics.
- [ ] `V1`–`V28` are already PCA components, so the original features can't be interpreted, and PCA may hide axis-aligned anomalies that iForest relies on.
- [ ] `Time` is seconds since the first transaction, not a timestamp: decide whether it leaks or adds signal.
- [ ] Some frauds may look like normal transactions ("not different"), which breaks iForest's core assumption. Analyse missed frauds.
- [ ] Duplicate transactions in the data: decide how to handle them.

Comparison with the paper:
- [ ] The paper's numbers come from other data sets, so only trends can be compared.
- [ ] ORCA's original C++ binary is not easily available → ORCA-style k-NN or a documented substitute.
- [ ] LOF (k=10) is slow on 284k points (paper reports NA for its large data sets): may need subsampling or a time cap.
- [ ] Paper does not specify number of repeated runs or seeds.
- [ ] Algorithm 2 split value: handle the degenerate case where all values of an attribute are equal.
- [ ] Kurtosis subspace selection (Sec. 5.3): exact subspace sizes and tie-breaking not fully specified.
- [ ] Hardware / language differences make absolute running times incomparable.
- C3/C4 setup: full data scored (no train/test split), 5 seeds (0–4), t=100 for C3 and ψ=256 for C4, ψ swept from 2 to 32768. ψ=2 gives AUC 0.5 because every tree has a single split (scores are almost all equal).
- Our C3 timings come from one machine and a pure-Python implementation, so only the trend is comparable with the paper.

---

## 10. Own Contributions vs Reused Code

- [ ] TODO: Keep this list accurate as the project develops.

| Component | Source |
|-----------|--------|
| iTree, iForest, PathLength, anomaly score (`src/iforest/`) | **Own implementation from scratch** (following Algorithms 1–3 of the paper) |
| Kurtosis attribute selection | **Own implementation** |
| Credit card preprocessing, evaluation protocol, fraud metrics analysis | **Own work** |
| ORCA-style k-NN baseline | TODO: own / third-party |
| LOF baseline | TODO: likely `scikit-learn` `LocalOutlierFactor` (reused) |
| Reference iForest for sanity checks | `scikit-learn` `IsolationForest` (reused, comparison only) |
| Metrics (ROC-AUC, AP) | `sklearn.metrics` (reused) |
| Credit card loader (`src/datasets/creditcard.py`) | Own code (uses `pandas`) |
| C3/C4 experiment scripts (`experiments/cc_psi.py`, `cc_trees.py`) | Own code (plots with `matplotlib`) |

---

## 11. References & Acknowledgements

- F. T. Liu, K. M. Ting, Z.-H. Zhou. *Isolation Forest.* ICDM 2008, pp. 413–422.
- A. Dal Pozzolo, O. Caelen, R. A. Johnson, G. Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE SSCI 2015. (Credit card data set — see the Kaggle page for the full list of citations requested.)
- S. D. Bay, M. Schwabacher. *Mining distance-based outliers in near linear time with randomization and a simple pruning rule.* KDD 2003. (ORCA)
- M. M. Breunig et al. *LOF: identifying density-based local outliers.* SIGMOD 2000.
- [ ] TODO: Add citations for any library or code reused (scikit-learn, ...).

---

## 12. Master TODO Checklist

### Phase 0 — Setup
- [x] Create folder structure and README
- [x] Obtain `creditcard.csv` → `data/raw/`
- [ ] Create a public GitHub repository and push
- [ ] Add `requirements.txt` and environment instructions
- [ ] Add a LICENSE (e.g. MIT) for own code

### Phase 1 — Core algorithm (`src/iforest/`)
- [ ] `c(n)` average path length (Eq. 1), incl. edge cases n ≤ 1, n = 2
- [ ] iTree construction (Algorithm 2) with height limit
- [ ] iForest training (Algorithm 1) with sub-sampling without replacement
- [ ] PathLength with `c(Size)` adjustment at external nodes (Algorithm 3)
- [ ] Anomaly score `s(x, ψ)` (Eq. 2)
- [ ] Unit tests in `tests/` and sanity comparison against `sklearn.ensemble.IsolationForest`
- [ ] Speed up scoring (vectorisation) — required for 284k transactions

### Phase 2 — Data (`src/datasets/`, `notebooks/`)
- [ ] Loader + preprocessing for `creditcard.csv`
- [ ] EDA notebook
- [ ] Define splits: full unsupervised, stratified, time-based, normal-only training

### Phase 3 — Baselines (`src/baselines/`)
- [ ] LOF (k=10)
- [ ] ORCA-style distance-based detector (k=5, N=n/8)
- [ ] (Optional) One-Class SVM, autoencoder, supervised reference model

### Phase 4 — Experiments (`experiments/cc_*`)
- [ ] C1 main result, C2 baselines
- [x] C3 ψ sensitivity, C4 number of trees
- [ ] C5 normal-only training
- [ ] C6 Kurtosis feature selection
- [ ] C7 threshold / alert-budget analysis
- [ ] C8 score visualisations / swamping-masking, C9 path-length convergence

### Phase 5 — Analysis & Documentation
- [ ] Fill §8 Results with tables and embedded figures
- [ ] Fill §8.2 comparison with the paper's claims
- [ ] Fill §9 Discrepancies with concrete numbers and explanations
- [ ] Error analysis: which frauds iForest misses, and why
- [ ] Extra ideas: confidence intervals over seeds; effect of height limit; Extended Isolation Forest as an extension
- [ ] Write report in `report/`
- [ ] Final check: README run instructions work from a fresh clone
