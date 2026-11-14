"""C8: sub-sampling vs swamping/masking, score distributions, 2-D score map (paper Sec. 3, Fig. 3-4).

Same protocol as C3/C4: full data fitted and scored, labels used only to evaluate, t = 100.

Parts (run all, or pick with --parts):
  masking        Direct masking test. Every tree's sub-sample holds a random number k of frauds.
                 If frauds mask each other (Sec. 3), frauds should get LONGER paths in trees
                 whose sample holds more frauds. psi = 256, 2048, 8192; 5 seeds x 100 trees.
  fullsample     Paper Fig. 4: whole data set as the sample (psi = n) vs psi = 256.
                 Slow (~2 min per seed); rows are saved per seed, rerun with --seeds to add more.
  distributions  Fraud vs genuine score distributions at psi = 256 and 8192 (seed 0), and what
                 fixed cut-offs s >= 0.5 / 0.6 / 0.7 flag.
  contour        Score map of an iForest trained on the two most fraud-separating attributes.

    python experiments/cc_scores.py
    python experiments/cc_scores.py --parts fullsample --seeds 1 2
"""
import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from datasets import load_creditcard
from iforest import IsolationForest, c
from utils.metrics import fraud_metrics
from utils.plotting import draw_score_map, lattice, most_separating_pair

NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
N_TREES = 100
TABLES, FIGURES = "results/tables", "results/figures"


def part_masking(X, y, seeds):
    fraud = np.flatnonzero(y == 1)
    genuine = np.random.default_rng(123).choice(np.flatnonzero(y == 0), 5000, replace=False)
    rows = []
    for psi in [256, 2048, 8192]:
        for seed in seeds:
            forest = IsolationForest(n_trees=N_TREES, sample_size=psi, random_state=seed).fit(X)
            for t, (tree, idx) in enumerate(zip(forest.trees_, forest.sample_indices_)):
                in_sample = np.zeros(len(y), dtype=bool)
                in_sample[idx] = True
                f, g = fraud[~in_sample[fraud]], genuine[~in_sample[genuine]]   # unseen points only
                rows.append({
                    "psi": psi, "seed": seed, "tree": t,
                    "frauds_in_sample": int(y[idx].sum()),
                    "fraud_path": tree.path_length(X[f]).mean() / c(psi),      # in units of c(psi)
                    "genuine_path": tree.path_length(X[g]).mean() / c(psi),
                })
        print(f"masking psi={psi} done", flush=True)
    trees = pd.DataFrame(rows)
    trees["gap"] = trees.genuine_path - trees.fraud_path        # how much shorter fraud paths are

    summary = []
    for psi, g in trees.groupby("psi"):
        row = {"psi": psi, "trees": len(g), "mean_frauds_in_sample": g.frauds_in_sample.mean()}
        for col in ["fraud_path", "genuine_path", "gap"]:
            slope, se = ols_slope(g.frauds_in_sample.to_numpy(), g[col].to_numpy())
            row[f"{col}_slope_per_fraud"] = slope
            row[f"{col}_slope_95ci"] = 1.96 * se
        summary.append(row)
    summary = pd.DataFrame(summary).round(5)
    trees.to_csv(f"{TABLES}/cc_scores_masking_trees.csv", index=False)
    summary.to_csv(f"{TABLES}/cc_scores_masking_summary.csv", index=False)
    print(summary.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (psi, g) in zip(axes, trees.groupby("psi")):
        for col, color, label in [("fraud_path", "#c0392b", "frauds"), ("genuine_path", "#3b6ea8", "genuine")]:
            b = g.groupby("frauds_in_sample")[col].agg(["mean", "std", "size"])
            b = b[b["size"] >= 10]                                   # skip bins with very few trees
            ax.errorbar(b.index, b["mean"], yerr=1.96 * b["std"] / np.sqrt(b["size"]),
                        fmt="o-", capsize=3, color=color, label=label)
        ax.set(title=f"psi = {psi} (mean {g.frauds_in_sample.mean():.1f} frauds per sample)",
               xlabel="frauds in the tree's sub-sample")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("mean path length of unseen points / c(psi)")
    fig.suptitle("Masking test: do frauds get longer paths when the tree saw more frauds? (mean, 95% CI)")
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/cc_scores_masking.png", dpi=150)


def ols_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.ptp(x) == 0:
        return np.nan, np.nan
    xm = x - x.mean()
    slope = (xm * (y - y.mean())).sum() / (xm ** 2).sum()
    resid = y - y.mean() - slope * xm
    se = np.sqrt((resid ** 2).sum() / (len(x) - 2) / (xm ** 2).sum())
    return slope, se


def part_fullsample(X, y, seeds):
    path = f"{TABLES}/cc_scores_fullsample_runs.csv"
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["seed"])
    rows = []
    for seed in seeds:
        for label, psi in [("psi = 256", 256), ("whole data (psi = n)", len(X))]:
            t0 = time.perf_counter()
            forest = IsolationForest(n_trees=N_TREES, sample_size=psi, random_state=seed).fit(X)
            train_time = time.perf_counter() - t0
            s = forest.score(X)
            rows.append({"sample": label, "psi": psi, "seed": seed, **fraud_metrics(y, s),
                         "train_time": train_time})
        print(f"fullsample seed {seed} done", flush=True)
    runs = pd.concat([old[~old.seed.isin(seeds)], pd.DataFrame(rows)], ignore_index=True)
    runs = runs.sort_values(["psi", "seed"])
    runs.to_csv(path, index=False)
    summary = runs.drop(columns=["seed", "psi"]).groupby("sample").agg(["mean", "std"]).round(4)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.to_csv(f"{TABLES}/cc_scores_fullsample_summary.csv")
    print(summary.to_string())


def part_distributions(X, y):
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, psi in zip(axes, [256, 8192]):
        s = IsolationForest(n_trees=N_TREES, sample_size=psi, random_state=0).fit(X).score(X)
        bins = np.linspace(s.min(), s.max(), 80)
        ax.hist(s[y == 0], bins=bins, density=True, alpha=0.6, color="#3b6ea8", label="genuine")
        ax.hist(s[y == 1], bins=bins, density=True, alpha=0.6, color="#c0392b", label="fraud")
        for cut, style in [(0.5, ":"), (0.6, "--")]:
            ax.axvline(cut, ls=style, c="k", lw=0.8, label=f"s = {cut}")
        ax.set(yscale="log", xlabel="anomaly score s", ylabel="density (each class normalised)",
               title=f"psi = {psi}, seed 0")
        ax.legend(fontsize=8)
        for cut in [0.5, 0.6, 0.7]:
            flag = s >= cut
            rows.append({"psi": psi, "cutoff": cut, "share_flagged": flag.mean(),
                         "precision": y[flag].mean() if flag.any() else np.nan,
                         "recall": y[flag].sum() / y.sum()})
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/cc_scores_distributions.png", dpi=150)
    table = pd.DataFrame(rows).round(4)
    table.to_csv(f"{TABLES}/cc_scores_cutoffs.csv", index=False)
    print(table.to_string(index=False))


def part_contour(X, y):
    a, b, name_a, name_b = most_separating_pair(X, y, NAMES)
    X2 = X[:, [a, b]]
    forest = IsolationForest(n_trees=N_TREES, sample_size=256, random_state=0).fit(X2)
    s = forest.score(X2)
    gx, gy, grid = lattice(X2)
    m = fraud_metrics(y, s)
    fig, ax = plt.subplots(figsize=(7, 6))
    fill = draw_score_map(ax, gx, gy, forest.score(grid), X2, y,
                          f"iForest on {name_a} and {name_b} only (psi=256, t=100): "
                          f"ROC-AUC {m['auc']:.3f}, AP {m['ap']:.3f}")
    fig.colorbar(fill, ax=ax, label="anomaly score s")
    ax.set(xlabel=name_a, ylabel=name_b)
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/cc_scores_contour.png", dpi=150)
    print(f"contour on {name_a}, {name_b}: AUC {m['auc']:.4f} AP {m['ap']:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parts", nargs="*", default=["masking", "fullsample", "distributions", "contour"])
    p.add_argument("--seeds", nargs="*", type=int, default=None,
                   help="masking: default 0-4; fullsample: default 0-2")
    args = p.parse_args()
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    X, y = load_creditcard()
    if "masking" in args.parts:
        part_masking(X, y, args.seeds or list(range(5)))
    if "fullsample" in args.parts:
        part_fullsample(X, y, args.seeds or list(range(3)))
    if "distributions" in args.parts:
        part_distributions(X, y)
    if "contour" in args.parts:
        part_contour(X, y)


if __name__ == "__main__":
    main()
