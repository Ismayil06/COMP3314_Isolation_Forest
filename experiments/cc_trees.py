"""C4: AUC vs number of trees t on the credit card data (paper Fig. 5)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, "src")
from datasets import load_creditcard
from iforest import IsolationForest, c

SEEDS = range(5)
SAMPLE_SIZE = 256
T_MAX = 300
T_VALUES = [1, 2, 5, 10, 20, 50, 100, 200, 300]


def main():
    X, y = load_creditcard()
    rows = []
    for seed in SEEDS:
        forest = IsolationForest(n_trees=T_MAX, sample_size=SAMPLE_SIZE, random_state=seed).fit(X)
        # fit once and score with the first k trees, so no refitting for every t
        total = np.zeros(len(X))
        for k, tree in enumerate(forest.trees_, start=1):
            total += tree.path_length(X)
            if k in T_VALUES:
                score = 2.0 ** (-(total / k) / c(forest.psi_))
                rows.append({
                    "seed": seed,
                    "t": k,
                    "auc": roc_auc_score(y, score),
                    "ap": average_precision_score(y, score),
                })
        print(f"seed {seed} done", flush=True)

    runs = pd.DataFrame(rows)
    summary = runs.groupby("t").agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        ap_mean=("ap", "mean"), ap_std=("ap", "std"),
    ).round(4)

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    runs.to_csv("results/tables/cc_trees_runs.csv", index=False)
    summary.to_csv("results/tables/cc_trees_summary.csv")
    print(summary.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, label in zip(axes, ["auc", "ap"], ["ROC-AUC", "Average precision"]):
        m, s = summary[f"{col}_mean"], summary[f"{col}_std"]
        ax.plot(summary.index, m, marker="o")
        ax.fill_between(summary.index, m - s, m + s, alpha=0.25)
        ax.set_xscale("log")
        ax.set_xlabel("number of trees t")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    fig.suptitle(f"iForest on credit card fraud, psi={SAMPLE_SIZE}, mean +/- std over {len(SEEDS)} seeds")
    fig.tight_layout()
    fig.savefig("results/figures/cc_trees.png", dpi=150)


if __name__ == "__main__":
    main()
