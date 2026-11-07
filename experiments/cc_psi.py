"""C3: AUC and runtime vs sub-sampling size psi on the credit card data (paper Fig. 6)."""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, "src")
from datasets import load_creditcard
from iforest import IsolationForest

SEEDS = range(5)
N_TREES = 100
PSI_VALUES = [2 ** i for i in range(1, 16)]  # 2 ... 32768


def main():
    X, y = load_creditcard()
    rows = []
    for psi in PSI_VALUES:
        for seed in SEEDS:
            # build a fresh forest for each psi/seed, timing the build (train) and the scoring separately
            t0 = time.perf_counter()
            forest = IsolationForest(n_trees=N_TREES, sample_size=psi, random_state=seed).fit(X)
            train_time = time.perf_counter() - t0
            t0 = time.perf_counter()
            score = forest.score(X)
            score_time = time.perf_counter() - t0
            rows.append({
                "psi": psi,
                "seed": seed,
                "auc": roc_auc_score(y, score),
                "ap": average_precision_score(y, score),
                "train_time": train_time,
                "score_time": score_time,
            })
        print(f"psi={psi} done", flush=True)

    runs = pd.DataFrame(rows)
    summary = runs.drop(columns="seed").groupby("psi").agg(["mean", "std"]).round(4)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    runs.to_csv("results/tables/cc_psi_runs.csv", index=False)
    summary.to_csv("results/tables/cc_psi_summary.csv")
    print(summary.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    panels = [("auc", "ROC-AUC"), ("ap", "Average precision"), ("train_time", "train time (s)")]
    for ax, (col, label) in zip(axes, panels):
        m, s = summary[f"{col}_mean"], summary[f"{col}_std"]
        ax.plot(summary.index, m, marker="o")
        ax.fill_between(summary.index, m - s, m + s, alpha=0.25)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("sub-sampling size psi")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    fig.suptitle(f"iForest on credit card fraud, t={N_TREES}, mean +/- std over {len(SEEDS)} seeds")
    fig.tight_layout()
    fig.savefig("results/figures/cc_psi.png", dpi=150)


if __name__ == "__main__":
    main()
