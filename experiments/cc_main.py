"""C1: iForest (ours) on credit card fraud with the paper defaults (t=100, psi=256)."""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

sys.path.insert(0, "src")
from datasets import load_creditcard
from iforest import IsolationForest
from utils.metrics import fraud_metrics

N_TREES = 100
SAMPLE_SIZE = 256
SEEDS = range(5)
FPR_TARGETS = (0.001, 0.01)


def recall_at_fpr(y, scores, fpr_target):
    """Recall at the highest threshold whose false-positive rate is still <= fpr_target."""
    fpr, tpr, _ = roc_curve(y, scores)
    ok = np.where(fpr <= fpr_target)[0]
    return float(tpr[ok[-1]]) if len(ok) else 0.0


def main():
    X, y = load_creditcard()
    print(f"creditcard: {len(y)} transactions, {int(y.sum())} frauds ({y.mean():.3%})", flush=True)

    rows, scored = [], []
    for seed in SEEDS:
        t0 = time.perf_counter()
        forest = IsolationForest(n_trees=N_TREES, sample_size=SAMPLE_SIZE, random_state=seed).fit(X)
        train_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        scores = forest.score(X)
        score_time = time.perf_counter() - t0

        row = {"seed": seed, "train_time": train_time, "score_time": score_time}
        row.update(fraud_metrics(y, scores))
        for fpr in FPR_TARGETS:
            row[f"recall_at_fpr_{fpr:.1%}"] = recall_at_fpr(y, scores, fpr)
        rows.append(row)
        scored.append((seed, scores))
        print(
            f"seed {seed}: AUC {row['auc']:.4f}  AP {row['ap']:.4f}  "
            f"P@492 {row['precision_at_n_frauds']:.4f}  "
            f"train {train_time:.2f}s  score {score_time:.2f}s",
            flush=True,
        )

    runs = pd.DataFrame(rows)
    summary = runs.drop(columns="seed").agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    runs.to_csv("results/tables/cc_main_runs.csv", index=False)
    summary.round(4).to_csv("results/tables/cc_main_summary.csv")
    print(f"\niForest (ours), t={N_TREES}, psi={SAMPLE_SIZE}, {len(list(SEEDS))} seeds")
    print(summary.round(4).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for seed, scores in scored:
        fpr, tpr, _ = roc_curve(y, scores)
        axes[0].plot(fpr, tpr, label=f"seed {seed}")
        precision, recall, _ = precision_recall_curve(y, scores)
        axes[1].plot(recall, precision, label=f"seed {seed}")
    axes[0].set_xlim(0, 0.05)
    axes[0].set(xlabel="false positive rate", ylabel="recall", title="ROC")
    axes[1].axhline(y.mean(), ls="--", c="gray", lw=0.8)
    axes[1].set(xlabel="recall", ylabel="precision", title="Precision-recall")
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"iForest on credit card fraud, t={N_TREES}, psi={SAMPLE_SIZE}")
    fig.tight_layout()
    fig.savefig("results/figures/cc_main.png", dpi=150)
    print("\nSaved results/tables/cc_main_*.csv and results/figures/cc_main.png")


if __name__ == "__main__":
    main()
