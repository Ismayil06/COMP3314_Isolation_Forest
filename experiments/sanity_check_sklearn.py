import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.metrics import roc_auc_score

from iforest import IsolationForest

SEEDS = range(5)


def make_datasets(rng):
    datasets = {}
    normal = rng.normal(size=(2000, 2))
    anomalies = rng.uniform(-4, 4, size=(5000, 2))
    anomalies = anomalies[np.abs(anomalies).max(axis=1) > 2.5][:40]
    datasets["gaussian_2d"] = (np.vstack([normal, anomalies]), np.r_[np.zeros(2000), np.ones(40)])
    normal = rng.normal(size=(2000, 10))
    anomalies = rng.normal(loc=0.0, scale=1.6, size=(40, 10))
    datasets["gaussian_10d"] = (np.vstack([normal, anomalies]), np.r_[np.zeros(2000), np.ones(40)])
    normal = np.vstack([rng.normal(-3, 1, size=(1000, 2)), rng.normal(3, 1, size=(1000, 2))])
    anomalies = rng.normal(loc=[0, 4], scale=0.3, size=(100, 2))
    datasets["clustered_anomalies"] = (np.vstack([normal, anomalies]), np.r_[np.zeros(2000), np.ones(100)])

    return datasets


def main():
    rows = []
    for name, (X, y) in make_datasets(np.random.default_rng(0)).items():
        for seed in SEEDS:
            ours = IsolationForest(n_trees=100, sample_size=256, random_state=seed).fit(X).score(X)
            sk = SklearnIsolationForest(n_estimators=100, max_samples=256, random_state=seed).fit(X)
            theirs = -sk.score_samples(X)          
            ours_again = IsolationForest(n_trees=100, sample_size=256, random_state=seed + 100).fit(X).score(X)
            rows.append({
                "dataset": name,
                "seed": seed,
                "auc_ours": roc_auc_score(y, ours),
                "auc_sklearn": roc_auc_score(y, theirs),
                "spearman_ours_vs_sklearn": spearmanr(ours, theirs).statistic,
                "spearman_ours_vs_ours": spearmanr(ours, ours_again).statistic,
            })

    results = pd.DataFrame(rows)
    summary = results.groupby("dataset").agg(
        auc_ours_mean=("auc_ours", "mean"), auc_ours_std=("auc_ours", "std"),
        auc_sklearn_mean=("auc_sklearn", "mean"), auc_sklearn_std=("auc_sklearn", "std"),
        spearman_ours_vs_sklearn=("spearman_ours_vs_sklearn", "mean"),
        spearman_ours_vs_ours=("spearman_ours_vs_ours", "mean"),
    ).round(4)

    os.makedirs("results/tables", exist_ok=True)
    results.to_csv("results/tables/sanity_check_sklearn_runs.csv", index=False)
    summary.to_csv("results/tables/sanity_check_sklearn_summary.csv")
    print(summary.to_string())
    print("\nSaved to results/tables/sanity_check_sklearn_*.csv")


if __name__ == "__main__":
    main()