"""Extension: SCiForest (Liu, Ting & Zhou, ECML PKDD 2010) vs our iForest.

Same protocol as C3/C4: full data fitted and scored, t = 100, psi = 256, seeds 0-4,
labels used only to evaluate. SCiForest settings are the paper's: q = 2, tau = 10.

Parts (run all, or pick with --parts):
  synthetic  Sanity check of the claim SCiForest was built on: one dense anomaly cluster
             (Mulcross-like, SCiForest paper Fig. 6a) and scattered anomalies. 3 seeds.
  main       Credit card comparison. Rows are saved per seed; rerun with --seeds to add more.
  contour    Side-by-side score maps on the same two attributes as C8.

    python experiments/cc_sciforest.py
    python experiments/cc_sciforest.py --parts main --seeds 3 4
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
from sklearn.metrics import precision_recall_curve, roc_auc_score

sys.path.insert(0, "src")
from datasets import load_creditcard
from iforest import IsolationForest
from sciforest import SCiForest
from utils.metrics import fraud_metrics, top_k
from utils.plotting import draw_score_map, lattice, most_separating_pair

NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TABLES, FIGURES, LOGS = "results/tables", "results/figures", "results/logs"
COLORS = {"iForest": "#3b6ea8", "SCiForest": "#c0392b"}
MODELS = {"iForest": lambda seed: IsolationForest(n_trees=100, sample_size=256, random_state=seed),
          "SCiForest": lambda seed: SCiForest(n_trees=100, sample_size=256, n_attributes=2,
                                              n_candidates=10, random_state=seed)}


def tree_shape(forest, big_node=32):
    """Mean leaf depth and share of splits (nodes with >= big_node points) that cut off one point."""
    depths, single, big = [], 0, 0
    for t in forest.trees_:
        leaf = t.left == -1
        depths.append(t.depth[leaf])
        internal = np.flatnonzero(~leaf & (t.size >= big_node))
        big += len(internal)
        single += int(np.sum((t.size[t.left[internal]] == 1) | (t.size[t.right[internal]] == 1)))
    return {"mean_leaf_depth": float(np.concatenate(depths).mean()),
            "single_point_split_share": single / big if big else np.nan}


# --------------------------------------------------------------------------- synthetic
def synthetic_data(kind, a, seed):
    rng = np.random.default_rng(seed)
    n_a = int(round(a * 4096 / (1 - a)))
    normal = rng.normal(size=(4096, 2))
    if kind == "dense cluster":
        anomalies = np.full(2, 3 / np.sqrt(2)) + 0.15 * rng.normal(size=(n_a, 2))
    else:
        angle, radius = rng.uniform(0, 2 * np.pi, n_a), rng.uniform(3, 5, n_a)
        anomalies = np.c_[radius * np.cos(angle), radius * np.sin(angle)]
    return np.vstack([normal, anomalies]), np.r_[np.zeros(4096), np.ones(n_a)]


def part_synthetic():
    rows = []
    for kind, grid in [("dense cluster", [0.01, 0.05, 0.1, 0.2, 0.3]), ("scattered", [0.01, 0.05, 0.1])]:
        for a in grid:
            for seed in range(3):
                X, y = synthetic_data(kind, a, seed)
                for name, make in MODELS.items():
                    rows.append({"anomalies": kind, "contamination": a, "seed": seed, "method": name,
                                 "auc": roc_auc_score(y, make(seed).fit(X).score(X))})
    runs = pd.DataFrame(rows)
    summary = runs.groupby(["anomalies", "contamination", "method"]).auc.mean().unstack("method").round(3)
    summary.to_csv(f"{TABLES}/sciforest_synthetic_summary.csv")
    print(summary.to_string())
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name in MODELS:
        g = runs[(runs.anomalies == "dense cluster") & (runs.method == name)].groupby("contamination").auc.mean()
        ax.plot(g.index, g.values, "o-", color=COLORS[name], label=name)
    ax.axhline(0.5, ls=":", c="k", lw=0.8)
    ax.set(xlabel="contamination (one dense anomaly cluster)", ylabel="ROC-AUC", ylim=(0, 1.02),
           title="Synthetic sanity check (3 seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/sciforest_synthetic.png", dpi=150)


# --------------------------------------------------------------------------- credit card
def part_main(X, y, seeds):
    path = f"{TABLES}/cc_sciforest_runs.csv"
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["seed"])
    rows = []
    for seed in seeds:
        for name, make in MODELS.items():
            t0 = time.perf_counter()
            forest = make(seed).fit(X)
            fit_time = time.perf_counter() - t0
            t0 = time.perf_counter()
            s = forest.score(X)
            score_time = time.perf_counter() - t0
            k = int(np.ceil(0.001 * len(y)))
            rows.append({"method": name, "seed": seed, **fraud_metrics(y, s),
                         "genuine_share_of_top_0.1%": 1 - y[top_k(s, k)].mean(),
                         "frauds_per_sample": float(np.mean([y[i].sum() for i in forest.sample_indices_])),
                         **tree_shape(forest), "fit_time": fit_time, "score_time": score_time})
            if seed == 0:
                np.save(f"{LOGS}/cc_sciforest_scores_{name}_seed0.npy", s)
        print(f"main seed {seed} done", flush=True)
    runs = pd.concat([old[~old.seed.isin(seeds)], pd.DataFrame(rows)], ignore_index=True)
    runs = runs.sort_values(["method", "seed"])
    runs.to_csv(path, index=False)
    summarise(runs, y)


def summarise(runs, y):
    summary = runs.drop(columns="seed").groupby("method").agg(["mean", "std"]).round(4)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.to_csv(f"{TABLES}/cc_sciforest_summary.csv")
    print(summary.T.to_string())

    c3 = f"{TABLES}/cc_psi_runs.csv"                     # our iForest must reproduce C3 exactly
    if os.path.exists(c3):
        ref = pd.read_csv(c3).query("psi == 256").set_index("seed")
        mine = runs[runs.method == "iForest"].set_index("seed")
        common = mine.index.intersection(ref.index)
        same = np.allclose(mine.loc[common, "auc"], ref.loc[common, "auc"], atol=1e-6)
        print(f"iForest reproduces C3 (psi=256) on seeds {list(common)}: {same}")

    scores = {m: f"{LOGS}/cc_sciforest_scores_{m}_seed0.npy" for m in MODELS}
    if not all(os.path.exists(p) for p in scores.values()):
        return
    scores = {m: np.load(p) for m, p in scores.items()}
    k = int(np.ceil(0.005 * len(y)))                     # which frauds does each catch?
    hit = {m: np.isin(np.flatnonzero(y == 1), top_k(s, k)) for m, s in scores.items()}
    pd.DataFrame([{"budget": f"top 0.5% ({k} transactions)", "frauds": int(y.sum()),
                   "both": int((hit["iForest"] & hit["SCiForest"]).sum()),
                   "iForest only": int((hit["iForest"] & ~hit["SCiForest"]).sum()),
                   "SCiForest only": int((~hit["iForest"] & hit["SCiForest"]).sum()),
                   "neither": int((~hit["iForest"] & ~hit["SCiForest"]).sum())}]
                 ).to_csv(f"{TABLES}/cc_sciforest_overlap.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    budgets = np.linspace(0.0005, 0.05, 200)
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y, s)
        axes[0].plot(r, p, color=COLORS[name], label=name)
        order = np.argsort(-s, kind="stable")
        caught = np.cumsum(y[order])[np.ceil(budgets * len(y)).astype(int) - 1] / y.sum()
        axes[1].plot(budgets * 100, caught, color=COLORS[name], label=name)
    axes[0].axhline(y.mean(), ls=":", c="k", lw=0.8, label="random")
    axes[0].set(xlabel="recall", ylabel="precision", yscale="log", title="Precision-recall (seed 0)")
    axes[1].set(xlabel="% of transactions reviewed", ylabel="share of frauds caught", ylim=(0, 1),
                title="Recall vs review budget (seed 0)")
    cols = ["auc", "ap", "recall_at_1.0%"]
    x = np.arange(len(cols))
    for off, name in [(-0.18, "iForest"), (0.18, "SCiForest")]:
        g = runs[runs.method == name][cols]
        axes[2].bar(x + off, g.mean(), 0.36, yerr=g.std(), capsize=4, color=COLORS[name], label=name)
    axes[2].set_xticks(x, ["ROC-AUC", "AP", "recall @ 1%"])
    axes[2].set(title=f"Mean ± SD over {runs.seed.nunique()} seeds", ylim=(0, 1))
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("iForest vs SCiForest on credit card fraud (full data, t=100, psi=256)")
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/cc_sciforest_main.png", dpi=150)


def part_contour(X, y, seeds):
    """Both methods on the same two attributes (picked with labels, so both are optimistic;
    the comparison between them is still like for like)."""
    a, b, name_a, name_b = most_separating_pair(X, y, NAMES)
    X2 = X[:, [a, b]]
    rows, forests = [], {}
    for seed in seeds:
        for name, make in MODELS.items():
            forest = make(seed).fit(X2)
            rows.append({"attributes": f"{name_a}, {name_b}", "method": name, "seed": seed,
                         **fraud_metrics(y, forest.score(X2))})
            if seed == seeds[0]:
                forests[name] = forest
    runs = pd.DataFrame(rows)
    summary = runs.drop(columns=["seed"]).groupby(["attributes", "method"]).agg(["mean", "std"]).round(4)
    summary.columns = [f"{x}_{y}" for x, y in summary.columns]
    summary.to_csv(f"{TABLES}/cc_sciforest_2d_summary.csv")
    print(summary[["auc_mean", "auc_std", "ap_mean", "ap_std", "recall_at_1.0%_mean"]].to_string())

    gx, gy, grid = lattice(X2)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, (name, forest) in zip(axes, forests.items()):
        m = summary.loc[(f"{name_a}, {name_b}", name)]
        fill = draw_score_map(ax, gx, gy, forest.score(grid), X2, y,
                              f"{name} on {name_a}, {name_b} only: ROC-AUC {m['auc_mean']:.3f}, "
                              f"AP {m['ap_mean']:.3f} ({len(seeds)} seeds)")
        ax.set_xlabel(name_a)
        fig.colorbar(fill, ax=ax, label="anomaly score s")
    axes[0].set_ylabel(name_b)
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/cc_sciforest_contour.png", dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parts", nargs="*", default=["synthetic", "main", "contour"])
    p.add_argument("--seeds", nargs="*", type=int, default=list(range(5)))
    args = p.parse_args()
    for d in (TABLES, FIGURES, LOGS):
        os.makedirs(d, exist_ok=True)
    if "synthetic" in args.parts:
        part_synthetic()
    if {"main", "contour"} & set(args.parts):
        X, y = load_creditcard()
        if "main" in args.parts:
            part_main(X, y, args.seeds)
        if "contour" in args.parts:
            part_contour(X, y, args.seeds)


if __name__ == "__main__":
    main()
