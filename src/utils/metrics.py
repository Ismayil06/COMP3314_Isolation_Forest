"""Fraud metrics on top of ROC-AUC (README §4, Metrics).

With 0.17 % fraud, ROC-AUC is dominated by the genuine class. These say what an analyst sees:
precision_at_k  share of fraud among the k highest scores (k = number of frauds: "R-precision")
recall_at_budget  share of all frauds caught when the top `budget` fraction is reviewed
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def top_k(scores, k):
    """Indices of the k highest scores (ties broken by position)."""
    return np.argsort(-np.asarray(scores), kind="stable")[:k]


def precision_at_k(y, scores, k):
    y = np.asarray(y)
    return float(y[top_k(scores, k)].mean())


def recall_at_budget(y, scores, budget):
    y = np.asarray(y)
    k = max(1, int(np.ceil(budget * len(y))))
    return float(y[top_k(scores, k)].sum() / y.sum())


def fraud_metrics(y, scores, budgets=(0.001, 0.005, 0.01)):
    y = np.asarray(y).astype(int)
    out = {"auc": roc_auc_score(y, scores), "ap": average_precision_score(y, scores),
           "precision_at_n_frauds": precision_at_k(y, scores, int(y.sum()))}
    for b in budgets:
        out[f"recall_at_{b:.1%}"] = recall_at_budget(y, scores, b)
    return out
