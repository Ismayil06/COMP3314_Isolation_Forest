import numpy as np

from utils.metrics import fraud_metrics, precision_at_k, recall_at_budget


def test_precision_at_k_counts_frauds_among_the_top_scores():
    y = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    s = np.array([.1, .9, .8, .2, 0, 0, 0, 0, 0, 0])
    assert precision_at_k(y, s, 2) == 0.5            # top 2 = rows 1 and 2, one is fraud


def test_recall_at_budget_reviews_the_top_fraction():
    y = np.r_[np.ones(2), np.zeros(98)]
    s = np.r_[[1.0, 0.0], np.linspace(0.5, 0.1, 98)]
    assert recall_at_budget(y, s, 0.01) == 0.5       # top 1 row: one of two frauds
    assert recall_at_budget(y, s, 1.0) == 1.0


def test_perfect_ranking_gives_perfect_metrics():
    y = np.r_[np.ones(5), np.zeros(995)]
    m = fraud_metrics(y, -np.arange(1000.0))
    assert m["auc"] == 1.0 and m["ap"] == 1.0 and m["precision_at_n_frauds"] == 1.0
    assert m["recall_at_1.0%"] == 1.0
