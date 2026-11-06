import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.metrics import roc_auc_score

from iforest import IsolationForest


def data(seed=0):
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=(1000, 3))
    anomalies = rng.normal(scale=2.0, size=(30, 3))
    return np.vstack([normal, anomalies]), np.r_[np.zeros(1000), np.ones(30)]


def test_same_auc_as_sklearn():
    X, y = data()
    ours = np.mean([roc_auc_score(y, IsolationForest(random_state=s).fit(X).score(X)) for s in range(3)])
    theirs = np.mean([roc_auc_score(y, -SklearnIsolationForest(random_state=s).fit(X).score_samples(X))
                      for s in range(3)])
    assert abs(ours - theirs) < 0.02


def test_ranks_points_like_sklearn():
    X, _ = data()
    ours = IsolationForest(random_state=0).fit(X).score(X)
    ours_again = IsolationForest(random_state=1).fit(X).score(X)
    theirs = -SklearnIsolationForest(random_state=0).fit(X).score_samples(X)
    assert spearmanr(ours, theirs).statistic > spearmanr(ours, ours_again).statistic - 0.03
    assert spearmanr(ours, theirs).statistic > 0.9