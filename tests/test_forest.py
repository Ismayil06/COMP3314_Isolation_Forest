import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from iforest import IsolationForest, c


def cluster_with_one_outlier(seed=0):
    X = np.random.default_rng(seed).normal(size=(500, 2))
    return np.vstack([X, [[8.0, 8.0]]])          


def test_forest_follows_algorithm_1():
    X = np.random.default_rng(0).normal(size=(1000, 3))
    forest = IsolationForest(n_trees=50, sample_size=256, random_state=0).fit(X)
    assert len(forest.trees_) == 50                      
    assert forest.psi_ == 256
    assert forest.height_limit_ == 8                          
    for tree in forest.trees_:
        assert tree.size[0] == 256                           
        assert tree.depth.max() <= 8


def test_subsamples_are_drawn_without_replacement():
    X = np.random.default_rng(1).normal(size=(1000, 2))
    forest = IsolationForest(n_trees=20, sample_size=256, random_state=1).fit(X)
    for idx in forest.sample_indices_:
        assert len(np.unique(idx)) == 256                     
        assert idx.min() >= 0 and idx.max() < 1000


def test_sample_size_larger_than_data_uses_all_points():
    X = np.random.default_rng(2).normal(size=(100, 2))
    forest = IsolationForest(n_trees=5, sample_size=256, random_state=2).fit(X)
    assert forest.psi_ == 100
    assert forest.height_limit_ == 7                         


def test_scores_lie_between_0_and_1():
    X = np.random.default_rng(3).normal(size=(500, 4))
    s = IsolationForest(random_state=3).fit(X).score(X)
    assert ((s > 0) & (s < 1)).all()


def test_score_is_equation_2():
    X = np.random.default_rng(4).normal(size=(300, 2))
    forest = IsolationForest(n_trees=30, random_state=4).fit(X)
    expected = 2.0 ** (-forest.path_length(X) / c(forest.psi_))
    assert forest.score(X) == pytest.approx(expected)


def test_average_path_equal_to_c_psi_gives_score_one_half():
    # Equation 2: E[h] = c(psi) gives exactly 2^-1 = 0.5
    assert 2.0 ** (-c(256) / c(256)) == 0.5


def test_obvious_outlier_gets_the_highest_score():
    X = cluster_with_one_outlier()
    forest = IsolationForest(random_state=5).fit(X)
    s = forest.score(X)
    assert np.argmax(s) == len(X) - 1                       
    assert s[-1] > 0.7
    assert s[:-1].mean() < 0.5


def test_outlier_has_a_much_shorter_path():
    X = cluster_with_one_outlier()
    h = IsolationForest(random_state=6).fit(X).path_length(X)
    assert h[-1] < 0.5 * h[:-1].mean()


def test_high_auc_on_easy_synthetic_data():
    rng = np.random.default_rng(7)
    normal = rng.normal(size=(1000, 2))
    anomalies = rng.uniform(-8, 8, size=(1000, 2))
    anomalies = anomalies[np.abs(anomalies).max(axis=1) > 5][:20]  
    X = np.vstack([normal, anomalies])
    y = np.r_[np.zeros(1000), np.ones(len(anomalies))]
    s = IsolationForest(random_state=7).fit(X).score(X)
    assert roc_auc_score(y, s) > 0.95


def test_more_trees_means_less_noise():
    X = np.random.default_rng(8).normal(size=(500, 2))
    point = X[:1]
    def spread(t):
        return np.std([IsolationForest(n_trees=t, random_state=s).fit(X).path_length(point)[0]
                       for s in range(30)])
    assert spread(100) < spread(1) / 5


def test_same_seed_same_scores_different_seed_different_scores():
    X = np.random.default_rng(9).normal(size=(200, 2))
    a = IsolationForest(n_trees=20, random_state=1).fit(X).score(X)
    b = IsolationForest(n_trees=20, random_state=1).fit(X).score(X)
    d = IsolationForest(n_trees=20, random_state=2).fit(X).score(X)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, d)


def test_new_points_can_be_scored_after_fitting():
    X = np.random.default_rng(10).normal(size=(300, 2))
    forest = IsolationForest(n_trees=50, random_state=10).fit(X)
    s = forest.score([[0.0, 0.0], [10.0, 10.0]])
    assert s[1] > s[0]                                        


def test_bad_input_is_rejected():
    with pytest.raises(ValueError):
        IsolationForest().fit([1.0, 2.0, 3.0])                
    with pytest.raises(ValueError):
        IsolationForest().fit([[1.0, np.nan], [2.0, 3.0]])    
    with pytest.raises(ValueError):
        IsolationForest().fit([[1.0, 2.0]])                  