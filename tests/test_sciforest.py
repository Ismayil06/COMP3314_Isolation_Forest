import numpy as np
from sklearn.metrics import roc_auc_score

from iforest import IsolationForest, c
from sciforest import SCiForest, SCiTree, sdgain


def dense_cluster(a=0.2, seed=0):
    """Gaussian normals plus one tight anomaly cluster 3 SD away (Mulcross-like, paper Fig. 6a)."""
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=(2000, 2))
    n_a = int(round(a * 2000 / (1 - a)))
    anomalies = np.full(2, 3 / np.sqrt(2)) + 0.15 * rng.normal(size=(n_a, 2))
    return np.vstack([normal, anomalies]), np.r_[np.zeros(2000), np.ones(n_a)]


def scattered(seed=0):
    rng = np.random.default_rng(seed)
    angle, radius = rng.uniform(0, 2 * np.pi, 60), rng.uniform(3, 5, 60)
    anomalies = np.c_[radius * np.cos(angle), radius * np.sin(angle)]
    return np.vstack([rng.normal(size=(2000, 2)), anomalies]), np.r_[np.zeros(2000), np.ones(60)]


def grown_tree(X, seed=0, **kw):
    return SCiTree(n_attributes=2, n_candidates=10, rng=np.random.default_rng(seed), **kw).fit(X)


def test_sdgain_matches_its_definition():
    y = np.random.default_rng(0).normal(size=25)
    ys, gain = sdgain(y)
    for i in range(len(y) - 1):
        left, right = ys[:i + 1], ys[i + 1:]
        expected = (ys.std() - 0.5 * (left.std() + right.std())) / ys.std()
        assert np.isclose(gain[i], expected)


def test_sdgain_never_splits_between_equal_values():
    _, gain = sdgain(np.array([1.0, 1.0, 2.0, 2.0]))
    assert np.isinf(gain[0]) and np.isinf(gain[2])
    assert np.isfinite(gain[1])


def test_sdgain_separates_two_groups():
    y = np.r_[np.random.default_rng(1).normal(0, 0.1, 50), np.random.default_rng(2).normal(5, 0.1, 10)]
    ys, gain = sdgain(y)
    best = int(np.argmax(gain))
    assert best == 49                                  # cut exactly between the two groups
    assert gain[best] > 0.9


def test_points_are_never_lost_or_duplicated():
    tree = grown_tree(np.random.default_rng(3).normal(size=(256, 4)))
    internal = np.flatnonzero(tree.left != -1)
    assert np.all(tree.size[internal] == tree.size[tree.left[internal]] + tree.size[tree.right[internal]])
    assert tree.size[tree.left == -1].sum() == 256


def test_leaves_hold_at_most_two_distinct_points():
    tree = grown_tree(np.random.default_rng(4).normal(size=(256, 3)))
    assert tree.size[tree.left == -1].max() <= 2       # Algorithm 1: grow until |X| <= 2


def test_identical_points_form_a_single_external_node():
    tree = grown_tree(np.ones((50, 3)))
    assert len(tree.size) == 1 and tree.size[0] == 50


def test_hyperplanes_use_q_distinct_attributes():
    tree = grown_tree(np.random.default_rng(5).normal(size=(256, 6)))
    for node in np.flatnonzero(tree.left != -1):
        assert len(tree.atts[node]) == 2
        assert len(set(tree.atts[node])) == 2


def test_training_points_always_fall_inside_the_acceptable_range():
    X = np.random.default_rng(6).normal(size=(256, 3))
    with_range = grown_tree(X, seed=7)
    without = grown_tree(X, seed=7, acceptable_range=False)
    assert np.allclose(with_range.path_length(X), without.path_length(X))


def test_acceptable_range_shortens_the_path_of_a_far_away_point():
    X = np.random.default_rng(8).normal(size=(256, 2))
    far = np.array([[60.0, -60.0]])
    with_range = grown_tree(X, seed=9).path_length(far)[0]
    without = grown_tree(X, seed=9, acceptable_range=False).path_length(far)[0]
    assert with_range < without


def test_scores_lie_between_0_and_1_and_follow_equation_2():
    X = np.random.default_rng(10).normal(size=(500, 3))
    forest = SCiForest(n_trees=20, random_state=0).fit(X)
    s = forest.score(X)
    assert np.all((s > 0) & (s <= 1))
    assert np.allclose(s, 2.0 ** (-forest.path_length(X) / c(forest.psi_)))


def test_same_seed_same_scores_different_seed_different_scores():
    X = np.random.default_rng(11).normal(size=(400, 3))
    a = SCiForest(n_trees=10, random_state=1).fit(X).score(X)
    b = SCiForest(n_trees=10, random_state=1).fit(X).score(X)
    d = SCiForest(n_trees=10, random_state=2).fit(X).score(X)
    assert np.array_equal(a, b) and not np.array_equal(a, d)


def test_beats_iforest_on_a_dense_anomaly_cluster():
    X, y = dense_cluster(a=0.2)
    sci = roc_auc_score(y, SCiForest(random_state=0).fit(X).score(X))
    ifo = roc_auc_score(y, IsolationForest(random_state=0).fit(X).score(X))
    assert sci > 0.85 and sci > ifo + 0.2               # SCiForest paper, Fig. 6(a)


def test_matches_iforest_on_scattered_anomalies():
    X, y = scattered()
    assert roc_auc_score(y, SCiForest(random_state=0).fit(X).score(X)) > 0.95
    assert roc_auc_score(y, IsolationForest(random_state=0).fit(X).score(X)) > 0.95
