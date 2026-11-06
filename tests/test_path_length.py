import numpy as np
import pytest

from iforest import ITree, c, harmonic


def grow(X, height_limit=100, seed=0):
    rng = np.random.default_rng(seed)
    return ITree(height_limit=height_limit, rng=rng).fit(np.asarray(X, dtype=float))


def path_length_one_point(tree, x, node=0):
    if tree.left[node] == -1:
        return tree.depth[node] + c(tree.size[node])
    if x[tree.split_att[node]] < tree.split_value[node]:
        return path_length_one_point(tree, x, tree.left[node])
    return path_length_one_point(tree, x, tree.right[node])


WALKTHROUGH = np.array([[1], [2], [3], [4], [10]], dtype=float)


def test_walkthrough_example_path_lengths():
    tree = grow(WALKTHROUGH)
    assert list(tree.path_length(WALKTHROUGH)) == [2, 3, 4, 4, 1]


def test_new_points_follow_the_same_tests():
    tree = grow(WALKTHROUGH)
    new = np.array([[9.5], [0.0]])
    assert list(tree.path_length(new)) == [1, 2]  


def test_fast_version_matches_algorithm_3_point_by_point():
    X = np.random.default_rng(1).normal(size=(300, 3))
    tree = grow(X[:256], height_limit=8)
    fast = tree.path_length(X)
    slow = np.array([path_length_one_point(tree, x) for x in X])
    assert fast == pytest.approx(slow)


def test_height_limit_zero_gives_c_of_n_for_everyone():
    X = np.random.default_rng(2).normal(size=(50, 2))
    tree = grow(X, height_limit=0)
    assert tree.path_length(X) == pytest.approx(np.full(50, c(50)))


def test_height_limit_one_adds_c_of_each_side():
    X = np.random.default_rng(3).normal(size=(40, 1))
    tree = grow(X, height_limit=1)
    left, right = tree.left[0], tree.right[0]
    h = tree.path_length(X)
    goes_left = X[:, 0] < tree.split_value[0]
    assert h[goes_left] == pytest.approx(1 + c(tree.size[left]))
    assert h[~goes_left] == pytest.approx(1 + c(tree.size[right]))


def test_full_tree_path_length_is_just_the_depth():
    X = np.random.default_rng(4).normal(size=(64, 2))
    tree = grow(X)
    h = tree.path_length(X)
    assert np.all(h == np.round(h))
    assert h.sum() == tree.depth[tree.left == -1].sum()


def test_points_far_outside_the_training_range_still_get_a_path_length():
    X = np.random.default_rng(5).normal(size=(100, 2))
    tree = grow(X, height_limit=7)
    h = tree.path_length(np.array([[1e6, -1e6], [0.0, 0.0]]))
    assert len(h) == 2
    assert (h >= 1).all()


def test_average_path_matches_our_derivation_on_evenly_spaced_points():
    n = 20
    X = np.arange(n, dtype=float).reshape(-1, 1)
    trees = [grow(X, seed=s) for s in range(2000)]
    average = np.mean([t.path_length(X) for t in trees], axis=0)
    theory = np.array([harmonic(i - 1) + harmonic(n - i) for i in range(1, n + 1)])
    assert average == pytest.approx(theory, abs=0.15)
    assert average.mean() == pytest.approx(c(n), abs=0.05)