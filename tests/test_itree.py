import numpy as np

from iforest import ITree


def grow(X, height_limit=100, seed=0):
    rng = np.random.default_rng(seed)
    return ITree(height_limit=height_limit, rng=rng).fit(np.asarray(X, dtype=float))


def external(tree):
    return tree.left == -1


def test_walkthrough_example_gives_a_full_tree():
    tree = grow([[1], [2], [3], [4], [10]])
    assert len(tree.size) == 9                 
    assert external(tree).sum() == 5           
    assert (~external(tree)).sum() == 4       
    assert (tree.size[external(tree)] == 1).all()


def test_full_tree_on_256_points_has_2psi_minus_1_nodes():
    X = np.random.default_rng(1).normal(size=(256, 3))
    tree = grow(X)
    assert len(tree.size) == 2 * 256 - 1       
    assert external(tree).sum() == 256
    assert (tree.size[external(tree)] == 1).all()


def test_height_limit_is_respected():
    X = np.random.default_rng(2).normal(size=(256, 3))
    tree = grow(X, height_limit=8)
    assert tree.depth.max() <= 8
    assert len(tree.size) <= 511
    assert tree.size[external(tree)].sum() == 256


def test_points_are_never_lost_or_duplicated():
    X = np.random.default_rng(3).normal(size=(200, 4))
    tree = grow(X, height_limit=6)
    assert tree.size[0] == 200                 # the root holds everything
    for node in np.flatnonzero(~external(tree)):
        assert tree.size[node] == tree.size[tree.left[node]] + tree.size[tree.right[node]]
    assert tree.size[external(tree)].sum() == 200


def test_internal_and_external_nodes_are_well_formed():
    X = np.random.default_rng(4).normal(size=(100, 2))
    tree = grow(X, height_limit=5)
    inner = ~external(tree)
    assert (tree.right[inner] >= 0).all()
    assert (tree.split_att[inner] >= 0).all()
    assert not np.isnan(tree.split_value[inner]).any()
    ext = external(tree)
    assert (tree.right[ext] == -1).all()
    assert (tree.split_att[ext] == -1).all()
    assert np.isnan(tree.split_value[ext]).all()


def test_children_are_one_level_deeper():
    X = np.random.default_rng(5).normal(size=(100, 2))
    tree = grow(X)
    for node in np.flatnonzero(~external(tree)):
        assert tree.depth[tree.left[node]] == tree.depth[node] + 1
        assert tree.depth[tree.right[node]] == tree.depth[node] + 1


def test_identical_points_form_a_single_external_node():
    tree = grow(np.ones((10, 3)))
    assert len(tree.size) == 1
    assert tree.size[0] == 10


def test_single_point_is_a_single_external_node():
    tree = grow([[5.0, 7.0]])
    assert len(tree.size) == 1
    assert tree.size[0] == 1


def test_constant_attribute_is_never_split_on():
    X = np.random.default_rng(6).normal(size=(100, 2))
    X[:, 0] = 3.0                             
    tree = grow(X)
    assert 0 not in tree.split_att


def test_split_values_lie_inside_the_data_range():
    X = np.random.default_rng(7).normal(size=(100, 3))
    tree = grow(X)
    for node in np.flatnonzero(~external(tree)):
        q = tree.split_att[node]
        assert X[:, q].min() <= tree.split_value[node] <= X[:, q].max()


def test_same_seed_gives_the_same_tree_and_different_seeds_differ():
    X = np.random.default_rng(8).normal(size=(50, 2))
    a, b, d = grow(X, seed=1), grow(X, seed=1), grow(X, seed=2)
    assert np.array_equal(a.split_value, b.split_value, equal_nan=True)
    assert not np.array_equal(a.split_value, d.split_value, equal_nan=True)