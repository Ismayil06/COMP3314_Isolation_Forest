"""SCiForest: Isolation Forest with a Split-selection Criterion.

Liu, Ting & Zhou, "On Detecting Clustered Anomalies Using SCiForest", ECML PKDD 2010.
Extension of our iForest (src/iforest): same API (fit, path_length, score), same c(n),
same sub-sampling without replacement. What changes, following the paper:

* Split = hyperplane  f(x) = sum_j c_j * x_j / sigma(X_j) - p  over q random attributes,
  c_j ~ U[-1, 1], sigma(X_j) computed on the node's data (Eq. 2).
* Of tau random hyperplanes, keep the one (and the split point p) with the highest
  Sdgain(Y) = (sigma(Y) - avg(sigma(Y_l), sigma(Y_r))) / sigma(Y)   (Eq. 3).
* No height limit: a node becomes a leaf when it holds <= 2 points (Algorithm 1, line 1).
* Acceptable range [-v, +v], v = max f - min f on the node's data: a scored point outside it
  gets no edge increment at that node (Algorithm 2).
"""
import numpy as np

from iforest.core import _check_data, c


def sdgain(Y):
    """Sdgain of every split position (Eq. 3).

    Y: (m,) or (m, k) projected values, one column per candidate hyperplane.
    Returns (Ys, gain): Ys = Y sorted down each column; gain[i, j] = Sdgain of putting the
    i+1 smallest values of column j on the left. Splits between equal values are impossible
    (the paper needs y_l < p <= y_r) and get -inf.
    """
    Y = np.asarray(Y, dtype=float)
    one_d = Y.ndim == 1
    if one_d:
        Y = Y[:, None]
    m = len(Y)
    Ys = np.sort(Y, axis=0)
    Yc = Ys - Ys.mean(axis=0)                 # centring keeps the prefix sums accurate
    s1, s2 = np.cumsum(Yc, axis=0), np.cumsum(Yc * Yc, axis=0)
    n_left = np.arange(1, m, dtype=float)[:, None]
    n_right = m - n_left
    sum_l, sq_l = s1[:-1], s2[:-1]
    sum_r, sq_r = s1[-1] - sum_l, s2[-1] - sq_l
    sd_l = np.sqrt(np.maximum(sq_l / n_left - (sum_l / n_left) ** 2, 0.0))
    sd_r = np.sqrt(np.maximum(sq_r / n_right - (sum_r / n_right) ** 2, 0.0))
    sd_all = np.sqrt(np.maximum(s2[-1] / m - (s1[-1] / m) ** 2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = (sd_all - 0.5 * (sd_l + sd_r)) / sd_all
    gain[~(Ys[1:] > Ys[:-1])] = -np.inf
    gain[:, ~(sd_all > 0)] = -np.inf
    return (Ys[:, 0], gain[:, 0]) if one_d else (Ys, gain)


class SCiTree:

    def __init__(self, n_attributes, n_candidates, rng, acceptable_range=True):
        self.n_attributes = n_attributes      # q
        self.n_candidates = n_candidates      # tau
        self.rng = rng
        self.acceptable_range = acceptable_range
        self.left = []
        self.right = []
        self.atts = []           # attribute indices of each node's hyperplane
        self.coefs = []          # c_j / sigma_j for those attributes
        self.split_value = []    # p
        self.lower = []          # acceptable range
        self.upper = []
        self.size = []
        self.depth = []

    def fit(self, X):
        # iterative instead of recursive: SCiTrees have no height limit and can be deep
        root = self._new_node(len(X), 0)
        stack = [(root, X)]
        while stack:
            node, Xn = stack.pop()
            if len(Xn) <= 2:
                continue
            split = self._best_split(Xn)
            if split is None:                   # all points identical: no split exists
                continue
            atts, coefs, p, f = split
            goes_left = f < 0
            self.atts[node], self.coefs[node], self.split_value[node] = atts, coefs, p
            if self.acceptable_range:
                v = f.max() - f.min()
                self.lower[node], self.upper[node] = -v, v
            d = self.depth[node] + 1
            self.left[node] = self._new_node(int(goes_left.sum()), d)
            self.right[node] = self._new_node(int((~goes_left).sum()), d)
            stack.append((self.left[node], Xn[goes_left]))
            stack.append((self.right[node], Xn[~goes_left]))
        self.left = np.array(self.left)
        self.right = np.array(self.right)
        self.split_value = np.array(self.split_value, dtype=float)
        self.lower = np.array(self.lower, dtype=float)
        self.upper = np.array(self.upper, dtype=float)
        self.size = np.array(self.size)
        self.depth = np.array(self.depth)
        return self

    def _new_node(self, size, depth):
        self.left.append(-1)
        self.right.append(-1)
        self.atts.append(None)
        self.coefs.append(None)
        self.split_value.append(np.nan)
        self.lower.append(-np.inf)
        self.upper.append(np.inf)
        self.size.append(size)
        self.depth.append(depth)
        return len(self.size) - 1

    def _best_split(self, X):
        sd = X.std(axis=0)
        can_use = np.flatnonzero(sd > 0)
        if len(can_use) == 0:
            return None
        q = min(self.n_attributes, len(can_use))
        # tau hyperplanes, each on q distinct attributes, coefficients ~ U[-1, 1]
        pick = np.argsort(self.rng.random((self.n_candidates, len(can_use))), axis=1)[:, :q]
        atts = can_use[pick]                                         # (tau, q)
        coefs = self.rng.uniform(-1.0, 1.0, size=atts.shape) / sd[atts]
        Y = np.einsum("mtq,tq->mt", X[:, atts], coefs)               # (m, tau) projections
        Ys, gain = sdgain(Y)
        best = int(np.argmax(gain))
        if not np.isfinite(gain.flat[best]):
            return None
        i, j = divmod(best, self.n_candidates)
        p = 0.5 * (Ys[i, j] + Ys[i + 1, j])                          # y_l < p <= y_r
        return atts[j], coefs[j], p, Y[:, j] - p

    def path_length(self, X):
        return self._path_length(np.ascontiguousarray(np.asarray(X, dtype=float).T))

    def _path_length(self, XT):
        # XT = X transposed (attributes as rows), so each gather reads one contiguous row
        n = XT.shape[1]
        h = np.zeros(n)
        stack = [(0, np.arange(n), np.zeros(n))]
        while stack:
            node, idx, e = stack.pop()
            if len(idx) == 0:
                continue
            if self.left[node] == -1:
                h[idx] = e + c(self.size[node])          # Algorithm 2, line 2
                continue
            y = -self.split_value[node]
            for a, w in zip(self.atts[node], self.coefs[node]):
                y = y + w * XT[a, idx]
            goes_right = y >= 0
            inside = np.where(goes_right, y < self.upper[node], y >= self.lower[node])
            e = e + inside                               # no increment outside the range
            stack.append((self.left[node], idx[~goes_right], e[~goes_right]))
            stack.append((self.right[node], idx[goes_right], e[goes_right]))
        return h


class SCiForest:
    def __init__(self, n_trees=100, sample_size=256, n_attributes=2, n_candidates=10,
                 acceptable_range=True, random_state=None):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.n_attributes = n_attributes          # q (paper default 2)
        self.n_candidates = n_candidates          # tau (paper default 10)
        self.acceptable_range = acceptable_range
        self.random_state = random_state

    def fit(self, X):
        X = _check_data(X)
        n = len(X)
        if n < 2:
            raise ValueError("Need at least 2 points to build a forest.")
        rng = np.random.default_rng(self.random_state)
        self.psi_ = min(self.sample_size, n)
        self.trees_ = []
        self.sample_indices_ = []
        for _ in range(self.n_trees):
            idx = rng.choice(n, size=self.psi_, replace=False)
            tree = SCiTree(self.n_attributes, self.n_candidates, rng, self.acceptable_range).fit(X[idx])
            self.trees_.append(tree)
            self.sample_indices_.append(idx)
        return self

    def path_length(self, X):
        XT = np.ascontiguousarray(_check_data(X).T)
        return np.mean([tree._path_length(XT) for tree in self.trees_], axis=0)

    def score(self, X):
        return 2.0 ** (-self.path_length(X) / c(self.psi_))

    def tree_stats(self, big_node=32):
        """Leaf depths and the share of splits (in nodes with >= big_node points) that cut off one point."""
        depths, single, big = [], 0, 0
        for t in self.trees_:
            leaf = t.left == -1
            depths.append(t.depth[leaf])
            internal = np.flatnonzero(~leaf & (t.size >= big_node))
            big += len(internal)
            single += int(np.sum((t.size[t.left[internal]] == 1) | (t.size[t.right[internal]] == 1)))
        depths = np.concatenate(depths)
        return {"mean_leaf_depth": float(depths.mean()), "max_leaf_depth": int(depths.max()),
                "single_point_split_share": single / big if big else float("nan")}
