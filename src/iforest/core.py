"""From-scratch Isolation Forest (Liu, Ting & Zhou, 2008), using NumPy only."""
import numpy as np


def harmonic(i):
    """Harmonic number H(i) = 1 + 1/2 + ... + 1/i, with H(0) = 0."""
    if i < 1:
        return 0.0
    return float(np.sum(1.0 / np.arange(1, i + 1)))


def c(n):
    """Average path length for n points: c(n) = 2H(n-1) - 2(n-1)/n  (paper, Eq. 1)."""
    if n <= 1:
        return 0.0
    return 2.0 * harmonic(n - 1) - 2.0 * (n - 1) / n

class ITree:
    """One isolation tree (paper, Algorithm 2), stored as flat lists of nodes."""

    def __init__(self, height_limit, rng):
        self.height_limit = height_limit  # l in the paper
        self.rng = rng                    # random number generator
        self.left = []         # number of the left child  (-1 = none)
        self.right = []        # number of the right child (-1 = none)
        self.split_att = []    # attribute q tested here   (-1 = external node)
        self.split_value = []  # split value p             (nan = external node)
        self.size = []         # how many training points reached this node
        self.depth = []        # how many edges from the root to this node

    def fit(self, X):
        """Grow the tree on data X (rows = points, columns = attributes)."""
        self._grow(X, depth=0)
        self.left = np.array(self.left)
        self.right = np.array(self.right)
        self.split_att = np.array(self.split_att)
        self.split_value = np.array(self.split_value, dtype=float)
        self.size = np.array(self.size)
        self.depth = np.array(self.depth)
        return self

    def _new_node(self, size, depth):
        """Add an empty node and return its number."""
        self.left.append(-1)
        self.right.append(-1)
        self.split_att.append(-1)
        self.split_value.append(np.nan)
        self.size.append(size)
        self.depth.append(depth)
        return len(self.size) - 1

    def _grow(self, X, depth):
        """Algorithm 2: build the subtree for data X at the given depth."""
        node = self._new_node(len(X), depth)

        if depth >= self.height_limit or len(X) <= 1:
            return node

        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        can_split = np.flatnonzero(maxs > mins)
        if len(can_split) == 0:
            return node
        q = int(self.rng.choice(can_split))
        p = self.rng.uniform(mins[q], maxs[q])
        if p <= mins[q]:      # probability ~0; keeps both sides non-empty
            p = maxs[q]

        goes_left = X[:, q] < p
        self.split_att[node] = q
        self.split_value[node] = p
        self.left[node] = self._grow(X[goes_left], depth + 1)
        self.right[node] = self._grow(X[~goes_left], depth + 1)
        return node