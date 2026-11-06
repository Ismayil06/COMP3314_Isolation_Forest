import numpy as np


def harmonic(i):
    if i < 1:
        return 0.0
    return float(np.sum(1.0 / np.arange(1, i + 1)))


def c(n):
    if n <= 1:
        return 0.0
    return 2.0 * harmonic(n - 1) - 2.0 * (n - 1) / n

class ITree:

    def __init__(self, height_limit, rng):
        self.height_limit = height_limit  
        self.rng = rng                    
        self.left = []        
        self.right = []        
        self.split_att = []   
        self.split_value = [] 
        self.size = []        
        self.depth = []       

    def fit(self, X):
        self._grow(X, depth=0)
        self.left = np.array(self.left)
        self.right = np.array(self.right)
        self.split_att = np.array(self.split_att)
        self.split_value = np.array(self.split_value, dtype=float)
        self.size = np.array(self.size)
        self.depth = np.array(self.depth)
        return self

    def _new_node(self, size, depth):
        self.left.append(-1)
        self.right.append(-1)
        self.split_att.append(-1)
        self.split_value.append(np.nan)
        self.size.append(size)
        self.depth.append(depth)
        return len(self.size) - 1

    def _grow(self, X, depth):
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
        if p <= mins[q]:    
            p = maxs[q]

        goes_left = X[:, q] < p
        self.split_att[node] = q
        self.split_value[node] = p
        self.left[node] = self._grow(X[goes_left], depth + 1)
        self.right[node] = self._grow(X[~goes_left], depth + 1)
        return node

    def path_length(self, X):
        h = np.zeros(len(X))
        stack = [(0, np.arange(len(X)))]         
        while stack:
            node, idx = stack.pop()
            if len(idx) == 0:
                continue
            if self.left[node] == -1:
                h[idx] = self.depth[node] + c(self.size[node])
            else:
                goes_left = X[idx, self.split_att[node]] < self.split_value[node]
                stack.append((self.left[node], idx[goes_left]))
                stack.append((self.right[node], idx[~goes_left]))
        return h