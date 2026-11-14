"""2-D anomaly-score maps (paper Fig. 3): score a lattice with a forest trained on two attributes."""
import numpy as np


def most_separating_pair(X, y, names):
    """The two attributes whose values alone rank frauds best (|ROC-AUC - 0.5| largest).

    Uses labels, so it only decides WHICH plane to draw; the forests drawn on it never see labels.
    """
    from sklearn.metrics import roc_auc_score
    strength = [abs(roc_auc_score(y, X[:, j]) - 0.5) for j in range(X.shape[1])]
    a, b = np.argsort(strength)[::-1][:2]
    return int(a), int(b), names[a], names[b]


def lattice(X2, n=200, q=(0.1, 99.9), pad=0.05):
    """An n x n grid covering the central 99.8 % of the data (extremes would squash the plot)."""
    lo, hi = np.percentile(X2, q[0], axis=0), np.percentile(X2, q[1], axis=0)
    span = hi - lo
    lo, hi = lo - pad * span, hi + pad * span
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n))
    return gx, gy, np.c_[gx.ravel(), gy.ravel()]


def draw_score_map(ax, gx, gy, s_grid, X2, y, title, levels=(0.5, 0.6, 0.7), n_genuine=3000, seed=0):
    """Filled score map, contour lines at `levels`, a sample of genuine points and all frauds."""
    s_grid = s_grid.reshape(gx.shape)
    fill = ax.contourf(gx, gy, s_grid, levels=20, cmap="YlOrRd", alpha=0.85)
    lines = ax.contour(gx, gy, s_grid, levels=[l for l in levels if s_grid.min() < l < s_grid.max()],
                       colors="k", linewidths=0.8)
    ax.clabel(lines, fmt="%.1f", fontsize=8)
    rng = np.random.default_rng(seed)
    genuine = rng.choice(np.flatnonzero(y == 0), size=min(n_genuine, int((y == 0).sum())), replace=False)
    ax.scatter(X2[genuine, 0], X2[genuine, 1], s=3, c="#3b6ea8", alpha=0.35, label="genuine (sample)")
    fraud = y == 1
    ax.scatter(X2[fraud, 0], X2[fraud, 1], s=9, facecolors="none", edgecolors="k", linewidths=0.6, label="fraud")
    ax.set_xlim(gx.min(), gx.max())
    ax.set_ylim(gy.min(), gy.max())
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right", fontsize=7)
    return fill
