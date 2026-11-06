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