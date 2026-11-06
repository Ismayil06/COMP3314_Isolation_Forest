import math

import pytest

from iforest import c, harmonic


def test_harmonic_small_values():
    assert harmonic(0) == 0.0
    assert harmonic(1) == 1.0
    assert harmonic(2) == pytest.approx(1.5)
    assert harmonic(3) == pytest.approx(11 / 6)


def test_c_is_zero_for_zero_or_one_point():
    assert c(0) == 0.0
    assert c(1) == 0.0


def test_c_matches_our_hand_calculations():
    assert c(2) == pytest.approx(1.0)
    assert c(3) == pytest.approx(5 / 3)
    assert c(4) == pytest.approx(13 / 6)
    assert c(5) == pytest.approx(77 / 30)


def test_c_satisfies_the_recurrence_we_derived():
    for n in range(1, 1000):
        assert c(n + 1) - c(n) == pytest.approx(2 / (n + 1), abs=1e-12)


def test_c_at_default_subsample_size():
    assert c(256) == pytest.approx(10.2487, abs=1e-3)


def test_c_grows_with_n():
    values = [c(n) for n in range(1, 500)]
    assert all(a < b for a, b in zip(values, values[1:]))


def test_paper_approximation_good_for_large_n_but_poor_for_small_n():
    def c_approx(n):
        return 2 * (math.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

    assert abs(c(256) - c_approx(256)) < 0.01
    assert abs(c(3) - c_approx(3)) > 0.4