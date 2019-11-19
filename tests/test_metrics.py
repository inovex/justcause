import pytest

import numpy as np

from justcause.metrics import bias, enormse, mean_absolute, pehe_score


def test_pehe():
    one = np.repeat(1, 100)
    assert pehe_score(one, one) == 0

    zero = np.repeat(0, 100)
    assert pehe_score(one, zero) == 1


def test_enormse():
    one = np.repeat(1, 100)
    two = np.repeat(2, 100)
    four = np.repeat(4, 100)

    # approx required, because of delta and rouding
    assert enormse(one, two) == pytest.approx(enormse(two, four), 0.001)


def test_absolute_mean():
    one = np.repeat(1, 100)
    two = np.repeat(2, 100)

    assert mean_absolute(one, two) == 1


def test_bias():
    one = np.repeat(1, 100)
    two = np.repeat(2, 100)

    assert bias(one, two) == 1
    assert bias(two, one) == -1
