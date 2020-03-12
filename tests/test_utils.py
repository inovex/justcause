import pytest

import numpy as np
from numpy.random import RandomState

from justcause.data.utils import to_rep_iter, to_rep_list
from justcause.learners.utils import replace_factual_outcomes
from justcause.utils import int_from_random_state


def test_iter_rep(dummy_df):
    assert "rep" in dummy_df.columns
    single_rep = next(to_rep_iter(dummy_df))
    assert "rep" not in single_rep.columns
    assert single_rep.shape[0] == 5


def test_to_rep_list(dummy_rep_df):
    replist = to_rep_list(dummy_rep_df)
    assert len(replist) == 5
    single_rep = replist[0]
    assert "rep" not in single_rep.columns
    assert single_rep.shape[0] == 10


def test_replace_factuals():
    y_0 = np.zeros(10)
    y_1 = np.ones(10)
    y = np.repeat(0.5, 10)
    t = np.zeros(10)
    t[5] = 1
    y_0, y_1 = replace_factual_outcomes(y_0, y_1, y, t)
    assert y_1[5] == y[5]
    assert y_1[0] != y[0]
    assert y_0[0] == y[0]


def test_int_from_random_state():
    rs = RandomState(5)
    rs_int = int_from_random_state(rs)
    assert isinstance(rs_int, int)

    rs = RandomState(5)
    rs_int_new = int_from_random_state(rs)
    # check if the same int is returned for the same RandomState
    assert rs_int == rs_int_new
    # check that integers are returned directly
    assert int_from_random_state(5) == 5

    with pytest.raises(ValueError):
        int_from_random_state("wrong-input")
