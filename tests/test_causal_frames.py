import pytest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from justcause.data.frames import CausalFrame


def test_create_causal_frame(dummy_df):
    CausalFrame(dummy_df, covariates=["a", "b"])

    with pytest.raises(AssertionError):
        CausalFrame(dummy_df)

    with pytest.raises(AssertionError):
        CausalFrame(dummy_df, covariates=["a", "b", "c"])


def test_causal_frame_operations(dummy_cf):
    cf = dummy_cf[dummy_cf["a"] <= 5]
    assert isinstance(cf, CausalFrame)

    dummy_cf.drop("b", axis=1)

    assert isinstance(cf["a"], pd.Series)
    assert not hasattr(cf["a"], "_names")


def test_names_extension(dummy_cf, dummy_df):
    with pytest.raises(AssertionError):
        _ = dummy_df.names.covariates

    covariates = dummy_cf.names.covariates
    assert covariates == ["a", "b"]

    others = dummy_cf.names.others
    assert others == ["rep", "sample_id"]


def test_np_extension(dummy_cf, dummy_df):
    with pytest.raises(AssertionError):
        _ = dummy_df.np.X

    X = dummy_cf.np.X
    assert isinstance(X, np.ndarray)
    assert_array_equal(dummy_cf[["a", "b"]].to_numpy(), X)

    y = dummy_cf.np.y
    assert isinstance(y, np.ndarray)
    assert_array_equal(dummy_cf["y"].to_numpy(), y)

    t = dummy_cf.np.t
    assert isinstance(t, np.ndarray)
    assert_array_equal(dummy_cf["t"].to_numpy(), t)

    dummy_cf_no_X = dummy_cf.drop(["a", "b"], axis=1)
    with pytest.raises(KeyError):
        _ = dummy_cf_no_X.np.X

    dummy_cf_no_y = dummy_cf.drop("y", axis=1)
    with pytest.raises(KeyError):
        _ = dummy_cf_no_y.np.y

    dummy_cf_no_t = dummy_cf.drop("t", axis=1)
    with pytest.raises(KeyError):
        _ = dummy_cf_no_t.np.t
