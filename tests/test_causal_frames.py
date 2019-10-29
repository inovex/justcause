import pytest

import pandas as pd

from justcause.data.frames import CausalFrame


def test_create_causal_frame(dummy_df):
    CausalFrame(dummy_df, covariates=["a", "b"])
    with pytest.raises(AssertionError):
        CausalFrame(dummy_df, covariates=["a", "b", "c"])


def test_causal_frame_operations(dummy_cf):
    cf = dummy_cf[dummy_cf["a"] <= 5]
    assert isinstance(cf, CausalFrame)
    assert isinstance(cf["a"], pd.Series)
    assert not hasattr(cf["a"], "_names")


# Todo: Test the functionality of the namespaces here
