# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for justcause.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""
import os
from distutils.util import strtobool

import pytest

import numpy as np
import pandas as pd

from justcause.data.frames import CausalFrame
from justcause.data.sets.ibm_acic import load_ibm_acic
from justcause.data.sets.ihdp import load_ihdp
from justcause.data.sets.twins import load_twins

RUNS_ON_CIRRUS = bool(strtobool(os.environ.get("CIRRUS_CI", "false")))


@pytest.fixture
def grf():
    """Assure the installation of Generalized Random Forests"""
    from justcause.learners.tree.causal_forest import CausalForest

    CausalForest.install_grf()


@pytest.fixture
def ihdp_data():
    return load_ihdp()


@pytest.fixture
def ibm_acic_data():
    return load_ibm_acic(select_rep=[0, 1])


@pytest.fixture
def twins_data():
    return load_twins()


@pytest.fixture
def dummy_df():
    N = 10
    return pd.DataFrame(
        {
            "a": np.arange(N),
            "b": 2 * np.arange(N),
            "t": (2 * np.arange(N) / N).astype(np.int),
            "y": np.linspace(0.0, 1.0, N),
            "rep": (2 * np.arange(N) / N).astype(np.int),
        }
    )


@pytest.fixture
def dummy_cf(dummy_df):
    return CausalFrame(dummy_df, covariates=["a", "b"])


def toy_data():
    X = [[0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 0]]
    Y = [0, 1, 0, 0.5]
    T = [0, 1, 0, 1]
    return X, T, Y
