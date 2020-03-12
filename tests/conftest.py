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
from justcause.data.sets.ibm import load_ibm
from justcause.data.sets.ihdp import load_ihdp
from justcause.data.sets.twins import load_twins

RUNS_ON_CIRRUS = bool(strtobool(os.environ.get("CIRRUS_CI", "false")))


@pytest.fixture
def ihdp_data_full():
    return load_ihdp()


@pytest.fixture
def ihdp_data():
    # Limit the replications for better runtime in tests
    return load_ihdp(select_rep=[0, 1])


@pytest.fixture
def ibm_acic_data():
    # Limit the replications for better runtime in tests
    return load_ibm(select_rep=[0, 1])


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
            "y_cf": 1.0 - np.linspace(0.0, 1.0, N),
            "y_0": np.linspace(0.0, 1.0, N),
            "y_1": np.linspace(0.0, 1.0, N),
            "mu_0": np.linspace(0.0, 1.0, N),
            "mu_1": np.linspace(0.0, 1.0, N),
            "ite": np.linspace(0.0, 1.0, N),
            "rep": (2 * np.arange(N) / N).astype(np.int),
            "sample_id": np.arange(10),
        }
    )


@pytest.fixture
def dummy_cf(dummy_df):
    return CausalFrame(dummy_df, covariates=["a", "b"])


@pytest.fixture
def dummy_rep_df():
    N = 10
    num_rep = 5
    return pd.DataFrame(
        {
            "a": np.repeat(np.arange(N), num_rep),
            "b": np.repeat(2 * np.arange(N), num_rep),
            "t": np.repeat((2 * np.arange(N) / N), num_rep).astype(np.int),
            "y": np.repeat(np.linspace(0.0, 1.0, N), num_rep),
            "rep": np.tile(np.arange(num_rep), N),
        }
    )


@pytest.fixture
def dummy_covariates_and_treatment():
    X_0 = np.full((100, 10), 1)
    X_1 = np.full((80, 10), 2)
    X = np.append(X_0, X_1, axis=0)
    T_0 = np.full(100, 1)
    T_1 = np.full(80, 0)
    t = np.append(T_0, T_1)
    return X, t
