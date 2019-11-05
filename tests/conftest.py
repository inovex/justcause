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
import rpy2.robjects.packages as rpackages
from rpy2 import robjects
from rpy2.robjects import StrVector

from justcause.data.frames import CausalFrame
from justcause.data.sets.ibm_acic import load_ibm_acic
from justcause.data.sets.ihdp import load_ihdp
from justcause.data.sets.twins import load_twins
from justcause.learners.utils import install_r_packages

RUNS_ON_CIRRUS = bool(strtobool(os.environ.get("CIRRUS_CI", "false")))


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


@pytest.fixture
def uninstall_grf():
    """ Ensures the grf packages is not installed before the test runs"""
    if rpackages.isinstalled("grf"):
        robjects.r.options(download_file_method="curl")
        utils = rpackages.importr("utils")
        utils.chooseCRANmirror(ind=0)

        utils.remove_packages(StrVector(["grf"]))


@pytest.fixture
def grf():
    """ Ensures grf is installed before running tests with it

    This is required as usually the user is requested to install the package manually

    """
    if not rpackages.isinstalled("grf"):
        install_r_packages(["grf"])
        return 0

    return 1
