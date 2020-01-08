"""
Testing the datasets functionality and integrity
"""
import os
from distutils.util import strtobool

import pytest

import numpy as np

from justcause.data.sets import load_ibm, load_ihdp
from justcause.data.sets.twins import get_twins_covariates

RUNS_ON_CIRRUS = bool(strtobool(os.environ.get("CIRRUS_CI", "false")))


def test_ihdp_data(ihdp_data, ihdp_data_full):
    assert len(ihdp_data[0]) == 747  # number of samples in rep
    assert len(ihdp_data_full) == 1000  # all replications

    # load single replication
    rep_list = load_ihdp(0)
    df = rep_list[0]
    assert len(rep_list) == 1
    assert len(df) == 747


def test_ibm_data(ibm_acic_data):
    rep = ibm_acic_data[0]
    assert len(rep) == 1000
    assert len(ibm_acic_data) == 2  # number of replications selected in conftest.py


@pytest.mark.skipif(RUNS_ON_CIRRUS, reason="Needs a lot of memory")
def test_ibm_data_load_all():
    df = load_ibm()[0]
    assert len(df) == 10000


def test_twins_data(twins_data):
    rep = twins_data[0]
    assert np.max(rep["mu_1"]) == 1
    assert np.min(rep["mu_1"]) == 0
    assert len(rep) == 8215


def test_twins_covariates():
    cov_df = get_twins_covariates()
    assert len(cov_df) == 8215
