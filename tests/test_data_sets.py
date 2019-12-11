"""
Testing the datasets functionality
"""
import os
from distutils.util import strtobool

import pytest

import numpy as np

from justcause.data.sets import load_ibm_acic, load_ihdp
from justcause.data.sets.twins import get_twins_covariates

RUNS_ON_CIRRUS = bool(strtobool(os.environ.get("CIRRUS_CI", "false")))


def test_ihdp_data(ihdp_data):
    ihdp_data = list(ihdp_data)
    assert len(ihdp_data[0]) == 747  # number of samples in rep
    assert len(ihdp_data) == 1000  # number of reps

    # load single replication
    df = next(load_ihdp(0))
    assert len(df) == 747


def test_ibm_acic_data(ibm_acic_data):
    ibm_acic_data = list(ibm_acic_data)
    rep = ibm_acic_data[0]
    assert len(rep) == 1000
    assert len(ibm_acic_data) == 2  # number of replications


@pytest.mark.skipif(RUNS_ON_CIRRUS, reason="Needs a lot of memory")
def test_ibm_acic_data_load_all():
    df = next(load_ibm_acic())
    assert len(df) == 10000


def test_twins_data(twins_data):
    twins_data = list(twins_data)
    rep = twins_data[0]
    assert np.max(rep["mu_1"]) == 1
    assert np.min(rep["mu_1"]) == 0
    assert len(rep) == 8215


def test_twins_covariates():
    cov_df = get_twins_covariates()
    assert len(cov_df) == 8215
