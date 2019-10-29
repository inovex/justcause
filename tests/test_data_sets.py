import numpy as np


def test_ihdp_data(ihdp_data):
    ihdp_data = list(ihdp_data)
    assert len(ihdp_data[0]) == 747  # number of samples in rep
    assert len(ihdp_data) == 1000  # number of reps


def test_ibm_acic_data(ibm_acic_data):
    ibm_acic_data = list(ibm_acic_data)
    rep = ibm_acic_data[0]
    assert len(rep) == 1000
    assert len(ibm_acic_data) == 2  # number of replications


def test_twins_data(twins_data):
    twins_data = list(twins_data)
    rep = twins_data[0]
    assert np.max(rep["y_1"]) == 1
    assert np.min(rep["y_1"]) == 0
    assert len(rep) == 8215
