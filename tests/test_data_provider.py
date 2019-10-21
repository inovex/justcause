import os

import pytest

import numpy as np
import pandas as pd

from justcause.data import get_train_test
from justcause.data.transport import create_data_dir, download, get_local_data_path

RUNS_ON_CIRRUS = os.environ.get("CI", False)


@pytest.mark.skipif(RUNS_ON_CIRRUS, "No huge downloads on Cirrus CI")
def test_ihdp_dataprovider(ihdp_data):
    """ Tests the new IHDP dataprovider"""
    assert ihdp_data is not None
    assert ihdp_data.data is not None
    assert ihdp_data.covariate_names is not None
    all_true = np.all(np.isin(ihdp_data.covariate_names, list(ihdp_data.data.columns)))
    assert all_true

    rep = ihdp_data.data.loc[ihdp_data.data["rep"] == 0]
    assert len(rep) == 747  # number of samples in rep
    assert len(ihdp_data.data.groupby("rep")) == 1000  # number of reps
    assert ihdp_data.has_test
    assert len(rep.loc[rep["test"]]) == 75  # number of test samples in rep


@pytest.mark.skipif(RUNS_ON_CIRRUS, "No huge downloads on Cirrus CI")
def test_ibm_dataprovider(ibm_data):
    assert ibm_data is not None
    assert ibm_data.data is not None
    all_true = np.all(np.isin(ibm_data.covariate_names, list(ibm_data.data.columns)))
    assert all_true
    rep = ibm_data.data[ibm_data.data["rep"] == 0]
    assert len(rep) == rep["size"].iloc[0]
    assert len(ibm_data.data.groupby("rep")) == 50  # number of replications


@pytest.mark.skipif(RUNS_ON_CIRRUS, "No huge downloads on Cirrus CI")
def test_twins_dataprovider(twins_data):
    assert twins_data is not None
    assert twins_data.data is not None
    assert len(twins_data.data) == 8215
    assert np.max(twins_data.data["y_1"]) == 1
    assert np.min(twins_data.data["y_1"]) == 0
    assert (
        len(twins_data.data[twins_data.data["rep"] == 0]) == 8215
    )  # Only one replication


def test_transport(tmpdir):
    """ Test utility functions for data loading"""

    create_data_dir(tmpdir)
    assert os.path.isdir(tmpdir)

    url = "https://raw.github.com/inovex/justcause-data/master/ihdp/covariates.gzip"
    result_path = str(tmpdir) + "/cov.gzip"
    download(url, result_path)
    assert os.path.isfile(result_path)
    assert pd.read_parquet(result_path) is not None

    with pytest.raises(IOError):
        get_local_data_path(
            url,
            dest_subdir="doesnotexist",
            dest_filename="doesnotmatter",
            download_if_missing=False,
        )


def test_train_test_split_provided(ihdp_data):
    """ Tests use of provided test indizes as split"""
    print(ihdp_data.data.head())
    train, test = get_train_test(ihdp_data)

    train_rep = train.loc[train["rep"] == 0]
    test_rep = test.loc[test["rep"] == 0]
    assert len(train_rep.loc[~train_rep["test"]]) == 672
    assert len(test_rep.loc[test_rep["test"]]) == 75
    assert len(train.groupby("rep")) == 1000  # number of replications still the same


@pytest.mark.xfail
def test_train_test_split_generated(ibm_data):
    num_instances = len(ibm_data.data[ibm_data.data["rep"] == 0])
    train, test = get_train_test(ibm_data, train_size=0.8)
    train_rep = train.loc[train["rep"] == 0]
    test_rep = test.loc[test["rep"] == 0]
    assert len(train_rep) == int(num_instances * 0.8)
    assert len(test_rep) == int(num_instances * 0.2)
