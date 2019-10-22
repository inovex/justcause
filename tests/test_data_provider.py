import os
from distutils.util import strtobool
from pathlib import Path

import pytest

import numpy as np
import pandas as pd

from justcause.data.transport import create_data_dir, download, get_local_data_path
from justcause.data.utils import get_train_test

RUNS_ON_CIRRUS = bool(strtobool(os.environ.get("CIRRUS_CI", "false")))


# @pytest.mark.skipif(RUNS_ON_CIRRUS, reason="Memory limits on Cirrus CI")
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


def test_ibm_dataprovider(ibm_data):
    assert ibm_data is not None
    assert ibm_data.data is not None
    all_true = np.all(np.isin(ibm_data.covariate_names, list(ibm_data.data.columns)))
    assert all_true
    rep = ibm_data.data[ibm_data.data["rep"] == 0]
    assert len(rep) == rep["size"].iloc[0]
    assert len(ibm_data.data.groupby("rep")) == 2  # number of replications


def test_twins_dataprovider(twins_data):
    assert twins_data is not None
    assert twins_data.data is not None
    assert len(twins_data.data) == 8215
    assert np.max(twins_data.data["y_1"]) == 1
    assert np.min(twins_data.data["y_1"]) == 0
    single_replication = twins_data.data[twins_data.data["rep"] == 0]
    assert len(single_replication) == 8215


def test_transport(tmpdir):
    """ Test utility functions for data loading"""

    tmpdir = Path(tmpdir)  # convert to pathlib Path
    create_data_dir(tmpdir)
    assert os.path.isdir(tmpdir)

    url = "https://raw.github.com/inovex/justcause-data/master/ihdp/covariates.gzip"
    result_path = tmpdir / Path("cov.gzip")
    download(url, result_path)
    assert result_path.is_file()
    assert pd.read_parquet(result_path) is not None

    non_existant_path = tmpdir / Path("does/not/exist")
    with pytest.raises(IOError):
        get_local_data_path(non_existant_path, download_if_missing=False)


def test_train_test_split_provided(ihdp_data):
    """Tests use of provided test indices as split"""
    train, test = get_train_test(ihdp_data)

    train_rep = train.loc[train["rep"] == 0]
    test_rep = test.loc[test["rep"] == 0]
    assert len(train_rep.loc[~train_rep["test"]]) == 672
    assert len(test_rep.loc[test_rep["test"]]) == 75
    assert len(train.groupby("rep")) == 1000


def test_train_test_split_generated(ibm_data):
    num_instances = len(ibm_data.data[ibm_data.data["rep"] == 0])
    train, test = get_train_test(ibm_data, train_size=0.8)
    train_rep = train.loc[train["rep"] == 0]
    test_rep = test.loc[test["rep"] == 0]
    assert len(train_rep) == int(num_instances * 0.8)
    assert len(test_rep) == int(num_instances * 0.2)
