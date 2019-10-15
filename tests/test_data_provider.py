import os

import pytest

import numpy as np
import pandas as pd

from justcause.data.sets.ibm import load_ibm_acic
from justcause.data.sets.ihdp import load_ihdp
from justcause.data.transport import create_data_dir, download, get_local_data_path


def test_ihdp_dataprovider():
    """ Tests the new IHDP dataprovider"""
    ihdp = load_ihdp()
    assert ihdp is not None
    assert ihdp.data is not None
    assert ihdp.covariate_names is not None
    all_true = np.all(np.isin(ihdp.covariate_names, list(ihdp.data.columns)))
    assert all_true

    rep = ihdp.data.loc[ihdp.data["rep"] == 0]
    assert len(rep) == 747  # number of samples in rep
    assert len(ihdp.data.groupby("rep")) == 1000  # number of reps
    assert ihdp.has_test
    assert len(rep.loc[rep["test"]]) == 75  # number of test samples in rep


def test_ibm_dataprovider():

    ibm = load_ibm_acic()
    assert ibm is not None
    assert ibm.data is not None
    all_true = np.all(np.isin(ibm.covariate_names, list(ibm.data.columns)))
    assert all_true
    rep = ibm.data[ibm.data["rep"] == 0]
    assert len(rep) == rep["size"].iloc[0]
    assert len(ibm.data.groupby("rep")) == 50  # number of replications


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
