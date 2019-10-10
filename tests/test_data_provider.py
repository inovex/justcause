import os

import pytest

import numpy as np
import pandas as pd

from justcause.data.sets.ihdp import load_ihdp
from justcause.data.transport import create_data_dir, download, get_data

test_dir = "tests/results/create_dir"


class TestDataProvider:
    def test_ihdp_dataprovider(self):
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

    def test_transport(self):
        """ Test utility functions for data loading"""

        create_data_dir(test_dir)
        assert os.path.isdir(test_dir)

        url = "https://raw.github.com/inovex/justcause-data/master/ihdp/covariates.gzip"
        result_path = test_dir + "/cov.gzip"
        download(url, result_path)
        assert os.path.isfile(result_path)
        assert pd.read_parquet(result_path) is not None

        with pytest.raises(IOError):
            get_data(
                url,
                dest_subdir="doesnotexist",
                dest_filename="doesnotmatter",
                download_if_missing=False,
            )

    def teardown_method(cls):
        """ Remove created directory"""
        import shutil

        if os.path.isdir(test_dir):
            shutil.rmtree(test_dir, ignore_errors=True)
