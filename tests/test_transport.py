import os
from pathlib import Path

import pytest

import pandas as pd

from justcause.data.transport import create_data_dir, download, get_local_data_path


def test_create_data_dir(tmpdir):
    tmpdir = Path(tmpdir)  # convert to pathlib Path
    create_data_dir(tmpdir)
    assert os.path.isdir(tmpdir)


def test_download(tmpdir):
    tmpdir = Path(tmpdir)  # convert to pathlib Path
    url = "https://raw.github.com/inovex/justcause-data/master/ihdp/covariates.parquet"
    result_path = tmpdir / Path("cov.gzip")
    download(url, result_path)
    assert result_path.is_file()
    assert pd.read_parquet(result_path) is not None


def test_get_local_data_path(tmpdir):
    """Test utility functions for data loading"""
    tmpdir = Path(tmpdir)  # convert to pathlib Path

    non_existent_path = tmpdir / Path("does/not/exist")
    with pytest.raises(IOError):
        get_local_data_path(
            non_existent_path, download_if_missing=False, base_path=tmpdir
        )

    path = Path("twins/covariates.parquet")
    data_path = get_local_data_path(path, download_if_missing=True, base_path=tmpdir)
    assert data_path == tmpdir / path

    # retry after it was downloaded
    path = Path("twins/covariates.parquet")
    data_path = get_local_data_path(path, download_if_missing=False, base_path=tmpdir)
    assert data_path == tmpdir / path
