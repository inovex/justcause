import os
from pathlib import Path

import pytest

import pandas as pd

from justcause.data.transport import create_data_dir, download, get_local_data_path


def test_transport(tmpdir):
    """ Test utility functions for data loading"""

    tmpdir = Path(tmpdir)  # convert to pathlib Path
    create_data_dir(tmpdir)
    assert os.path.isdir(tmpdir)

    url = "https://raw.github.com/inovex/justcause-data/master/ihdp/covariates.parquet"
    result_path = tmpdir / Path("cov.gzip")
    download(url, result_path)
    assert result_path.is_file()
    assert pd.read_parquet(result_path) is not None

    non_existant_path = tmpdir / Path("does/not/exist")
    with pytest.raises(IOError):
        get_local_data_path(non_existant_path, download_if_missing=False)
