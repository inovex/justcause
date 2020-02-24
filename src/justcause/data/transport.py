"""Helper functions for downloading and storing data from the justcause-data repository

The functions provided here are mostly used in `justcause.data.utils` to get access
to a specified dataset. The transport module hides the fact, that the dataset might
have to be downloaded first.

"""
# Uncomment only when we require Python >= 3.7
# from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import pandas as pd
import requests

_logger = logging.getLogger(__name__)

#: URL for retrieving the datasets
DATA_URL: str = "https://raw.github.com/inovex/justcause-data/master/"
#: Directory for storing the datasets locally
DATA_DIR: Path = Path("~/.justcause_data").expanduser()
#: Name of the file holding the covariates
COVARIATES_FILE: Path = Path("covariates.parquet")
#: Name of the file holding the outcomes and replications
OUTCOMES_FILE: Path = Path("outcomes.parquet")


def get_covariates_df(dataset_name: str) -> pd.DataFrame:
    """Returns the covariates dataframe from local storage or the data repository

    Args:
        dataset_name: unique identifier of the dataset from the justcause-data
            repository (e.g. 'ibm' or 'ihdp')

    """
    path = Path(dataset_name) / COVARIATES_FILE
    return get_dataframe(path)


def get_outcomes_df(dataset_name: str) -> pd.DataFrame:
    """Returns the outcomes dataframe from local storage or the data repository

    Args:
        dataset_name: unique identifier of the dataset from the justcause-data
            repository (e.g. 'ibm' or 'ihdp')

    """
    path = Path(dataset_name) / OUTCOMES_FILE
    return get_dataframe(path)


def get_dataframe(data_path: PathLike) -> pd.DataFrame:
    """Returns the DataFrame at the given relative path

    Resolves the data path by downloading the data from the repository if it is
    not available at the `DATA_DIR` path.

    Args:
        data_path: relative path of the DataFrame (for example "ibm/covariates.parquet")

    """
    path = get_local_data_path(data_path, download_if_missing=True)
    df = pd.read_parquet(path)
    return df


def create_data_dir(path: Path):
    """Creates the directory at the given path if it does not exist

    Args:
        path: of the directory

    """
    if not path.is_dir():
        path.mkdir(parents=True)


def download(url: str, dest_path: PathLike, chunk_size: Optional[int] = None):
    """Downloads the file at url to a location specified in the dest_path

    Args:
        url: the full url of the file to download
        dest_path: the file path to write the file to
        chunk_size: the chunk size used for writing on the file descriptor

    """
    chunk_size = 2 ** 20 if chunk_size is None else chunk_size
    req = requests.get(url, stream=True)
    req.raise_for_status()

    with open(dest_path, "wb") as fd:
        _logger.info(f"Downloading from {url}...")
        for chunk in req.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def get_local_data_path(
    path: PathLike,
    download_if_missing: bool = True,
    base_url: str = DATA_URL,
    base_path: PathLike = DATA_DIR,
) -> PathLike:
    """Returns the local file path of a dataset url

    If the requested local file corresponding to the url of the dataset
    does not exist, it is downloaded form the url and the local path is returned

    Args:
        path: name of the subdirectory implicitly car
        download_if_missing: download the dataset if it is not present locally
        base_url: base url of data repository
        base_path: base path where the datasets are cached locally

    Returns:
        usable local path to the file

    Raises:
        IOError if file does not exist and download is set to False

    """
    url = urljoin(str(base_url), str(path))
    path = Path(base_path) / path
    create_data_dir(path.parent)

    if not path.is_file():
        if download_if_missing:
            download(url, path)
        else:
            raise IOError(f"Dataset {path} is missing.")

    return path
