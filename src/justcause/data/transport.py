import os

import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.expanduser("~"), "justcause_data")

COV_FILE = "covariates.gzip"
REP_FILE = "outcomes.gzip"


def create_data_dir(path):
    """
    Creates the directory at the given path if it does not exist

    Args:
        path: of the directory

    Returns:

    """

    if not os.path.isdir(path):
        os.makedirs(path)


def download(url, dest_path):
    """ Download file at url to specified location

    Args:
        url:
        dest_path:

    """

    req = requests.get(url, stream=True)
    req.raise_for_status()

    with open(dest_path, "wb") as fd:
        print("downloading from ", url)
        for chunk in req.iter_content(chunk_size=2 ** 20):
            fd.write(chunk)


def get_local_data_path(url, dest_subdir, dest_filename, download_if_missing=True):
    """ Downloads the file from url if necessary; returns local file path

    If the requested local file does not exist, it is downloaded form the url and the
    local path is returned

    Args:
        url: url from where to download
        dest_subdir: name of the subdirectory
        dest_filename: name of the file
        download_if_missing: download the

    Returns: path to the file
    Raises: IOError if file does not exist and download is set to False

    """

    data_dir = os.path.join(os.path.abspath(DATA_DIR), dest_subdir)

    create_data_dir(data_dir)

    dest_path = os.path.join(data_dir, dest_filename)

    if not os.path.isfile(dest_path):
        if download_if_missing:
            download(url, dest_path)
        else:
            raise IOError("Dataset missing.")

    return dest_path


def load_parquet_dataset(base_url, dest_subdir):
    """ Load dataset stored in parquet at the specified directory

    Takes advantage of the efficient storage format parquet

    Args:
        base_url: base url where the files can be found
        dest_subdir: base directory where to put the results

    Returns:

    """
    COV_URL = os.path.join(base_url, COV_FILE)
    REP_URL = os.path.join(base_url, REP_FILE)

    cov_path = get_local_data_path(
        COV_URL, dest_subdir, COV_FILE, download_if_missing=True
    )
    cov = pd.read_parquet(cov_path)
    rep_path = get_local_data_path(
        REP_URL, dest_subdir, REP_FILE, download_if_missing=True
    )
    rep = pd.read_parquet(rep_path)
    return cov, rep
