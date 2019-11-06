from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests

_logger = logging.getLogger(__name__)

#: URL for retrieving the datasets
DATA_URL: str = "https://raw.github.com/inovex/justcause-data/master/"
#: Directory for storing the datasets locally
DATA_DIR: Path = Path("~/.justcause_data").expanduser()


def create_data_dir(path: Path):
    """
    Creates the directory at the given path if it does not exist

    Args:
        path: of the directory

    Returns:

    """
    if not path.is_dir():
        path.mkdir(parents=True)


def download(url: str, dest_path: PathLike, chunk_size: Optional[int] = None):
    """ Download file at url to specified location

    Args:
        url:
        dest_path:
        chunk_size:

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
    """Downloads the file from url if necessary; returns local file path

    If the requested local file does not exist, it is downloaded form the url and the
    local path is returned

    Args:
        path: name of the subdirectory
        download_if_missing: download the
        base_url: base url of repository
        base_path: base path where the datasets are cached

    Returns: path to the file
    Raises: IOError if file does not exist and download is set to False

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
