from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path
from urllib.parse import urljoin

import requests

from . import DATA_DIR

DATA_URL = "https://raw.github.com/inovex/justcause-data/master/"

_logger = logging.getLogger(__name__)


def create_data_dir(path: Path):
    """
    Creates the directory at the given path if it does not exist

    Args:
        path: of the directory

    Returns:

    """
    if not path.is_dir():
        path.mkdir(parents=True)


def download(url: str, dest_path: PathLike, chunk_size: int = 2 ** 20):
    """ Download file at url to specified location

    Args:
        url:
        dest_path:
        chunk_size:

    """
    req = requests.get(url, stream=True)
    req.raise_for_status()

    with open(dest_path, "wb") as fd:
        _logger.info(f"Downloading from {url}...")
        for chunk in req.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def get_local_data_path(
    path: PathLike, download_if_missing: bool = True, base_url: str = DATA_URL
) -> PathLike:
    """Downloads the file from url if necessary; returns local file path

    If the requested local file does not exist, it is downloaded form the url and the
    local path is returned

    Args:
        path: name of the subdirectory
        download_if_missing: download the
        base_url: base url of repository

    Returns: path to the file
    Raises: IOError if file does not exist and download is set to False

    """
    url = urljoin(str(base_url), str(path))
    path = DATA_DIR / path
    create_data_dir(path.parent)

    if not path.is_file():
        if download_if_missing:
            download(url, path)
        else:
            raise IOError("Dataset missing.")

    return path
