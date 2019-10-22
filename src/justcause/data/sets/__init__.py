from pathlib import Path
from os import PathLike

import pandas as pd

from ..transport import get_local_data_path

COVARIATES_FILE = Path("covariates.gzip")
OUTCOMES_FILE = Path("outcomes.gzip")


def get_covariates_df(dataset_name: str) -> pd.DataFrame:
    path = Path(dataset_name) / COVARIATES_FILE
    return get_dataframe(path)


def get_outcomes_df(dataset_name: str) -> pd.DataFrame:
    path = Path(dataset_name) / OUTCOMES_FILE
    return get_dataframe(path)


def get_dataframe(data_path: PathLike) -> pd.DataFrame:
    path = get_local_data_path(data_path, download_if_missing=True)
    df = pd.read_parquet(path)
    return df
