from numbers import Number
from os import PathLike
from pathlib import Path
from typing import Iterator, List, Optional, Union

import pandas as pd
from numpy.random import RandomState

from .transport import get_local_data_path

COVARIATES_FILE = Path("covariates.parquet")
OUTCOMES_FILE = Path("outcomes.parquet")

#: Type aliases
Indices = Union[List[int], int]
OptRandState = Optional[Union[int, RandomState]]


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


def select_replication(df: pd.DataFrame, indices: Indices):
    if isinstance(indices, Number):
        return df.loc[df["rep"] == indices]
    else:
        return df.loc[df["rep"].isin(indices)]


def iter_rep(df: pd.DataFrame) -> Iterator[pd.DataFrame]:
    """Iterate over all replications in dataset
    """
    for rep in df["rep"].unique():
        yield df[df["rep"] == rep]
