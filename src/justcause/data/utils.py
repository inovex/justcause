from numbers import Number
from os import PathLike
from pathlib import Path
from typing import List, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from .transport import get_local_data_path

COVARIATES_FILE = Path("covariates.parquet")
OUTCOMES_FILE = Path("outcomes.parquet")

#: Type alias
Indices = Union[List[int], int]


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


def group_split(group, train_size, random_state):
    train, test = train_test_split(
        group, train_size=train_size, random_state=random_state
    )
    train.loc[:, "test"] = False
    test.loc[:, "test"] = True
    return pd.concat([train, test]).sort_values("sample_id")


def get_train_test(data_bunch, train_size=0.8, random_state=None):
    """ Applies a train_test_split on each replication

    Args:
        data_bunch: data bunch or dataframe containing the dataset
        train_size: between 0 and 1, indicating the ratio of train/test in each section
        random_state: random seed for train_test_split

    Returns: (train, test) tuple

    """
    if type(data_bunch) is pd.DataFrame:
        df = data_bunch
    else:
        df = data_bunch.data

    use_test = "has_test" in data_bunch and data_bunch.has_test is True
    if use_test:
        return df.loc[~df["test"]], df.loc[df["test"]]

    df = (
        df.groupby("rep")
        .apply(group_split, train_size=0.8, random_state=random_state)
        .reset_index(drop=True)
    )
    return df.loc[~df["test"]], df.loc[df["test"]]
