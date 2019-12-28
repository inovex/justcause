from numbers import Number
from os import PathLike
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Union

import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.utils import check_random_state

from .frames import DATA_COLS, CausalFrame, Col
from .transport import get_local_data_path

#: Name of the file holding the covariates
COVARIATES_FILE: Path = Path("covariates.parquet")
#: Name of the file holding the outcomes and replications
OUTCOMES_FILE: Path = Path("outcomes.parquet")

Indices = Union[List[int], int]
OptRandState = Optional[Union[int, RandomState]]
Frame = Union[CausalFrame, pd.DataFrame]


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
        return df.loc[df[Col.rep] == indices]
    else:
        return df.loc[df[Col.rep].isin(indices)]


def iter_rep(df: Frame) -> Iterator[Frame]:
    """Iterate over all replications in dataset
    """
    for rep in df[Col.rep].unique():
        yield df[df[Col.rep] == rep].drop(Col.rep, axis=1)


def _add_outcomes(
    df: pd.DataFrame, m_0: np.ndarray, m_1: np.ndarray, y_0: np.ndarray, y_1: np.ndarray
) -> pd.DataFrame:
    """Adds outcomes and derivatives of them to the DataFrame

    Calculates the factual and counterfactual distributions from potential outcomes
    given the treatment in the dataframe

    Args:
        df: dataframe to add to
        size: number of samples for which to add outcomes
        outcome:

    Returns:

    """
    df = df.copy()  # avoid side-effects
    t = df[Col.t].to_numpy().astype(np.bool)
    y = np.where(t, y_1, y_0)
    y_cf = np.where(t, y_0, y_1)

    df[Col.y], df[Col.y_cf] = y, y_cf
    df[Col.mu_0], df[Col.mu_1] = m_0, m_1
    df[Col.ite] = y_1 - y_0  # add explicitly
    return df


def generate_data(
    covariates: Union[Callable, np.ndarray, pd.DataFrame],
    treatment: Callable,
    outcomes: Callable,
    n_samples: Optional[int] = None,
    n_replications: int = 1,
    covariate_names: Optional[List[str]] = None,
    random_state: OptRandState = None,
    **kwargs,
) -> Iterator[Union[CausalFrame, pd.DataFrame]]:
    """Generate CausalFrame from covariates, treatment and outcome functions
    """
    random_state = check_random_state(random_state)

    if n_samples is None:
        assert not callable(
            covariates
        ), "Covariates must not be a callable if `n_samples` was not specified"
        n_samples = covariates.shape[0]
    elif callable(covariates):
        covariates = covariates(n_samples, random_state=random_state, **kwargs)
        assert (
            covariates.shape[0] == n_samples
        ), "Covariate function should return a dataframe with `n_samples` rows"
    else:
        indices = random_state.choice(covariates.shape[0], n_samples, replace=False)
        covariates = covariates[indices, :]

    if covariate_names is None:
        if isinstance(covariates, pd.DataFrame):
            covariate_names = list(covariates.columns)
        else:
            covariate_names = [f"x{i}" for i in range(covariates.shape[1])]

    # No need to check `covariate_names` since Pandas does it
    cov_df = pd.DataFrame(data=covariates, columns=covariate_names)
    cov_df[Col.sample_id] = np.arange(n_samples)

    rep_dfs = list()
    for i in range(n_replications):
        rep_df = pd.DataFrame(columns=DATA_COLS)
        rep_df[Col.t] = treatment(covariates, random_state=random_state, **kwargs)
        assert (
            rep_df.shape[0] == n_samples
        ), "Treatment function must return vector with dimension `n_samples`"

        mu_0, mu_1, y_0, y_1 = outcomes(covariates, random_state=random_state, **kwargs)
        assert (
            y_0.shape[0] == y_1.shape[0] == n_samples
        ), "Outcome function must return vectors, each with dimension `n_samples`"

        rep_df = _add_outcomes(rep_df, mu_0, mu_1, y_0, y_1)
        rep_df[Col.sample_id] = np.arange(n_samples)
        rep_df[Col.rep] = i
        rep_dfs.append(rep_df)

    rep_df = pd.concat(rep_dfs)
    df = pd.merge(cov_df, rep_df, how="inner", on=Col.sample_id)
    df = CausalFrame(df, covariates=covariate_names)
    return iter_rep(df)
