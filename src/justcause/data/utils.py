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


def to_rep_iter(df: Frame) -> Iterator[Frame]:
    """Turns a monolithic DataFrame into a generator of replication-wise DataFrames

    Args:
        df: the Frame to be split up into a generator

    Returns:
        generator: a python iterator yielding one replication at a time

    """
    assert (
        Col.rep in df.columns
    ), "replication information required to perform the split"
    for rep in df[Col.rep].unique():
        yield df[df[Col.rep] == rep].drop(Col.rep, axis=1)


def to_rep_list(df: Frame) -> List[Frame]:
    """
    Turns a monolithic DataFrame into a list of Frames, one for each replication

    Args:
        df: CausalFrame or DataFrame to be split into a list of Frames
            by the replication

    Returns:
        list: a list of Causal- or DataFrames, one for each replication

    """
    assert (
        Col.rep in df.columns
    ), "replication information required to perform the split"
    return [
        df[df[Col.rep] == rep].drop(Col.rep, axis=1) for rep in df[Col.rep].unique()
    ]


def _add_outcomes(
    df: pd.DataFrame,
    mu_0: np.ndarray,
    mu_1: np.ndarray,
    y_0: np.ndarray,
    y_1: np.ndarray,
) -> pd.DataFrame:
    """Adds outcomes and derivatives of them to the DataFrame

    Calculates the factual and counterfactual distributions from potential outcomes
    given the treatment in the dataframe and adds everything to the Frame

    Args:
        df: dataframe to add the outcomes to with len(df) = num_instances
        mu_0: The noiseless, untreated outcome to be added to the Frame,
            shape (num_instances)
        mu_1: The noiseless, treated outcome to be added to the Frame,
            shape (num_instances)
        y_0: The untreated outcome with added noise,
            shape (num_instances)
        y_1: the treated outcome with added noise,
            shape (num_instances)

    Returns:
        df: The DataFrame with added outcome columns

    """
    df = df.copy()  # avoid side-effects
    t = df[Col.t].to_numpy().astype(np.bool)
    y = np.where(t, y_1, y_0)
    y_cf = np.where(t, y_0, y_1)

    df[Col.y], df[Col.y_cf] = y, y_cf
    df[Col.mu_0], df[Col.mu_1] = mu_0, mu_1
    df[Col.ite] = mu_1 - mu_0  # add explicitly
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
) -> List[Union[CausalFrame, pd.DataFrame]]:
    """

    Todo: Docstring

    Args:
        covariates:
        treatment:
        outcomes:
        n_samples:
        n_replications:
        covariate_names:
        random_state:
        **kwargs:

    Returns:

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
    return to_rep_list(df)


def add_pot_outcomes_if_missing(cf: CausalFrame):
    """Generate missing potential outcomes y_0/y_1 from mu_0/my_1

    Args:
        cf: CausalFrame with potentially missing potential outcomes

    Returns:
        CausalFrame with columns y_0 and y_1

    """
    if Col.y_0 not in cf.columns:
        cf[Col.y_0] = cf[Col.mu_0]
    if Col.y_1 not in cf.columns:
        cf[Col.y_1] = cf[Col.mu_1]
    return cf
