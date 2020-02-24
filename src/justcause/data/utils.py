"""Tools for manipulating causal data sets and generating synthetic data

The most important exported function is `justcause.data.utils.generate_data`, which
allows the user to parametrically generate data sets.

"""
from numbers import Number
from typing import Callable, Iterator, List, Optional, Union

import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.utils import check_random_state

from .frames import DATA_COLS, CausalFrame, Col

Indices = Union[List[int], int]
OptRandState = Optional[Union[int, RandomState]]
Frame = Union[CausalFrame, pd.DataFrame]


def select_replication(df: pd.DataFrame, indices: Indices) -> pd.DataFrame:
    """Returns the indicated subset of replications from a DataFrame

    Args:
        df: DataFrame with a `justcause.data.frames.Col.rep` column
        indices: an index or array-like of indices to select

    Returns:
        The subset of replications as a DataFrame

    """
    if isinstance(indices, Number):
        return df.loc[df[Col.rep] == indices]
    else:
        return df.loc[df[Col.rep].isin(indices)]


def to_rep_iter(df: Frame) -> Iterator[Frame]:
    """Turns a monolithic Frame into a generator of replication-wise Frames

    Args:
        df: the Frame to be split up into a generator

    Returns:
        generator: an iterator yielding one replication at a time

    """
    assert (
        Col.rep in df.columns
    ), "replication information required to perform the split"
    for rep in df[Col.rep].unique():
        yield df[df[Col.rep] == rep].drop(Col.rep, axis=1)


def to_rep_list(df: Frame) -> List[Frame]:
    """Turns a monolithic DataFrame into a list of Frames, one for each replication

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
) -> List[CausalFrame]:
    """Generate a synthetic DGP from the given functions

    Following the convention described in the "Usage" chapter of the docs, this
    method can be used to sample a DGP from the parametric definitions of covariates,
    outcomes and treatment. See `justcause.data.generators.ihdp` for an example.


    Args:
        covariates: a callable taking n_samples and random_state and yielding the
            covariates OR a 2D array of covariates with n_sample rows OR a
            DataFrame with the same size
        treatment: a callable taking covariates and random_state and yielding the
            treatment indicator vector
        outcomes: a callable taking covariates and random_state and yielding
            the four outcomes mu_0, mu_1, y_0, y_1 for these covariates
        n_samples: number of samples to generate, only used if covariates is a callable
        n_replications: number of replications to generate of the DGP
        covariate_names: desired names of the covariates in the resulting DGP, defaults
            to [x0, x1, ...]
        random_state: random_state to fix random number generation throughout the
            generation process
        **kwargs: further keyword arguments passed to all callables for manual settings

    Returns:
        a dataset as list of replications generated from the functions.

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
        if isinstance(covariates, pd.DataFrame):
            covariates = covariates.iloc[indices, :]  # ensure proper access
        else:
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

        # Enumerate samples in one replication for join with covariates
        rep_df[Col.sample_id] = np.arange(n_samples)
        rep_df[Col.rep] = i
        rep_dfs.append(rep_df)

    rep_df = pd.concat(rep_dfs)
    df = pd.merge(cov_df, rep_df, how="inner", on=Col.sample_id)
    cf = CausalFrame(df, covariates=covariate_names)
    return to_rep_list(cf)


def add_pot_outcomes_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Generate missing potential outcomes y_0/y_1 from mu_0/my_1

    Args:
        df: CausalFrame with potentially missing potential outcomes

    Returns:
        CausalFrame with columns y_0 and y_1

    """
    if Col.y_0 not in df.columns:
        df[Col.y_0] = df[Col.mu_0]
    if Col.y_1 not in df.columns:
        df[Col.y_1] = df[Col.mu_1]
    return df
