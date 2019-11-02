from typing import Callable, Union, Optional, List

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from .. import DATA_COLS
from ..frames import CausalFrame
from ..utils import OptRandState


def _add_outcomes(df: pd.DataFrame, y_0: np.ndarray, y_1: np.ndarray) -> pd.DataFrame:
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
    y_01 = np.c_[y_0, y_1]
    # Todo: find a smarter way to do this here!
    y = [row[id] for row, id in zip(y_01, df["t"].values)]
    y_cf = [row[1 - id] for row, id in zip(y_01, df["t"].values)]
    df["y"], df["y_cf"] = y, y_cf
    df["y_0"], df["y_1"] = y_0, y_1
    df["ite"] = y_1 - y_0  # add explicitly
    return df


def generate_data(
    covariates: Union[Callable, np.ndarray, pd.DataFrame],
    treatment: Callable,
    outcomes: Callable,
    n_samples: Optional[int] = None,
    n_replications: int = 1,
    covariate_names: Optional[List[str]] = None,
    random_state: OptRandState = None,
) -> CausalFrame:
    """Generate CausalFrame from covariates, treatment and outcome functions
    """
    random_state = check_random_state(random_state)

    if n_samples is None:
        assert not callable(
            covariates
        ), "Covariates must not be a callable if `n_samples` was not specified"
        n_samples = covariates.shape[0]
    elif callable(covariates):
        covariates = covariates(n_samples)
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
    cov_df["sample_id"] = np.arange(n_samples)

    rep_dfs = list()
    for i in range(n_replications):
        rep_df = pd.DataFrame(columns=DATA_COLS)
        rep_df["t"] = treatment(covariates)
        assert (
            rep_df.shape[0] == n_samples
        ), "Treatment function must return vector with dimension `n_samples`"

        y_0, y_1 = outcomes(covariates)
        assert (
            y_0.shape[0] == y_1.shape[0] == n_samples
        ), "Outcome function must return vectors with dimension `n_samples"

        rep_df = _add_outcomes(rep_df, y_0, y_1)
        rep_df["sample_id"] = np.arange(n_samples)
        rep_df["rep"] = i
        rep_dfs.append(rep_df)

    rep_df = pd.concat(rep_dfs)
    df = pd.merge(cov_df, rep_df, how="inner", on="sample_id")
    return CausalFrame(df, covariates=covariate_names)
