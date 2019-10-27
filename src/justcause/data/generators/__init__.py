import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from .. import DATA_COLS


def _add_outcomes(covariates, df, outcome):
    """ Adds outcomes and derivatives of them to the DataFrame

    Calculates the factual and counterfactual distributions from potential outcomes
    given the treatment in the dataframe

    Args:
        covariates: set of covariates for the outcome function
        df: dataframe to add to
        size: number of samples for which to add outcomes
        outcome:

    Returns:

    """
    Y_0, Y_1 = outcome(covariates)
    union = np.c_[Y_0, Y_1]
    Y = [row[id] for row, id in zip(union, df["t"].values)]
    Y_CF = [row[1 - id] for row, id in zip(union, df["t"].values)]
    df["y"], df["y_cf"] = Y, Y_CF
    df["y_0"], df["y_1"] = Y_0, Y_1
    df["ite"] = Y_1 - Y_0  # add explicitly
    return df


def generate_data(
    covariates, treatment, outcome, size=None, replications=1, covariate_names=None
):
    """ Sets up a dataset as Bunch from the given treatment and outcome functions

    Generates outcomes and treatment based on the covariates. Provided callables
    take parameters covariates and size and return np.array of shape (size)
    or (size, 2) for the outcomes.

    If covariate is a callable, size must be specified

    Args:
        covariates: set of covariates as np.array or callable to generate them
        treatment: callable taking parameters (covariates)
        outcome: callable taking parameters (covariates)
        size: number of samples in each replication
        replications: number of replications
        covariate_names:

    Returns:

    """

    if size is None:
        size = len(covariates)

    if callable(covariates):
        covariates = covariates(size)

    elif size < len(covariates):
        choice = np.random.choice(range(len(covariates)), size=size)
        covariates = covariates[choice]

    if covariate_names:
        assert len(covariate_names) == covariates.shape[1]
        cov_col = covariate_names
    else:
        # Default naming
        cov_col = ["x" + str(i) for i in range(covariates.shape[1])]

    rep_df = pd.DataFrame(DATA_COLS)
    cov_df = pd.DataFrame(data=covariates, columns=cov_col)

    # Add id column
    cov_df["sample_id"] = np.arange(len(covariates))

    for i in range(replications):
        replication = pd.DataFrame(columns=DATA_COLS)
        replication["t"] = treatment(covariates)
        replication = _add_outcomes(covariates, replication, outcome)
        replication["sample_id"] = np.arange(len(covariates))
        replication["rep"] = i
        rep_df = rep_df.append(replication, ignore_index=True)

    df = pd.merge(cov_df, rep_df, how="inner", on="sample_id")
    return Bunch(data=df, covariate_names=cov_col, has_test=False)
