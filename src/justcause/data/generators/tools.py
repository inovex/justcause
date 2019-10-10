import numpy as np
import pandas as pd
from sklearn.utils import Bunch

STANDARD_COLUMNS = ["t", "y", "y_cf", "y_0", "y_1", "ite", "replication", "sample_id"]


def _add_outcomes(covariates, df, size, outcome_fct):
    """ Adds outcomes and derivatives of them to the DataFrame

    Calculates the factual and counterfactual distributions from potential outcomes
    given the treatment in the dataframe

    Args:
        covariates: set of covariates for the outcome function
        df: dataframe to add to
        size: number of samples for which to add outcomes
        outcome_fct:

    Returns:

    """
    Y_0, Y_1 = outcome_fct(covariates, size)
    union = np.c_[Y_0, Y_1]
    Y = [row[id] for row, id in zip(union, df["t"].values)]
    Y_CF = [row[1 - id] for row, id in zip(union, df["t"].values)]
    df["y"] = Y
    df["y_cf"] = Y_CF
    df["y_0"] = Y_0
    df["y_1"] = Y_1
    df["ite"] = Y_1 - Y_0  # add explicitly
    return df


def data_from_generative_function(
    covariates, treatment_fct, outcome_fct, size=None, replications=1
):
    """ Sets up a dataset as Bunch from the given treatment and outcome functions

    Generates outcomes and treatment based on the covariates. Provided callables
    take parameters covariates and size and return np.array of shape (size)
    or (size, 2) for the outcomes.

    If covariate is a callable, size must be specified

    Args:
        covariates: set of covariates or callable to generate them
        treatment_fct: callable taking parameters (covariates, size)
        outcome_fct: callable taking parameters (covariates, size)
        size:
        replications:

    Returns:

    """

    if size is None:
        size = len(covariates)

    if callable(covariates):
        covariates = covariates(size)

    cov_col = ["x" + str(i) for i in range(covariates.shape[1])]

    rep_df = pd.DataFrame(STANDARD_COLUMNS)
    cov_df = pd.DataFrame(data=covariates, columns=cov_col)
    cov_df["sample_id"] = np.arange(len(covariates))

    for i in range(replications):
        replication = pd.DataFrame(columns=STANDARD_COLUMNS)
        replication["t"] = treatment_fct(covariates, size=size)
        replication = _add_outcomes(covariates, replication, size, outcome_fct)
        replication["sample_id"] = np.arange(len(covariates))
        rep_df = rep_df.append(replication, ignore_index=True)

    df = pd.merge(cov_df, rep_df, how="inner", on="sample_id")
    return Bunch(data=df, covariate_names=cov_col, has_test=False)
