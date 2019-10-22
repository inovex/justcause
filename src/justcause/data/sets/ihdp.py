import pandas as pd
from sklearn.datasets.base import Bunch

from . import get_covariates_df, get_outcomes_df

DATASET_NAME = "ihdp"


def load_ihdp() -> Bunch:
    covariates = get_covariates_df(DATASET_NAME)
    outcomes = get_outcomes_df(DATASET_NAME)

    outcomes["sample_id"] = outcomes.groupby("rep").cumcount()
    full = pd.merge(covariates, outcomes, how="left", on="sample_id")
    full["ite"] = full["y_1"] - full["y_0"]

    cov_names = [col for col in covariates.columns if col != "sample_id"]
    bunch = Bunch(data=full, covariate_names=cov_names, has_test=True)
    return bunch


def get_ihdp_covariates() -> pd.DataFrame:
    return get_covariates_df(DATASET_NAME)
