import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch

from ..utils import get_covariates_df, get_outcomes_df

DATASET_NAME = "twins"


def load_twins() -> Bunch:
    covariates = get_covariates_df(DATASET_NAME)
    outcomes = get_outcomes_df(DATASET_NAME)

    full = pd.merge(covariates, outcomes, on="sample_id")
    full["ite"] = full["y_1"] - full["y_0"]
    full["rep"] = np.repeat(0, len(full))

    cov_names = [col for col in covariates.columns if col != "sample_id"]
    bunch = Bunch(data=full, covariate_names=cov_names)
    return bunch


def get_twins_covariates() -> pd.DataFrame:
    return get_covariates_df(DATASET_NAME)
