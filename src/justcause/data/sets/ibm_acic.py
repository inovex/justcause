from typing import Optional

import pandas as pd
from sklearn.datasets.base import Bunch

from ..utils import Indices, get_covariates_df, get_outcomes_df, select_replication

DATASET_NAME = "ibm_acic"


def load_ibm_acic(select_rep: Optional[Indices] = None) -> Bunch:
    covariates = get_covariates_df(DATASET_NAME)
    outcomes = get_outcomes_df(DATASET_NAME)

    if select_rep is not None:
        outcomes = select_replication(outcomes, select_rep)

    full = pd.merge(covariates, outcomes, on="sample_id")
    full["ite"] = full["y_1"] - full["y_0"]

    cov_names = [col for col in covariates.columns if col != "sample_id"]
    bunch = Bunch(data=full, covariate_names=cov_names, has_test=False)
    return bunch


def get_ibm_acic_covariates() -> pd.DataFrame:
    return get_covariates_df(DATASET_NAME)
