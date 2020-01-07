from typing import List

import numpy as np
import pandas as pd

from ..frames import CausalFrame, Col
from ..utils import get_covariates_df, get_outcomes_df, to_rep_list

DATASET_NAME: str = "twins"


def load_twins() -> List[CausalFrame]:
    """

    TODO: Docstring

    There are no replications in the twins dataset, thus no is selection required

    Returns:

    """
    covariates = get_covariates_df(DATASET_NAME)
    outcomes = get_outcomes_df(DATASET_NAME)

    full = pd.merge(covariates, outcomes, on=Col.sample_id)
    full[Col.ite] = full[Col.mu_1] - full[Col.mu_0]
    full[Col.rep] = np.repeat(0, len(full))

    cov_names = [col for col in covariates.columns if col != Col.sample_id]
    df = CausalFrame(full, covariates=cov_names)
    return to_rep_list(df)


def get_twins_covariates() -> pd.DataFrame:
    """

    TODO: Docstring

    Returns:

    """
    return get_covariates_df(DATASET_NAME).drop(Col.sample_id, axis=1)
