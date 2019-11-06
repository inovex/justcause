from typing import Iterable

import numpy as np
import pandas as pd

from ..frames import CausalFrame, Col
from ..utils import get_covariates_df, get_outcomes_df, iter_rep

DATASET_NAME: str = "twins"


def load_twins() -> Iterable[CausalFrame]:
    covariates = get_covariates_df(DATASET_NAME)
    outcomes = get_outcomes_df(DATASET_NAME)

    full = pd.merge(covariates, outcomes, on=Col.sample_id)
    full[Col.ite] = full[Col.mu_1] - full[Col.mu_0]
    full[Col.rep] = np.repeat(0, len(full))

    cov_names = [col for col in covariates.columns if col != Col.sample_id]
    df = CausalFrame(full, covariates=cov_names)
    return iter_rep(df)


def get_twins_covariates() -> pd.DataFrame:
    return get_covariates_df(DATASET_NAME).drop(Col.sample_id, axis=1)
