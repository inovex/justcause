from typing import Iterable, Optional

import pandas as pd

from ..frames import CausalFrame, Col
from ..utils import (
    Indices,
    get_covariates_df,
    get_outcomes_df,
    iter_rep,
    select_replication,
)

DATASET_NAME: str = "ibm_acic"


def load_ibm_acic(select_rep: Optional[Indices] = None) -> Iterable[CausalFrame]:
    covariates = get_covariates_df(DATASET_NAME)
    outcomes = get_outcomes_df(DATASET_NAME)

    if select_rep is not None:
        outcomes = select_replication(outcomes, select_rep)

    full = pd.merge(covariates, outcomes, on=Col.sample_id)
    full[Col.ite] = full[Col.mu_1] - full[Col.mu_0]

    cov_names = [col for col in covariates.columns if col != Col.sample_id]
    df = CausalFrame(full, covariates=cov_names)
    return iter_rep(df)


def get_ibm_acic_covariates() -> pd.DataFrame:
    return get_covariates_df(DATASET_NAME).drop(Col.sample_id, axis=1)
