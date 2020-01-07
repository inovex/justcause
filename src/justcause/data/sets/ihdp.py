from typing import List, Optional

import pandas as pd

from ..frames import CausalFrame, Col
from ..utils import (
    Indices,
    get_covariates_df,
    get_outcomes_df,
    select_replication,
    to_rep_list,
)

DATASET_NAME: str = "ihdp"


def load_ihdp(select_rep: Optional[Indices] = None) -> List[CausalFrame]:
    """

    TODO: Docstring

    Args:
        select_rep:

    Returns:

    """
    covariates = get_covariates_df(DATASET_NAME)
    outcomes = get_outcomes_df(DATASET_NAME)

    if select_rep is not None:
        outcomes = select_replication(outcomes, select_rep)

    outcomes[Col.sample_id] = outcomes.groupby(Col.rep).cumcount()
    full = pd.merge(covariates, outcomes, on=Col.sample_id)
    full[Col.ite] = full[Col.mu_1] - full[Col.mu_0]

    cov_names = [col for col in covariates.columns if col != Col.sample_id]
    df = CausalFrame(full, covariates=cov_names)
    return to_rep_list(df)


def get_ihdp_covariates() -> pd.DataFrame:
    """

    TODO: Docstring

    Returns:

    """
    return get_covariates_df(DATASET_NAME).drop(Col.sample_id, axis=1)
