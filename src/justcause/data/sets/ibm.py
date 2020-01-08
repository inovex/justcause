"""Provides access to *one* setting of the the IBM ACIC challenge data.

The challenge data and further explanations of the corresponding
data can be found in [2]. The authors also provide a rough evaluation guideline
in their paper [1], which unfortunately has not been maintained since publication.

References:
    [1] Y. Shimoni, C. Yanover, E. Karavani, and Y. Goldschmnidt,
    “Benchmarking Framework for Performance-Evaluation of Causal Inference Analysis,”
    2018.

    [2] Data Set Download: https://www.synapse.org/#!Synapse:syn11738767/wiki/512854

"""
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

DATASET_NAME: str = "ibm_acic"


def load_ibm_acic(select_rep: Optional[Indices] = None) -> List[CausalFrame]:
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

    full = pd.merge(covariates, outcomes, on=Col.sample_id)
    full[Col.ite] = full[Col.mu_1] - full[Col.mu_0]

    cov_names = [col for col in covariates.columns if col != Col.sample_id]
    df = CausalFrame(full, covariates=cov_names)
    return to_rep_list(df)


def get_ibm_acic_covariates() -> pd.DataFrame:
    """

    TODO: Docstring

    Returns:

    """
    return get_covariates_df(DATASET_NAME).drop(Col.sample_id, axis=1)
