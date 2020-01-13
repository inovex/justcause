"""Provides access to *one* setting of the the IBM benchmarking data.

The challenge data and further explanations of the corresponding
data can be found in [2]. The authors also provide a rough evaluation guideline
in their paper [1], which unfortunately has not been maintained since publication.
The DGP is based on the same covariates as the ACIC2018 [3] challenge, which we hope to
implement as a data set in a future version.

References:
    [1] Y. Shimoni, C. Yanover, E. Karavani, and Y. Goldschmnidt,
    “Benchmarking Framework for Performance-Evaluation of Causal Inference Analysis,”
    2018.

    [2] Data Set Download: https://www.synapse.org/#!Synapse:syn11738767/wiki/512854

    [3] ACIC2018 challenge: https://www.synapse.org/#!Synapse:syn11294478/wiki/494269

"""
from typing import List, Optional

import pandas as pd

from ..frames import CausalFrame, Col
from ..transport import get_covariates_df, get_outcomes_df
from ..utils import (
    Indices,
    add_pot_outcomes_if_missing,
    select_replication,
    to_rep_list,
)

DATASET_NAME: str = "ibm"


def load_ibm(select_rep: Optional[Indices] = None) -> List[CausalFrame]:
    """Provides the IBM benchmarking data in the common JustCause format.

    BEWARE: the replications have different sizes and should be used with caution.

    Args:
        select_rep: the desired replications

    Returns:
        data: list of CausalFrames, one for each replication

    """
    covariates = get_covariates_df(DATASET_NAME)
    outcomes = get_outcomes_df(DATASET_NAME)

    if select_rep is not None:
        outcomes = select_replication(outcomes, select_rep)

    full = pd.merge(covariates, outcomes, on=Col.sample_id)
    full[Col.ite] = full[Col.mu_1] - full[Col.mu_0]
    full = add_pot_outcomes_if_missing(full)

    cov_names = [col for col in covariates.columns if col != Col.sample_id]
    cf = CausalFrame(full, covariates=cov_names)
    return to_rep_list(cf)


def get_ibm_covariates() -> pd.DataFrame:
    """Return the covariates of the IBM benchmarking data

    The same covariates are used in the ACIC2018 challenge.

    """
    return get_covariates_df(DATASET_NAME).drop(Col.sample_id, axis=1)
