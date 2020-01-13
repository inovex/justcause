"""Provides access to the so called Twins data set.

The Twins dataset is first compounded and analyzed in [1]. The version here is the
exact data used in [2] provided
[here](https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/TWINS).

References:
    [1] D. Almond, K. Y. Chay, and D. S. Lee,
    “The costs of low birth weight,” Quarterly Journal of Economics. 2005.
    [2] [1] C. Louizos, U. Shalit, J. Mooij, D. Sontag, R. Zemel, and M. Welling,
    “Causal Effect Inference with Deep Latent-Variable Models,” 2017.

"""
from typing import List

import numpy as np
import pandas as pd

from ..frames import CausalFrame, Col
from ..transport import get_covariates_df, get_outcomes_df
from ..utils import add_pot_outcomes_if_missing, to_rep_list

DATASET_NAME: str = "twins"


def load_twins() -> List[CausalFrame]:
    """Returns the Twins dataset as a list of one replication.

    There are no replications in the twins dataset, thus no is selection required. For
    consistency, the data is still returned as a list of on CausalFrame.

    Returns:
        data: a list with one CausalFrame for the one replication

    """
    covariates = get_covariates_df(DATASET_NAME)
    outcomes = get_outcomes_df(DATASET_NAME)

    full = pd.merge(covariates, outcomes, on=Col.sample_id)
    full[Col.ite] = full[Col.mu_1] - full[Col.mu_0]
    full[Col.rep] = np.repeat(0, len(full))
    full = add_pot_outcomes_if_missing(full)

    cov_names = [col for col in covariates.columns if col != Col.sample_id]
    cf = CausalFrame(full, covariates=cov_names)
    return to_rep_list(cf)


def get_twins_covariates() -> pd.DataFrame:
    """Returns the covariates of the original Twins dataset."""
    return get_covariates_df(DATASET_NAME).drop(Col.sample_id, axis=1)
