"""Provides access to the famous IHDP data set.

The IHDP dataset implemented here was first proposed in [1], where J. Hill explains
how the dataset is generated to be biased. Also, upon close inspection, we can see there
that the fundamental overlap condition is not met for the IHDP data. Also, we show in
the thesis [5] that different versions of the IHDP data circulate, and that the R script
used to generate the data is not working anymore.

The actual dataset you can access with the methods in this module is provided by
C. Louizos, author of [2]. The exact same data was used in [3] and [4]. The results
in these paper coincide with the majority of papers analyzed in [5], which is why we
decided to use this exact data instead of reimplementing the data generating process.

References:
    [1] J. L. Hill, “Bayesian Nonparametric Modeling for Causal Inference,”
    J. Comput. Graph. Stat., vol. 20, no. 1, pp. 217–240, 2011.
    [2] C. Louizos, U. Shalit, J. Mooij, D. Sontag, R. Zemel, and M. Welling,
    “Causal Effect Inference with Deep Latent-Variable Models,” 2017.
    [3] F. D. Johansson, U. Shalit, and D. Sontag,
    “Learning Representations for Counterfactual Inference,” 2016.
    [4] [1] U. Shalit, F. D. Johansson, and D. Sontag,
    “Estimating individual treatment effect: generalization bounds and algorithms,”
    2017.
    [5] Maximilian Franz, "A Systematic Review of Machine Learning Estimators for
    Causal Effects", Bachelor Thesis, Karlsruhe Institute of Technology, 2019.
    See `docs/thesis-mfranz.pdf`.

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

DATASET_NAME: str = "ihdp"


def load_ihdp(select_rep: Optional[Indices] = None) -> List[CausalFrame]:
    """Returns the selected replications of the IHDP data

    For perfomance, it is better to use the select_rep parameter to select replications
    instead of working on the list returned by a default call.

    Args:
        select_rep: indices of the replications to select

    Returns:
        data: a list of CausalFrames

    """
    covariates = get_covariates_df(DATASET_NAME)
    outcomes = get_outcomes_df(DATASET_NAME)

    if select_rep is not None:
        outcomes = select_replication(outcomes, select_rep)

    outcomes[Col.sample_id] = outcomes.groupby(Col.rep).cumcount()
    full = pd.merge(covariates, outcomes, on=Col.sample_id)
    full[Col.ite] = full[Col.mu_1] - full[Col.mu_0]
    full = add_pot_outcomes_if_missing(full)

    cov_names = [col for col in covariates.columns if col != Col.sample_id]
    cf = CausalFrame(full, covariates=cov_names)
    return to_rep_list(cf)


def get_ihdp_covariates() -> pd.DataFrame:
    """Returns the original covariates out of the IHDP"""
    return get_covariates_df(DATASET_NAME).drop(Col.sample_id, axis=1)
