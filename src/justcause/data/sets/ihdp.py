from typing import Iterable, Optional

import pandas as pd

from ..frames import CausalFrame
from ..utils import (
    Indices,
    get_covariates_df,
    get_outcomes_df,
    iter_rep,
    select_replication,
)

DATASET_NAME = "ihdp"


def load_ihdp(select_rep: Optional[Indices] = None) -> Iterable[CausalFrame]:
    covariates = get_covariates_df(DATASET_NAME)
    outcomes = get_outcomes_df(DATASET_NAME)
    # Drop test column for now. Test set as defined in
    # Christos Louizos et al. Causal Effect Inference with
    # Deep Latent-Variable Models. Tech. rep. 2017.
    del outcomes["test"]

    if select_rep is not None:
        outcomes = select_replication(outcomes, select_rep)

    outcomes["sample_id"] = outcomes.groupby("rep").cumcount()
    full = pd.merge(covariates, outcomes, on="sample_id")
    full["ite"] = full["y_1"] - full["y_0"]

    cov_names = [col for col in covariates.columns if col != "sample_id"]
    df = CausalFrame(full, covariates=cov_names)
    return iter_rep(df)


def get_ihdp_covariates() -> pd.DataFrame:
    return get_covariates_df(DATASET_NAME)
