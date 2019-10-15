import pandas as pd
from sklearn.datasets.base import Bunch

from . import DATA_PATH
from ..transport import load_parquet_dataset


def load_ibm_acic():
    dest_path = "ibm_acic"
    url = DATA_PATH + "ibm_acic/"

    covariates, replications = load_parquet_dataset(url, dest_subdir=dest_path)

    full = pd.merge(covariates, replications, how="left", on="sample_id")
    full["ite"] = full["y_1"] - full["y_0"]

    cov_names = list(covariates.columns)
    cov_names.remove("sample_id")

    acic = Bunch(data=full, covariate_names=cov_names, has_test=False)

    return acic
