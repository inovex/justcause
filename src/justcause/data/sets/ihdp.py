import pandas as pd
from sklearn.datasets.base import Bunch

from . import DATA_PATH
from ..transport import get_local_data_path, load_parquet_dataset


def load_ihdp():
    base = DATA_PATH + "ihdp"

    covariates, replications = load_parquet_dataset(base, "ihdp")

    replications["sample_id"] = replications.groupby("rep").cumcount()
    full = pd.merge(covariates, replications, how="left", on="sample_id")
    full["ite"] = full["y_1"] - full["y_0"]

    cov_names = list(covariates.columns)
    cov_names.remove("sample_id")

    ihdp = Bunch(data=full, covariate_names=cov_names, has_test=True)

    return ihdp


def get_ihdp_covariates():
    url = DATA_PATH + "ihdp/covariates.gzip"
    path = get_local_data_path(url, "ihdp", "covariates")
    covariates = pd.read_parquet(path)
    return covariates
