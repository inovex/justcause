import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch

from . import DATA_PATH
from ..transport import get_local_data_path, load_parquet_dataset


def load_twins():
    base = DATA_PATH + "twins"

    covariates, outcomes = load_parquet_dataset(base, "twins")

    full = pd.merge(covariates, outcomes, how="left", on="sample_id")
    full["rep"] = np.repeat(0, len(full))

    cov_names = list(covariates.columns)
    cov_names.remove("sample_id")

    twins = Bunch(data=full, covariate_names=cov_names, has_test=True)

    return twins


def get_twins_covariates():
    url = DATA_PATH + "twins/covariates.gzip"
    path = get_local_data_path(url, "twins", "covariates.gzip")
    covariates = pd.read_parquet(path)
    return covariates
