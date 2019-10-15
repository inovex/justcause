import numpy as np
import pandas as pd

from . import data_from_generative_function
from ..sets import DATA_PATH
from ..transport import get_local_data_path


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def multi_modal_effect(covariates):
    prob = sigmoid(covariates[:, 7]) > 0.5
    return np.random.normal((3 * prob) + 1 * (1 - prob), 0.1)  # Explicitly multimodal


def exponential_effect(X):
    return np.exp(1 + sigmoid(X[:, 7]))  # use birth weight


def get_ihdp_covariates():
    url = DATA_PATH + "ihdp/covariates.gzip"
    path = get_local_data_path(url, "ihdp", "covariates")
    covariates = pd.read_parquet(path)
    return covariates


def outcome_fct(covariates, setting="multi-modal"):
    Y_0 = np.random.normal(0, 0.2, size=len(covariates))

    if setting == "multi-modal":
        Y_1 = Y_0 + multi_modal_effect(covariates)
    else:
        Y_1 = Y_0 + exponential_effect(covariates)

    return Y_0, Y_1


def treatment_assignment(cov):
    return np.random.binomial(1, p=sigmoid(cov[:, 7]))


def multi_expo_on_ihdp(setting="multi-modal", size=None, replications=1):
    covariates = get_ihdp_covariates()
    return data_from_generative_function(
        covariates, treatment_assignment, outcome_fct, replications=replications
    )
