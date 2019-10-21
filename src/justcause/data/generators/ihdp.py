import numpy as np

from . import data_from_generative_function
from ..sets.ihdp import get_ihdp_covariates


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def multi_modal_effect(covariates):
    prob = sigmoid(covariates[:, 7]) > 0.5
    return np.random.normal((3 * prob) + 1 * (1 - prob), 0.1)  # Explicitly multimodal


def exponential_effect(X):
    return np.exp(1 + sigmoid(X[:, 7]))  # use birth weight


def outcome_fct(covariates, setting="multi-modal"):
    Y_0 = np.random.normal(0, 0.2, size=len(covariates))
    Y_1 = Y_0 + multi_modal_effect(covariates)

    return Y_0, Y_1


def expo_outcome_fct(covariates):
    Y_0 = np.random.normal(0, 0.2, size=len(covariates))
    Y_1 = Y_0 + exponential_effect(covariates)
    return Y_0, Y_1


def treatment_assignment(cov):
    """ Assigns treatment based on covariate"""
    return np.random.binomial(1, p=sigmoid(cov[:, 7]))


def multi_expo_on_ihdp(setting="multi-modal", size=None, replications=1):
    """
    Reproduces a specific DGP based on IHDP covariates.

    Y_0 = N(0, 0.2)
    Y_1 = Y_0 + \tau
    T = BERN[sigmoid(X_8)]

    and \tau is either
    \tau = exp(1 + sigmoid(X_8)) # exponential

    or
    c = I(sigmoid(X_8) > 0.5) # indicator based on feature
    \tau = N(3*c + (1 - c), 0.1)

    Args:
        setting: either 'multi-modal' or 'exponential'
        size: number of instances
        replications: number of replications

    Returns: the Bunch containing data and information

    """
    # Use covariates as nd.array
    covariates = get_ihdp_covariates().drop("sample_id", axis="columns").values

    if setting == "multi-modal":
        outcome = outcome_fct
    else:
        outcome = expo_outcome_fct

    return data_from_generative_function(
        covariates, treatment_assignment, outcome, size=size, replications=replications
    )
