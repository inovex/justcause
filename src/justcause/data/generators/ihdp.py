import numpy as np
from scipy.special import expit

from ..sets.ihdp import get_ihdp_covariates
from ..utils import generate_data


def multi_modal_effect(covariates):
    prob = expit(covariates[:, 7]) > 0.5
    return np.random.normal((3 * prob) + 1 * (1 - prob), 0.1)  # Explicitly multimodal


def exponential_effect(covariates):
    return np.exp(1 + expit(covariates[:, 7]))  # use birth weight


def multi_outcome(covariates):
    Y_0 = np.random.normal(0, 0.2, size=len(covariates))
    Y_1 = Y_0 + multi_modal_effect(covariates)
    return Y_0, Y_1


def expo_outcome(covariates):
    Y_0 = np.random.normal(0, 0.2, size=len(covariates))
    Y_1 = Y_0 + exponential_effect(covariates)
    return Y_0, Y_1


def treatment_assignment(covariates):
    """ Assigns treatment based on covariate """
    return np.random.binomial(1, p=expit(covariates[:, 7]))


def multi_expo_on_ihdp(setting="multi-modal", n_samples=None, n_replications=1):
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
        n_samples: number of instances
        n_replications: number of replications

    Returns: the Bunch containing data and information

    """
    # Use covariates as nd.array
    covariates = get_ihdp_covariates().values

    if setting == "multi-modal":
        outcome = multi_outcome
    else:
        outcome = expo_outcome

    return generate_data(
        covariates,
        treatment_assignment,
        outcome,
        n_samples=n_samples,
        n_replications=n_replications,
    )
