import numpy as np
from scipy.special import expit

from . import generate_data
from ..sets.ibm_acic import get_ibm_acic_covariates

# Todo: pass RandomState to all functions using randomness
# Use sklearn.utils.check_random_state for this


def generate_covariates(size, num_covariates):
    """ Generates covariates sampled from normal distribution """
    rand = np.random.normal(0, 1, size=size * num_covariates)
    return rand.reshape((size, num_covariates))


def simple_treatment(covariates):
    """Random assignment"""
    return np.random.binomial(1, 0.5, size=len(covariates))  # random assignment


def hard_treatment(covariates):
    """Confounded assignment"""
    return np.random.binomial(1, expit(covariates[:, 1]), size=len(covariates))


def simple_outcomes(covariates):
    ite = (
        expit(covariates[:, 2] + covariates[:, 3]) * 3
    )  # make effect large, but all positive
    Y_0 = expit(covariates[:, 1])
    Y_1 = Y_0 + ite
    return Y_0, Y_1


def hard_outcomes(covariates):
    Y_0 = expit(covariates[:, 1])
    Y_1 = Y_0 + expit(covariates[:, 2] + covariates[:, 3]) / 2
    return Y_0, Y_1


def toy_data_synthetic(setting="simple", size=1000, num_features=10, replications=1):
    """
    Generates a toy example proposed by Stefan Wager.

    The idea is that simple setting has a larger treatment effect and
    is thus easier for the estimators to grasp.

    Args:
        setting: 'simple' or 'hard'
        size: number of instances
        num_features: number of covariates per instance
        replications: number of replications of the process

    Returns: the data Bunch containin the requested DGP

    """
    covariates = generate_covariates(size, num_covariates=num_features)

    if setting == "simple":
        treatment = simple_treatment
        outcome = simple_outcomes
    else:
        treatment = hard_treatment
        outcome = hard_outcomes

    return generate_data(
        covariates, treatment, outcome, size=size, replications=replications
    )


def toy_example_emcs(setting="simple", size=1000, replications=1):
    """Generates the same toy example but on real covariates"""
    covariates = get_ibm_acic_covariates().values
    if size > len(covariates):
        raise AssertionError(
            "requested size {} is bigger than available covariates {}".format(
                size, len(covariates)
            )
        )

    if setting == "simple":
        treatment = simple_treatment
        outcome = simple_outcomes
    else:
        treatment = hard_treatment
        outcome = hard_outcomes

    return generate_data(
        covariates, treatment, outcome, size=size, replications=replications
    )
