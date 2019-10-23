import numpy as np

from . import data_from_generative_function
from ..sets.ibm_acic import get_ibm_acic_covariates

# Todo: pass RandomState to all functions using randomness
# Use sklearn.utils.check_random_state for this


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_covariates(size, num_covariates):
    rand = np.random.normal(0, 1, size=size * num_covariates)
    return rand.reshape((size, num_covariates))


def simple_treatment(covariates):
    """Random assignment"""
    return np.random.binomial(1, 0.5, size=len(covariates))  # random assignment


def hard_treatment(covariates):
    """Confounded assignment"""
    return np.random.binomial(1, sigmoid(covariates[:, 1]), size=len(covariates))


def simple_outcomes(covariates):
    ite = (
        sigmoid(covariates[:, 2] + covariates[:, 3]) * 3
    )  # make effect large, but all positive
    Y_0 = sigmoid(covariates[:, 1])
    Y_1 = Y_0 + ite
    return Y_0, Y_1


def hard_outcomes(covariates):
    X = covariates
    ite = sigmoid(X[:, 2] + X[:, 3]) / 2
    Y_0 = sigmoid(X[:, 1])
    Y_1 = Y_0 + ite
    return Y_0, Y_1


def toy_example_dgp(setting="simple", size=1000, num_covariates=10, replications=1):
    """Generates a toy example proposed by Stefan Wager"""
    covariates = generate_covariates(size, num_covariates=num_covariates)

    if setting == "simple":
        treatment = simple_treatment
        outcome = simple_outcomes
    else:
        treatment = hard_treatment
        outcome = hard_outcomes

    return data_from_generative_function(
        covariates, treatment, outcome, size=size, replications=replications
    )


def toy_example_emcs(setting="simple", size=1000, num_covariates=10, replications=1):
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

    return data_from_generative_function(
        covariates, treatment, outcome, size=size, replications=replications
    )
