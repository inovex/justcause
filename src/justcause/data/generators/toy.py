"""Implementation of a DGP by Stefan Wager to show the quality of Causal Forests.

The two settings were defined to show in which setting CausalForests outperform a
simple T-Learner. See Chapter 5.5 in the thesis [1].

References:
    [1] Maximilian Luca Franz, "A Systematic Review of Machine Learning Estimators for
        Causal Effects", Bachelor Thesis, Karlsruhe Institute of Technology, 2019.
        See `docs/thesis-mfranz.pdf`.

"""
from typing import List

from numpy.random import RandomState
from scipy.special import expit

from ..sets.ibm import get_ibm_covariates
from ..utils import Frame, OptRandState, generate_data


def _generate_covariates(size, num_covariates, random_state: RandomState):
    rand = random_state.normal(0, 1, size=size * num_covariates)
    return rand.reshape((size, num_covariates))


def _simple_treatment(covariates, *, random_state: RandomState, **kwargs):
    return random_state.binomial(1, 0.5, size=len(covariates))  # random assignment


def _hard_treatment(covariates, *, random_state: RandomState, **kwargs):
    return random_state.binomial(1, expit(covariates[:, 1]), size=len(covariates))


def _simple_outcomes(covariates, *, random_state: RandomState, **kwargs):
    ite = expit(covariates[:, 2] + covariates[:, 3]) * 3
    y_0 = expit(covariates[:, 1])
    y_1 = y_0 + ite
    mu_0, mu_1 = y_0, y_1
    return mu_0, mu_1, y_0, y_1


def _hard_outcomes(covariates, *, random_state: RandomState, **kwargs):
    y_0 = expit(covariates[:, 1])
    y_1 = y_0 + expit(covariates[:, 2] + covariates[:, 3]) / 2
    mu_0, mu_1 = y_0, y_1
    return mu_0, mu_1, y_0, y_1


def toy_data_synthetic(
    setting: str = "simple",
    n_samples: int = 1000,
    num_features: int = 10,
    n_replications: int = 1,
    random_state: OptRandState = None,
) -> List[Frame]:
    """Generates the toy example proposed by Stefan Wager.

    The idea is that simple setting has a larger treatment effect and
    is thus easier for the estimators to grasp.

    Args:
        setting: desired setting, either 'simple' or 'hard'
        n_samples: number of samples per replication
        num_features: number of covariates per instance
        n_replications: number of replications
        random_state: a RandomState or seed from which to draw random numbers

    Returns:
        data: a list of CausalFrames, one for each replication

    """
    if setting == "simple":
        treatment = _simple_treatment
        outcome = _simple_outcomes
    elif setting == "hard":
        treatment = _hard_treatment
        outcome = _hard_outcomes
    else:
        raise AssertionError("setting not known")

    return generate_data(
        _generate_covariates,
        treatment,
        outcome,
        n_samples=n_samples,
        n_replications=n_replications,
        random_state=random_state,
        num_covariates=num_features,
    )


def toy_data_emcs(
    setting: str = "simple",
    n_samples: int = 1000,
    n_replications: int = 1,
    random_state: RandomState = None,
) -> List[Frame]:
    """Generates the toy dataset based on the covariates of the IBM ACIC dataset

    Args:
        setting: desired setting, either 'simple' or 'hard'
        n_samples: number of samples per replication
        n_replications: number of replications
        random_state: a RandomState or seed from which to draw random numbers

    Returns:
        data: a list of CausalFrames, one for each replication
    """
    covariates = get_ibm_covariates().to_numpy()
    assert (
        n_samples < covariates.shape[0]
    ), "requested size {} is bigger than available covariates {}".format(
        n_samples, covariates.shape[0]
    )

    if setting == "simple":
        treatment = _simple_treatment
        outcome = _simple_outcomes
    elif setting == "hard":
        treatment = _hard_treatment
        outcome = _hard_outcomes
    else:
        raise RuntimeError("Undefined setting {}".format(setting))

    return generate_data(
        covariates,
        treatment,
        outcome,
        n_samples=n_samples,
        n_replications=n_replications,
        random_state=random_state,
    )
