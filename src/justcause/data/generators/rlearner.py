"""Re-Implementation of the four DGPs in the R-Learner paper [1].

Exports the generator function, which can be used to access the DGP, and a helper
to transform poential outcomes of the R-Learner into our convention. All other helpers
are not meant for external usage.

In the four settings, different sampling strategies for the covariates are used -
normal and uniform. There are non-linear effects, constant effects and smooth effects.

References:
    [1] X. Nie and S. Wager,
    “Quasi-Oracle Estimation of Heterogeneous Treatment Effects.”,
    4th of February 2019.

"""
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.utils import check_random_state

from ..frames import CausalFrame
from ..utils import generate_data

OptRandState = Optional[Union[int, RandomState]]
Frame = Union[CausalFrame, pd.DataFrame]


def outcomes_from_base(base: np.array, tau: np.array) -> Tuple[np.array, np.array]:
    """Transforms outcomes from the DGP format used in the paper to our convention.

    Args:
        base: the base value for all samples in one replication. This is usually the
            mean of mu_0 and mu_1
        tau: the treatment effect for all samples in one replication, which is later
            implicit in the difference tau = mu_1 - mu_0

    Returns:
        mu_0, mu_1: The potential outcomes without noise
    """

    mu_0 = base - 0.5 * tau
    mu_1 = base + 0.5 * tau
    return mu_0, mu_1


def _normal_covariates(n_samples, n_covariates, random_state=None):
    random_state = check_random_state(random_state)
    return random_state.normal(0, 1, size=n_covariates * n_samples).reshape(
        (n_samples, n_covariates)
    )


def _uniform_covariates(n_samples, n_covariates, random_state=None):
    random_state = check_random_state(random_state)
    return random_state.uniform(0, 1, size=n_covariates * n_samples).reshape(
        (n_samples, n_covariates)
    )


def _outcome_a(covariates, random_state=None):
    X = covariates  # for brevity
    base = (
        np.sin(np.pi * X[:, 0] * X[:, 1])
        + 2 * (X[:, 2] - 0.5) ** 2
        + X[:, 3]
        + 0.5 * X[:, 4]
    )
    tau = (X[:, 0] + X[:, 1]) / 2
    return outcomes_from_base(base, tau)


def _treatment_a(covariates, random_state=None):
    X = covariates  # for brevity
    propensity = np.clip(np.sin(np.pi * X[:, 0] * X[:, 1]), 0.1, 0.9)
    return random_state.binomial(1, p=propensity)


def _outcome_b(covariates, random_state=None):
    X = covariates  # for brevity
    base = np.maximum(0, np.maximum(X[:, 0] + X[:, 1], X[:, 2])) + np.maximum(
        0, X[:, 3] + X[:, 4]
    )
    tau = X[:, 0] + np.log(1 + np.exp(X[:, 1]))
    return outcomes_from_base(base, tau)


def _treatment_b(covariates, random_state=None):
    return random_state.binomial(1, p=0.5, size=len(covariates))


def _outcome_c(covariates, random_state=None):
    X = covariates  # for brevity
    base = 2 * np.log(1 + np.exp(X[:, 0] + X[:, 1] + X[:, 2]))
    tau = np.ones(len(covariates))
    return outcomes_from_base(base, tau)


def _treatment_c(covariates, random_state=None):
    X = covariates  # for brevity
    propensity = 1 / (1 + np.exp(X[:, 1] + X[:, 2]))
    return random_state.binomial(1, p=propensity)


def _outcome_d(covariates, random_state=None):
    X = covariates  # for brevity
    base = (
        np.maximum(X[:, 0] + X[:, 1] + X[:, 2], 0) + np.maximum(X[:, 3] + X[:, 4], 0)
    ) / 2
    tau = np.maximum(X[:, 0] + X[:, 1] + X[:, 2], 0) + np.maximum(X[:, 3] + X[:, 4], 0)
    return outcomes_from_base(base, tau)


def _treatment_d(covariates, random_state=None):
    X = covariates
    propensity = 1 / (1 + np.exp(-1 * X[:, 0]) + np.exp(-1 * X[:, 1]))
    return random_state.binomial(1, p=propensity)


def _make_outcome(outcome_in, sigma):
    def outcome(covariates, random_state, **kwargs):
        random_state = check_random_state(random_state)  # prepare random_state
        noise = sigma * random_state.normal(0, 1, size=len(covariates))
        mu_0, mu_1 = outcome_in(covariates, random_state)
        return mu_0, mu_1, mu_0 + noise, mu_1 + noise

    return outcome


def rlearner_simulation_data(
    n_samples: int = 500,
    n_covariates: int = 6,
    n_replications: int = 500,
    sigma: float = 0.5,
    setting: str = "A",
    random_state: OptRandState = None,
) -> List[Frame]:
    """Provides the simulated experiment data used in the original RLearner paper.

    Defaults are based on the original paper [1]. Note that indices in R start with 1,
    which is why all indices from the original script are used minus one, although it
    should not make a difference, as all covariates are sampled from the
    same distribution.

    References:
        [1] X. Nie and S. Wager,
        “Quasi-Oracle Estimation of Heterogeneous Treatment Effects.”,
        4th of February 2019.

    Args:
        n_samples: Number of samples per replications.
        n_covariates: Number of covariates per instance.
        n_replications: Number of replications.
        sigma: Scalar multiplier of the gaussian noise added to the outcomes.
        setting: the desired setting in {A, B, C, D}.

    Returns:
        data: a list of CausalFrames, one for each replication
    """

    covariates = _normal_covariates(n_samples, n_covariates, random_state)

    if setting == "A":
        covariates = _uniform_covariates(n_samples, n_covariates, random_state)
        outcome = _make_outcome(_outcome_a, sigma)
        treatment = _treatment_a
    elif setting == "B":
        outcome = _make_outcome(_outcome_b, sigma)
        treatment = _treatment_b
    elif setting == "C":
        outcome = _make_outcome(_outcome_c, sigma)
        treatment = _treatment_c
    elif setting == "D":
        outcome = _make_outcome(_outcome_d, sigma)
        treatment = _treatment_d
    else:
        raise AssertionError("setting not known")

    return generate_data(
        covariates,
        treatment,
        outcome,
        n_samples=n_samples,
        n_replications=n_replications,
        random_state=random_state,
    )
