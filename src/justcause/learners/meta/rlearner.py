"""Wrapper of the python RLearner implemented in the ``causalml`` package"""
from typing import Optional, Union

import numpy as np
from numpy.random import RandomState
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state

from ..propensity import estimate_propensities

StateType = Optional[Union[int, RandomState]]


class RLearner:
    """A wrapper of the BaseRRegressor from ``causalml``

    Defaults to LassoLars regression as a base learner if not specified otherwise.
    Allows to either specify one learner for both tasks or two distinct learners
    for the task outcome and effect learning.

    References:
        CausalML Framework `on Github <https://github.com/uber/causalml/>'_.

        [1] X. Nie and S. Wager,
            “Quasi-Oracle Estimation of Heterogeneous Treatment Effects.”
    """

    def __init__(
        self,
        learner=None,
        outcome_learner=None,
        effect_learner=None,
        random_state: StateType = None,
    ):
        """Setup an RLearner

        Args:
            learner: default learner for both outcome and effect
            outcome_learner: specific learner for outcome
            effect_learner: specific learner for effect
            random_state: RandomState or int to be used for K-fold splitting. NOT used
                in the learners, this has to be done by the user.
        """
        from causalml.inference.meta import BaseRRegressor

        if learner is None and (outcome_learner is None and effect_learner is None):
            learner = LinearRegression()

        self.random_state = check_random_state(random_state)
        self.model = BaseRRegressor(
            learner, outcome_learner, effect_learner, random_state=random_state
        )

    def __str__(self):
        """Simple string representation for logs and outputs"""
        return "{}(outcome={}, effect={})".format(
            self.__class__.__name__,
            self.model.model_mu.__class__.__name__,
            self.model.model_tau.__class__.__name__,
        )

    def __repr__(self):
        return self.__str__()

    def fit(self, x: np.array, t: np.array, y: np.array, p: np.array = None) -> None:
        """Fits the RLearner on given samples.

        Defaults to `justcause.learners.propensities.estimate_propensities`
        for ``p`` if not given explicitly, in order to allow a generic call
        to the fit() method

        Args:
            x: covariate matrix of shape (num_instances, num_features)
            t: treatment indicator vector, shape (num_instances)
            y: factual outcomes, (num_instances)
            p: propensities, shape (num_instances)

        """
        if p is None:
            # Propensity is needed by CausalML, so we estimate it,
            # if it was not provided
            p = estimate_propensities(x, t)

        self.model.fit(x, p, t, y)

    def predict_ite(self, x: np.array, *args) -> np.array:
        """Predicts ITE for given samples; ignores the factual outcome and treatment

        Args:
            x: covariates used for precition
            *args: NOT USED but kept to work with the standard ``fit(x, t, y)`` call

        """

        # assert t is None and y is None, "The R-Learner does not use factual outcomes"
        return self.model.predict(x).flatten()

    def estimate_ate(
        self, x: np.array, t: np.array, y: np.array, p: Optional[np.array] = None
    ) -> float:
        """Estimate the average treatment effect (ATE) by fit and predict on given data

        Estimates the ATE as the mean of ITE predictions on the given data.

        Args:
            x: covariates of shape (num_samples, num_covariates)
            t: treatment indicator vector, shape (num_instances)
            y: factual outcomes, (num_instances)
            p: propensities, shape (num_instances)

        Returns:
            the average treatment effect estimate


        """
        self.fit(x, t, y, p)
        ite = self.predict_ite(x, t, y)
        return float(np.mean(ite))
