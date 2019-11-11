from typing import Optional, Union

import numpy as np
from causalml.inference.meta import BaseRRegressor
from causalml.propensity import ElasticNetPropensityModel
from numpy.random import RandomState
from sklearn.linear_model import LassoLars
from sklearn.utils import check_random_state

StateType = Optional[Union[int, RandomState]]


class RLearner:
    """ An adapter to the BaseRRegressor from causalml

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
        """

        Args:
            learner: default learner for both outcome and effect
            outcome_learner: specific learner for outcome
            effect_learner: specific learner for effect
            random_state: RandomState or int to be used for K-fold splitting. NOT used
                in the learners, this has to be done by the user.
        """
        if learner is None and (outcome_learner is None and effect_learner is None):
            learner = LassoLars()

        self.random_state = check_random_state(random_state)
        self.model = BaseRRegressor(
            learner, outcome_learner, effect_learner, random_state=random_state
        )

    def __str__(self):
        """ Simple string representation for logs and outputs"""
        return "{}(outcome={}, effect={})".format(
            self.__class__.__name__,
            self.model.model_mu.__class__.__name__,
            self.model.model_tau.__class__.__name__,
        )

    def __repr__(self):
        return self.__str__()

    def fit(self, x: np.array, t: np.array, y: np.array, p: np.array = None) -> None:
        """ Fits the RLearner on given samples

        Defaults to ElasticNetPropensityModel for propensity if not given expclicitly,
        in order to allow a generic call to fit()

        Args:
            x: covariate matrix of shape (num_instances, num_features)
            t: treatment indicator vector, shape (num_instances)
            y: factual outcomes, (num_instances)
            p: propensities, shape (num_instances)

        Returns: None
        """
        # TODO: Replace this with a default_propensity from utils
        # TODO: Maybe just assert that propensity is not None and explain why
        #  --> responsibility with the user
        if p is None:
            # Propensity is needed by CausalML, so we estimate it,
            # if it was not provided
            p_learner = ElasticNetPropensityModel()
            p = p_learner.fit_predict(x, t)

        self.model.fit(x, p, t, y)

    def predict_ite(
        self, x: np.array, t: np.array = None, y: np.array = None
    ) -> np.array:
        """ Predicts ITE for given samples; ignores the factual outcome and treatment"""

        assert t is None and y is None, "The R-Learner does not use factual outcomes"
        return self.model.predict(x)

    def estimate_ate(
        self, x: np.array, t: np.array, y: np.array, p: Optional[np.array] = None
    ):
        self.fit(x, t, y, p)
        ite = self.predict_ite(x, t, y)
        return float(np.mean(ite))
