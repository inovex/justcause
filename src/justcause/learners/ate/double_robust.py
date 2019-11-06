import copy

import numpy as np
from sklearn.linear_model import LassoLars

from ..utils import fit_predict_propensity, set_propensity_learner


class DoubleRobustEstimator:
    """ Implements Double Robust Estmation with generic learners based on the
        equations of M. Davidian


    References:
        [1] M. Davidian, “Double Robustness in Estimation of Causal Treatment Effects”
                2007. Presentation
                http://www.stat.ncsu.edu/∼davidian North

    """

    def __init__(
        self, propensity_learner=None, learner=None, learner_c=None, learner_t=None
    ):
        """

        Args:
            propensity_learner: propensity regression model
            learner: generic outcome learner for both outcomes
            learner_c: specific control outcome learner
            learner_t: specific treatment outcome learner
        """
        self.propensity_learner = set_propensity_learner(propensity_learner)

        if learner is None:
            if learner_c is None and learner_t is None:
                self.learner_c = LassoLars()
                self.learner_t = LassoLars()
            else:
                self.learner_c = learner_c
                self.learner_t = learner_t

        else:
            self.learner_c = copy.deepcopy(learner)
            self.learner_t = copy.deepcopy(learner)

        # TODO: This is not very clean here
        self.delta = 0.0001

    def __str__(self):
        """ Simple string representation for logs and outputs"""
        return "{}(control={}, treated={}, propensity={})".format(
            self.__class__.__name__,
            self.learner_c.__class__.__name__,
            self.learner_t.__class__.__name__,
            self.propensity_learner.__class__.__name__,
        )

    def predict_ate(self, x, t, y, propensity=None):
        """ **Fits** and Predicts average treatment effect of the given population"""
        # TODO: Out-of-sample prediction makes little sense here

        self.fit(x, t, y)

        # predict propensity
        if propensity is None:
            propensity = fit_predict_propensity(self.propensity_learner, x, t)

        dr1 = (
            np.sum(
                ((t * y) / (propensity + self.delta))
                - ((t - propensity + self.delta) / (propensity + self.delta))
                * self.learner_t.predict(x)
            )
            / x.shape[0]
        )
        dr0 = (
            np.sum(
                ((1 - t) * y / (1 - propensity + self.delta))
                - ((t - propensity + self.delta) / (1 - propensity + self.delta))
                * self.learner_c.predict(x)
            )
            / x.shape[0]
        )
        return dr1 - dr0

    def fit(self, x, t, y) -> None:
        """ Fits the outcome models on treated and control separately"""
        self.learner_c.fit(x[t == 0], y[t == 0])
        self.learner_t.fit(x[t == 1], y[t == 1])
