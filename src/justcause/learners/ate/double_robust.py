import copy
from typing import Optional

import numpy as np
from sklearn.linear_model import LassoLars

from ..propensity import estimate_propensities


class DoubleRobustEstimator(object):
    """Implements Double Robust Estmation with generic learners based on the
        equations of M. Davidian


    References:
        [1] M. Davidian, “Double Robustness in Estimation of Causal Treatment Effects”
                2007. Presentation
                http://www.stat.ncsu.edu/∼davidian North

    """

    def __init__(
        self,
        propensity_learner=None,
        learner=None,
        learner_c=None,
        learner_t=None,
        delta=0.001,
    ):
        """Inits the DoubleRobustEstimator

        Args:
            propensity_learner: a classifier model with probability estimation method
                `predict_proba` like a sklearn LogisticRegression
            learner: generic outcome regression model for both outcomes
            learner_c: specific control outcome regression model
            learner_t: specific treatment outcome regression model
        """
        self.propensity_learner = propensity_learner

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

        self.delta = delta

    def __str__(self):
        """Simple string representation for logs and outputs"""
        return "{}(control={}, treated={}, propensity={})".format(
            self.__class__.__name__,
            self.learner_c.__class__.__name__,
            self.learner_t.__class__.__name__,
            self.propensity_learner.__class__.__name__,
        )

    def estimate_ate(
        self,
        x: np.array,
        t: np.array,
        y: np.array,
        propensity: Optional[np.array] = None,
    ) -> float:
        """Estimates average treatment effect of the given population

        Args:
            x: covariates in shape (num_instances, num_features)
            t: binary treatment indicator vector of shape (num_instances)
            y: factual outcomes of shape (num_instances)
            propensity: explicit propensity scores of all instances to be used for
                weighting in the double robust estimation formula

        Returns:
            ate: estimate of the average treatment effect for the population

        """

        self._fit(x, t, y)

        if propensity is None:
            # estimate propensity if not given
            if self.propensity_learner is None:
                propensity = estimate_propensities(x, t)
            else:
                self.propensity_learner.fit(x, t)
                propensity = self.propensity_learner.predict_proba(x)[:, 1]

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
        return float(dr1 - dr0)

    def _fit(self, x: np.array, t: np.array, y: np.array) -> None:
        """Helper to fit the outcome learners on treated and control separately"""
        self.learner_c.fit(x[t == 0], y[t == 0])
        self.learner_t.fit(x[t == 1], y[t == 1])
