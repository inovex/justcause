import copy
from typing import Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import LassoLars

from ..utils import replace_factual_outcomes

#: Type alias for predict_ite return type
SingleComp = Union[Tuple[np.array, np.array, np.array], np.array]


class TLearner:
    """ Base class for all T-Learners; bundles duplicate code

    Defaults to a LassoLars learner for both treated and control

    """

    def __init__(self, learner=None, learner_c=None, learner_t=None):
        """
        Takes either one base learner for both or two specific base learners

        Args:
            learner: base learner for treatment and control outcomes
            learner_c: base learner for control outcome
            learner_t: base learner for treatment  outcome
        """
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

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """ Simple string representation for logs and outputs"""
        return "{}(control={}, treated={})".format(
            self.__class__.__name__,
            self.learner_c.__class__.__name__,
            self.learner_t.__class__.__name__,
        )

    def fit(
        self, x: np.array, t: np.array, y: np.array, weights: Optional[np.array] = None,
    ) -> None:
        """

        Args:
            x: covariates, shape (num_instances, num_features)
            t: treatment indicator
            y: factual outcomes
            weights: sample weights for weighted fitting
        """
        assert (
            t is not None and y is not None
        ), "treatment and factual outcomes are required to fit Causal Forest"
        if weights is not None:
            assert len(weights) == len(t), "weights must match the number of instances"
            self.learner_c.fit(x[t == 0], y[t == 0], sample_weights=weights)
            self.learner_t.fit(x[t == 1], y[t == 1], sample_weights=weights)
        else:
            # Fit without weights to avoid unknown argument error
            self.learner_c.fit(x[t == 0], y[t == 0])
            self.learner_t.fit(x[t == 1], y[t == 1])

    def predict_ite(
        self,
        x: np.array,
        t: np.array = None,
        y: np.array = None,
        return_components: bool = False,
        replace_factuals: bool = False,
    ) -> SingleComp:
        """ Predicts ITE for the given samples

        Args:
            x: covariates in shape (num_instances, num_features)
            t: treatment indicator, binary in shape (num_instances)
            y: factual outcomes in shape (num_instances)
            return_components: whether to return Y(0) and Y(1) predictions separately
            replace_factuals

        Returns: a vector of ITEs for the inputs;
            also returns Y(0) and Y(1) for all inputs if return_components is True
        """
        y_0 = self.learner_c.predict(x)
        y_1 = self.learner_t.predict(x)
        if return_components:
            if t is not None and y is not None and replace_factuals:
                y_0, y_1 = replace_factual_outcomes(y_0, y_1, y, t)
            return y_1 - y_0, y_0, y_1
        else:
            return y_1 - y_0

    def estimate_ate(
        self,
        x: np.array,
        t: np.array = None,
        y: np.array = None,
        weights: Optional[np.array] = None,
    ) -> float:
        """ Estimates the average treatment effect of the given population

        First, it fits the model on the given population, then predicts ITEs and uses
        the mean as an estimate for the ATE

        Args:
            x: covariates
            t: treatment indicator
            y: factual outcomes
            weights: sample weights for weighted fitting

        Returns: ATE estimate as the mean of ITEs
        """
        self.fit(x, t, y, weights)
        ite = self.predict_ite(x, t, y)
        return float(np.mean(ite))
