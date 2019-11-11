from typing import Optional, Tuple, Union

import numpy as np

from ..utils import replace_factual_outcomes

# Return Type of predict_ite
SingleComp = Union[Tuple[np.array, np.array, np.array], np.array]


class SLearner(object):
    """ Generic S-Learner for the binary treatment case

    :ref:
    [S-Learner](﻿https://arxiv.org/pdf/1706.03461.pdf)
    """

    def __init__(self, learner):
        """
        Args:
            learner: a sklearn regressor with learners `fit` and `predict`
        """
        self.learner = learner

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """ Simple string representation for logs and outputs"""
        return ("{}(learner={})").format(
            self.__class__.__name__, self.learner.__class__.__name__
        )

    def fit(
        self, x: np.array, t: np.array, y: np.array, weights: Optional[np.array] = None,
    ) -> None:
        """ Fits (optionally weighted) learner on the given samples

        Args:
            x: covariates
            t: treatment indicator
            y: factual outcomes
            weights: weights to be used by the learner
        """
        train = np.c_[x, t]
        if weights is not None:
            assert len(weights) == len(t), "weights must match the number of instances"
            self.learner.fit(train, y, sample_weight=weights)
        else:
            # Fit without weights to avoid unknown argument error
            self.learner.fit(train, y)

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
            replace_factuals: Whether to use the given factuals in the prediction

        Returns: a vector of ITEs for the inputs;
            also returns Y(0) and Y(1) for all inputs if return_components is True
        """
        y_0 = self.learner.predict(np.c_[x, np.zeros(x.shape[0])])
        y_1 = self.learner.predict(np.c_[x, np.ones(x.shape[0])])

        if t is not None and y is not None and replace_factuals:
            # Use factuals outcomes where possible
            assert len(t) == len(y), "outcome and treatment must be of same length"
            assert len(t) == len(y_0), "treatment indicators must match covariates"
            y_0, y_1 = replace_factual_outcomes(y_0, y_1, y, t)

        if return_components:
            return y_1 - y_0, y_0, y_1
        else:
            return y_1 - y_0

    def estimate_ate(
        self, x: np.array, t: np.array = None, y: np.array = None
    ) -> float:
        """Estimates the ATE of the given population

        First, fits the learner on the population, then uses the mean of ITE
        predicitions as the ATE estimate

        Args:
            x: covariates
            t: treatment
            y: factual outcomes

        Returns: ATE estimate for the given population

        """
        self.fit(x, t, y)
        ite = self.predict_ite(x, t, y)
        return float(np.mean(ite))
