import numpy as np

from ..utils import replace_factual_outcomes


class BaseSLearner(object):
    """ Base class for all S-Learners; bundles duplicate code"""

    def __init__(self, regressor):
        """

        :param regressor: a sklearn regressor with learners `fit` and `predict`
        """
        self.regressor = regressor

    def __repr__(self):
        return self.__str__()

    def predict_ite(
        self, x, t=None, y=None, return_components=False, replace_factuals=False
    ):
        """

        Args:
            x: covariates in shape (num_instances, num_features)
            t: treatment indicator, binary in shape (num_instances)
            y: factual outcomes in shape (num_instances)
            return_components: whether to return Y(0) and Y(1) predictions separately
            replace_factuals: Whether to use the given factuals in the prediction
        Returns:

        """
        y_0 = self.regressor.predict(np.c_[x, np.zeros(x.shape[0])])
        y_1 = self.regressor.predict(np.c_[x, np.ones(x.shape[0])])

        if t is not None and y is not None and replace_factuals:
            # Use factuals outcomes where possible
            assert len(t) == len(y), "outcome and treatment must be of same length"
            assert len(t) == len(y_0), "treatment indicators must match covariates"
            y_0, y_1 = replace_factual_outcomes(y_0, y_1, y, t)

        if return_components:
            return y_1 - y_0, y_0, y_1
        else:
            return y_1 - y_0

    def predict_ate(self, x, t=None, y=None):
        """ Predicts ATE as a mean of ITE predictions"""
        return np.mean(self.predict_ite(x, t, y))


class SLearner(BaseSLearner):
    """
    Implements a generic S-Learner for the binary treatment case

    :ref:
    [S-Learner](﻿https://arxiv.org/pdf/1706.03461.pdf)
    """

    def __str__(self):
        return ("{}(regressor={})").format(
            self.__class__.__name__, self.regressor.__class__.__name__
        )

    def fit(self, x, t, y) -> None:
        """ Fits the base learner on the outcome function"""
        train = np.c_[x, t]
        self.regressor.fit(train, y)


class WeightedSLearner(BaseSLearner):
    """
    Implements a weighted generic S-Learner, where samples are weighted with the
    inverse probability of treatment

    References
    [S-Learner](﻿https://arxiv.org/pdf/1706.03461.pdf)
    """

    def __init__(self, propensity_regressor, regressor):
        """
        :param regressor: a sklearn regressor with methods `fit` and `predict`
        """
        super().__init__(regressor)
        assert (
            "predict_proba" in propensity_regressor
        ), "propensity regressor must have predict_proba function"

        self.propensity_regressor = propensity_regressor

    def fit(self, x, t, y, propensity=None) -> None:
        if propensity is not None:
            assert len(propensity) == len(t)
            ipt = 1 / propensity
        else:
            self.propensity_regressor.fit(x, t)
            ipt = 1 / self.propensity_regressor.predict_proba(x)

        train = np.c_[x, t]
        self.regressor.fit(train, y, sample_weight=ipt)
