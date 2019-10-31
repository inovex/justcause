from typing import Optional

import numpy as np
from causalml.propensity import ElasticNetPropensityModel

from ..utils import replace_factual_outcomes


class BaseSLearner(object):
    """ Base class for all S-Learners; bundles duplicate code"""

    def __init__(self, learner):
        """
        Args:
            learner: a sklearn regressor with learners `fit` and `predict`
        """
        self.learner = learner

    def __repr__(self):
        return self.__str__()

    def predict_ite(
        self,
        x: np.array,
        t: np.array = None,
        y: np.array = None,
        return_components: bool = False,
        replace_factuals: bool = False,
    ) -> np.array:
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

    def predict_ate(self, x: np.array, t: np.array = None, y: np.array = None):
        """Predicts ATE as a mean of ITE predictions

        Args:
            x: covariates
            t: treatment
            y: factual outcomes

        Returns: ATE for given data

        """
        return np.mean(self.predict_ite(x, t, y))


class SLearner(BaseSLearner):
    """ Generic S-Learner for the binary treatment case

    :ref:
    [S-Learner](﻿https://arxiv.org/pdf/1706.03461.pdf)
    """

    def __str__(self):
        """ Simple string representation for logs and outputs"""
        return ("{}(regressor={})").format(
            self.__class__.__name__, self.learner.__class__.__name__
        )

    def fit(self, x, t, y) -> None:
        """ Fits the base learner on the outcome function"""
        train = np.c_[x, t]
        self.learner.fit(train, y)


class WeightedSLearner(BaseSLearner):
    """ Weighted generic S-Learner

    Samples are weighted before prediction with the inverse probability of
    treatment.

    References
    [S-Learner](﻿https://arxiv.org/pdf/1706.03461.pdf)
    """

    # TODO: Add CalibratedCV in order to ensure correct probabilites
    def __init__(self, learner, propensity_learner=None):
        """
        Checks if the given propenstiy_regressor has the predict_proba function
        as required.

        Args:
            learner: The outcome learner fitting (x, t) -> y
            propensity_learner: the propensity learner fitting x -> p(T | X)
        """
        super().__init__(learner)

        if propensity_learner is None:
            self.propensity_learner = ElasticNetPropensityModel()
        else:
            assert hasattr(
                propensity_learner, "predict_proba"
            ), "propensity learner must have predict_proba method"

            self.propensity_learner = propensity_learner

    def __str__(self):
        """ Simple String Representation for logs and outputs"""
        return ("{}(regressor={}, propensity={})").format(
            self.__class__.__name__,
            self.learner.__class__.__name__,
            self.propensity_learner.__class__.__name__,
        )

    def fit(
        self,
        x: np.array,
        t: np.array,
        y: np.array,
        propensity: Optional[np.array] = None,
    ):
        """ Fits weighted regressor on the given samples

        If propensity scores are not given explicitly, the propensity regressor
        of the module is used

        Args:
            x: covariates
            t: treatment indicator
            y: factual outcomes
            propensity: propensity scores to be used
        """
        if propensity is not None:
            assert len(propensity) == len(t)
            ipt = 1 / propensity
        else:
            if isinstance(self.propensity_learner, ElasticNetPropensityModel):
                # Use special API of Elastic Net
                ipt = self.propensity_learner.fit_predict(x, t)
            else:
                # Use predict_proba of sklearn classifier
                self.propensity_learner.fit(x, t)
                ipt = 1 / self.propensity_learner.predict_proba(x)[:, 1]

        train = np.c_[x, t]
        self.learner.fit(train, y, sample_weight=ipt)
