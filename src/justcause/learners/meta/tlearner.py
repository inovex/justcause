import copy
from typing import Optional, Tuple, Union

import numpy as np
from causalml.propensity import ElasticNetPropensityModel
from sklearn.linear_model import LassoLars

from ..utils import replace_factual_outcomes

#: Type alias for predict_ite return type
SingleComp = Union[Tuple[np.array, np.array, np.array], np.array]


class BaseTLearner:
    """ Base class for all T-Learners; bundles duplicate code

    Defaults to a LassoLars learner for both treated and control

    """

    def __init__(self, learner=None, learner_c=None, learner_t=None, random_state=None):
        """
        Takes either one base learner for both or two specific base learners

        Args:
            learner: base learner for treatment and control outcomes
            learner_c: base learner for control outcome
            learner_t: base learner for treatment  outcome
            random_state: random state; currently unused
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

        self.random_state = random_state

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """ Simple string representation for logs and outputs"""
        return "{}(control={}, treated={})".format(
            self.__class__.__name__,
            self.learner_c.__class__.__name__,
            self.learner_t.__class__.__name__,
        )

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

    def predict_ate(self, x: np.array, t: np.array = None, y: np.array = None) -> float:
        return float(np.mean(self.predict_ite(x, t, y)))


class TLearner(BaseTLearner):
    """
    Implements a generic T-learner

    References:
        [1] S. R. Künzel, J. S. Sekhon, P. J. Bickel, and B. Yu,
        “Meta-learners for Estimating Heterogeneous Treatment Effects
            using Machine Learning,” 2019.
    """

    def fit(self, x: np.array, t: np.array, y: np.array) -> None:
        self.learner_c.fit(x[t == 0], y[t == 0])
        self.learner_t.fit(x[t == 1], y[t == 1])


class WeightedTLearner(BaseTLearner):
    """
    Implements a weighted generic T-Learner

    Propensity learner defaults to the ElasticNetPropensityModel proposed
    in CausalML. Otherwise requires the propensity learner to have a
    predict_proba method

    References:
        CausalML Framework `on Github <https://github.com/uber/causalml/>'_.

        [1] S. R. Künzel, J. S. Sekhon, P. J. Bickel, and B. Yu,
        “Meta-learners for Estimating Heterogeneous Treatment Effects
            using Machine Learning,” 2019.

    """

    def __init__(
        self,
        learner=None,
        learner_c=None,
        learner_t=None,
        propensity_learner=None,
        random_state=None,
    ):
        """
        Args:
            learner: base learner with parameter 'sample_weight' in fit()
            learner_c: specific learner for control outcome
            learner_t: specific learner for control outcome
            propensity_learner: calibrated classifier for propensity estimation
                must have 'predict_proba'
            random_state:
        """
        super().__init__(learner, learner_c, learner_t, random_state)
        if propensity_learner is None:
            self.propensity_learner = ElasticNetPropensityModel()
        else:
            assert hasattr(
                propensity_learner, "predict_proba"
            ), "propensity learner must have predict_proba method"

            self.propensity_learner = propensity_learner

    def __str__(self):
        """ Simple string representation for logs and outputs"""
        return "{}(control={}, treated={}, propensity={})".format(
            self.__class__.__name__,
            self.learner_c.__class__.__name__,
            self.learner_t.__class__.__name__,
            self.propensity_learner.__class__.__name__,
        )

    def fit(
        self,
        x: np.array,
        t: np.array,
        y: np.array,
        propensity: Optional[np.array] = None,
    ) -> None:
        """ Fits the T-learner with weighted samples

        If propensity scores are not given explicitly, the propensity learner
        of the module is used

        Args:
            x: covariates
            t: treatment indicator
            y: factual outcomes
            propensity: propensity scores to be used
        """
        if propensity is None:
            if isinstance(self.propensity_learner, ElasticNetPropensityModel):
                propensity = self.propensity_learner.fit_predict(x, t)
            else:
                self.propensity_learner.fit(x, t)
                propensity = self.propensity_learner.predict_proba(x)[:, 1]

        ipt = 1 / propensity
        self.learner_c.fit(x[t == 0], y[t == 0], sample_weight=ipt[t == 0])
        self.learner_t.fit(x[t == 1], y[t == 1], sample_weight=ipt[t == 1])
