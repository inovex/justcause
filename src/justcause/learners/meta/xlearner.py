from typing import Optional

import numpy as np
from sklearn.linear_model import LassoLars

from justcause.learners.utils import replace_factual_outcomes

from ..propensity import estimate_propensities


class XLearner:
    """Wrapper of the BaseXRegressor from causalml

    Defaults to `sklearn.linear_model.LassoLars` as a base learner if not specified
    otherwise. Allows to either specify one learner for all or four distinct learners
    for the tasks
        - outcome control
        - outcome treated
        - effect control
        - effect treated

    References:
        [1] CausalML Framework `on Github <https://github.com/uber/causalml/>'_.

        [2] S. R. Künzel, J. S. Sekhon, P. J. Bickel, and B. Yu,
            “Meta-learners for Estimating Heterogeneous
            Treatment Effects using Machine Learning,” 2019.

    """

    def __init__(
        self,
        learner=None,
        outcome_learner_c=None,
        outcome_learner_t=None,
        effect_learner_c=None,
        effect_learner_t=None,
    ):
        """Setup a XLearner

        All learners must have ``fit(x, y)`` and ``predict(x)`` methods.

        Args:
            learner: default learner for all roles
            outcome_learner_c: specific learner for control outcome function
            outcome_learner_t: specific learner for treated outcome function
            effect_learner_c: specific learner for treated effect
            effect_learner_t: specific learner for control effect
        """
        from causalml.inference.meta import BaseXRegressor

        if (learner is not None) or (
            (outcome_learner_c is not None)
            and (outcome_learner_t is not None)
            and (effect_learner_c is not None)
            and (effect_learner_t is not None)
        ):
            self.model = BaseXRegressor(
                learner,
                outcome_learner_c,
                outcome_learner_t,
                effect_learner_c,
                effect_learner_t,
            )
        else:
            # Assign default learner
            learner = LassoLars()
            self.model = BaseXRegressor(
                learner,
                outcome_learner_c,
                outcome_learner_t,
                effect_learner_c,
                effect_learner_t,
            )

    def __str__(self):
        """Simple string representation for logs and outputs"""
        return "{}(outcome_c={}, outcome_t={}, effect_c={}, effect_t={})".format(
            self.__class__.__name__,
            self.model.model_mu_c.__class__.__name__,
            self.model.model_mu_t.__class__.__name__,
            self.model.model_tau_c.__class__.__name__,
            self.model.model_tau_t.__class__.__name__,
        )

    def __repr__(self):
        return self.__str__()

    def fit(self, x: np.array, t: np.array, y: np.array) -> None:
        """Fits the RLearner on given samples

        Args:
            x: covariate matrix of shape (num_instances, num_features)
            t: treatment indicator vector, shape (num_instances)
            y: factual outcomes, (num_instances)

        """
        self.model.fit(x, t, y)

    def predict_ite(
        self,
        x: np.array,
        t: np.array = None,
        y: np.array = None,
        p: Optional[np.array] = None,
        return_components: bool = False,
        replace_factuals: bool = False,
    ) -> np.array:
        """Predicts ITE for the given population

        If propensities ``p`` are not given, they are estimated using the default
        implementation `justcause.learners.propensities.estimate_propensities`

        Args:
            x: covariates
            t: treatment indicator
            y: factual outcomes
            p: propensity scores
            return_components: whether to return Y(1) and Y(0) for all instances or not
            replace_factuals: whether to replace predicted outcomes with the factual
                outcomes where applicable

        Returns:
            the ITE prediction either with or without components
        """
        if p is None:
            # Set default propensity, because CausalML currently requires it
            p = estimate_propensities(x, t)

        if return_components:
            ite, y_0, y_1 = self.model.predict(x, p, t, y, return_components=True)
            if t is not None and y is not None and replace_factuals:
                y_0, y_1 = replace_factual_outcomes(y_0, y_1, y, t)
            return ite.flatten(), y_0.flatten(), y_1.flatten()
        else:
            return self.model.predict(x, p, t, y, return_components=False).flatten()

    def estimate_ate(
        self,
        x: np.array,
        t: np.array,
        y: np.array,
        propensities: Optional[np.array] = None,
    ) -> float:
        """Predicts ATE for given samples as mean of ITE predictions"""
        self.fit(x, t, y)
        return float(np.mean(self.predict_ite(x, t, y, propensities)))
