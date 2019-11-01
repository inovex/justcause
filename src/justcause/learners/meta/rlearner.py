import numpy as np
from causalml.inference.meta import BaseRRegressor
from causalml.propensity import ElasticNetPropensityModel
from sklearn.linear_model import LassoLars


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

    def __init__(self, learner=None, outcome_learner=None, effect_learner=None):
        if learner is None and (outcome_learner is None and effect_learner is None):
            learner = LassoLars()

        self.model = BaseRRegressor(learner, outcome_learner, effect_learner)

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
        # TODO: Allow to pass a specific propensity learner to the class,
        #       see WeightedTLearner
        if p is None:
            p_learner = ElasticNetPropensityModel()
            p = p_learner.fit_predict(x, t)

        self.model.fit(x, p, t, y)

    def predict_ite(
        self, x: np.array, t: np.array = None, y: np.array = None
    ) -> np.array:
        """ Predicts ITE for given samples; ignores the factual outcome and treatment"""

        assert t is None and y is None, "The R-Learner does not use factual outcomes"
        return self.model.predict(x)

    def predict_ate(self, x: np.array, t: np.array, y: np.array) -> float:
        """ Predicts ATE for given samples; ignores the factual outcome and treatment"""
        return float(np.mean(self.predict_ite(x, t, y)))
