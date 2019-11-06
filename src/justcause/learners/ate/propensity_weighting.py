import numpy as np

from ..utils import fit_predict_propensity, set_propensity_learner


class PSWEstimator:
    """ Implements the simple propensity score weighting estimator """

    def __init__(self, propensity_learner=None):
        self.propensity_learner = set_propensity_learner(propensity_learner)

        # TODO: Not clean here
        self.delta = 0.001

    def __str__(self):
        return "{}(p_learner={})".format(
            self.__class__.__name__, self.propensity_learner.__class__.__name__
        )

    def predict_ate(self, x, t, y, propensity=None):
        """ Fits the models on given population and calculates the ATE"""
        # TODO: Discuss: Out-of-sample prediction makes little sense here

        num_samples = x.shape[0]

        if propensity is None:
            propensity = fit_predict_propensity(self.propensity_learner, x, t)

        m1 = np.sum(((t * y + self.delta) / (propensity + self.delta)))
        m0 = np.sum((((1 - t) * y + self.delta) / (1 - propensity + self.delta)))
        return (m1 - m0) / num_samples

    def fit(self, x, t, y) -> None:
        """Shell method to avoid errors"""
        # TODO: Discuss use and convention of ATE learners
        pass
