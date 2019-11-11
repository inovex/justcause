from typing import Optional

import numpy as np

from ..propensity import fit_predict_propensity, set_propensity_learner


class PSWEstimator:
    """ Implements the simple propensity score weighting estimator """

    def __init__(self, propensity_learner=None, delta=0.001):
        self.propensity_learner = set_propensity_learner(propensity_learner)

        self.delta = delta

    def __str__(self):
        return "{}(p_learner={})".format(
            self.__class__.__name__, self.propensity_learner.__class__.__name__
        )

    def estimate_ate(
        self,
        x: np.array,
        t: np.array,
        y: np.array,
        propensities: Optional[np.array] = None,
    ) -> float:
        """ Fits the models on given population and calculates the ATE"""

        num_samples = x.shape[0]

        if propensities is None:
            propensities = fit_predict_propensity(self.propensity_learner, x, t)

        m1 = np.sum(((t * y + self.delta) / (propensities + self.delta)))
        m0 = np.sum((((1 - t) * y + self.delta) / (1 - propensities + self.delta)))
        return float((m1 - m0) / num_samples)
