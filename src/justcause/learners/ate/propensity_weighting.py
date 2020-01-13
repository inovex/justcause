from typing import Optional

import numpy as np

from ..propensity import estimate_propensities


class PSWEstimator(object):
    """Implements the simple propensity score weighting estimator

    To estimate the average treatment effect, the inverse propensity weighted
    mean outcomes of treated and control are calculated and subtracted.

    References:
        [1] http://freerangestats.info/blog/2017/04/09/propensity-v-regression

    """

    def __init__(self, propensity_learner=None, delta=0.001):
        self.propensity_learner = propensity_learner
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
        r"""Estimates average treatment effect of the given population

        The estimation for a finite sample is simply:

        .. math::
            \hat{\tau}_{IPW} = n^{-1} \sum_{i = 1}^n \frac{T_i Y_i}{\hat{p}(x)} -
                n^{-1} \sum_{i = 1}^n \frac{(1-T_i) Y_i}{1 - \hat{p}(x)}.

        Args:
            x: covariate matrix of the population of shape
                (num_instances, num_covariates)
            t: binary treatment indicator vector of shape (num_instances)
            y: factual outcome vector of shape (num_instances)
            propensities:

        Returns:
            ate: Estimate of the average treatment effect across the population

        """

        num_samples = x.shape[0]

        if propensities is None:
            if self.propensity_learner is None:
                propensities = estimate_propensities(x, t)
            else:
                self.propensity_learner.fit(x, t)
                propensities = self.propensity_learner.predict_proba(x)[:, 1]

        m1 = np.sum(((t * y + self.delta) / (propensities + self.delta)))
        m0 = np.sum((((1 - t) * y + self.delta) / (1 - propensities + self.delta)))
        return float((m1 - m0) / num_samples)
