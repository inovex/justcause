import numpy as np
from learners.causal_method import CausalMethod

from justcause.utils import get_regressor_name


class SLearner(CausalMethod):
    """
    Implements a generic S-Learner

    :ref:
    [S-Learner](﻿https://arxiv.org/pdf/1706.03461.pdf)
    """

    def __init__(self, regressor, seed=0):
        """

        :param regressor: a sklearn regressor with learners `fit` and `predict`
        """
        super().__init__(seed)
        self.regressor = regressor
        self.is_trained = False

    def __str__(self):
        return "S-Learner - " + get_regressor_name(self.regressor)

    @staticmethod
    def union(x, t):
        return np.c_[x, t]

    def predict_ite(self, x, t=None, y=None):
        return self.regressor.predict(
            self.union(x, np.ones(x.shape[0]))
        ) - self.regressor.predict(self.union(x, np.zeros(x.shape[0])))

    def predict_ate(self, x, t=None, y=None):
        return np.mean(self.predict_ite(x))

    def fit(self, x, t, y, refit=False) -> None:
        train = self.union(x, t)
        self.regressor.fit(train, y)
        self.is_trained = True


class WeightedSLearner(CausalMethod):
    """
    Implements a generic S-Learner

    :ref:
    [S-Learner](﻿https://arxiv.org/pdf/1706.03461.pdf)
    """

    def __init__(self, propensity_regressor, regressor, dgp, seed=0):
        """

        :param regressor: a sklearn regressor with learners `fit` and `predict`
        """
        super().__init__(seed)
        self.propensity_regressor = propensity_regressor
        self.regressor = regressor
        self.dgp = dgp
        self.is_trained = False

    def __str__(self):
        return "Weighted S-Learner - " + get_regressor_name(self.regressor)

    @staticmethod
    def union(x, t):
        return np.c_[x, t]

    def predict_ite(self, x, t=None, y=None):
        return self.regressor.predict(
            self.union(x, np.ones(x.shape[0]))
        ) - self.regressor.predict(self.union(x, np.zeros(x.shape[0])))

    def predict_ate(self, x, t=None, y=None):
        return np.mean(self.predict_ite(x))

    def fit(self, x, t, y, refit=False) -> None:
        self.propensity_regressor.fit(x, t)
        prob = self.propensity_regressor.predict_proba(x)
        prob = self.dgp.get_train_propensity()

        weights = 1 / prob
        train = self.union(x, t)
        self.regressor.fit(train, y, sample_weight=weights)
        self.is_trained = True
