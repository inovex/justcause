import copy
import numpy as np

from ..causal_method import CausalMethod
from ...utils import get_regressor_name


class SLearner(CausalMethod):
    """
    Implements a generic S-Learner

    :ref:
    [S-Learner](﻿https://arxiv.org/pdf/1706.03461.pdf)
    """

    def __init__(self, regressor, seed=0):
        """

        :param regressor: a sklearn regressor with methods `fit` and `predict`
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
        return self.regressor.predict(self.union(x, np.ones(x.shape[0]))) - self.regressor.predict(self.union(x, np.zeros(x.shape[0])))

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

        :param regressor: a sklearn regressor with methods `fit` and `predict`
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
        return self.regressor.predict(self.union(x, np.ones(x.shape[0]))) - self.regressor.predict(self.union(x, np.zeros(x.shape[0])))

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

class WeightedTLearner(CausalMethod):
    """
    Implements a generic S-Learner

    :ref:
    [S-Learner](﻿https://arxiv.org/pdf/1706.03461.pdf)
    """

    def __init__(self, propensity_regressor, regressor, dgp, seed=0):
        """

        :param regressor: a sklearn regressor with methods `fit` and `predict`
        """
        super().__init__(seed)
        self.propensity_regressor = propensity_regressor
        self.regressor_one = regressor
        self.regressor_two = copy.deepcopy(regressor)
        self.dgp = dgp
        self.is_trained = False

    def __str__(self):
        return "Weighted T-Learner - " + get_regressor_name(self.regressor_one)

    @staticmethod
    def union(x, t):
        return np.c_[x, t]

    def predict_ite(self, x, t=None, y=None):
        return self.regressor_one.predict(x) - self.regressor_two.predict(x)

    def fit(self, x, t, y, refit=False) -> None:
        self.propensity_regressor.fit(x, t)
        prob = self.propensity_regressor.predict_proba(x)
        prob = self.dgp.get_train_propensity()
        weights = 1 / prob
        x_treatment = x[t == 1]
        x_control = x[t == 0]
        self.regressor_one.fit(x_treatment, y[t == 1], sample_weight=weights[t==1])
        self.regressor_two.fit(x_control, y[t == 0], sample_weight=weights[t==0])


class TLearner(CausalMethod):
    """
    Implements a generic T-learner :py:meth:`.fit()`

    :references:
    [T-Learner](﻿https://arxiv.org/pdf/1706.03461.pdf)
    """

    def __init__(self, regressor, regressor_two=None, seed=0):
        """

        :param seed: Random seed
        :param regressor: a sklearn regressor with methods `fit` and `predict`
        :param regressor_two: a sklearn regressor with methods `fit` and `predict`
        """
        super().__init__(seed)
        self.regressor_one = regressor
        self.is_trained = False

        if regressor_two is None:
            # Create a full, independent copy of the first regressor
            self.regressor_two = copy.deepcopy(regressor)
        else:
            self.regressor_two = regressor_two

    def __str__(self):
        return "T-Learner - " \
               + get_regressor_name(self.regressor_one) \
               + " & " + get_regressor_name(self.regressor_two)

    def predict_ate(self, x, t=None, y=None):
        return np.mean(self.predict_ite(x))

    def fit(self, x, t, y, refit=False) -> None:
        x_treatment = x[t == 1]
        x_control = x[t == 0]
        self.regressor_one.fit(x_treatment, y[t == 1])
        self.regressor_two.fit(x_control, y[t == 0])

    def predict_ite(self, x, t=None, y=None):
        return self.regressor_one.predict(x) - self.regressor_two.predict(x)

