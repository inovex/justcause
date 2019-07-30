import copy
import numpy as np
from causaleval.methods.causal_method import CausalMethod

class SingleOutcomeRegression(CausalMethod):
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
        return "SingleOutcomeRegression using " + str(self.regressor)

    @staticmethod
    def union(x, t):
        return np.c_[x, t]

    def predict_ite(self, x):
        if self.is_trained:
            return self.regressor.predict(self.union(x, np.ones(x.shape[0]))) - self.regressor.predict(self.union(x, np.zeros(x.shape[0])))
        else:
            raise AssertionError('Must fit method before prediction')

    def predict_ate(self, x):
        return np.mean(self.predict_ite(x))

    def fit(self, x, t, y, refit=False) -> None:
        train = self.union(x, t)
        self.regressor.fit(train, y)
        self.is_trained = True



class DoubleOutcomeRegression(CausalMethod):
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
        return "DoubleOutcomeRegression using " + str(self.regressor_one) + " and " + str(self.regressor_two)

    def predict_ate(self, x):
        return np.mean(self.predict_ite(x))

    def fit(self, x, t, y, refit=False) -> None:
        x_treatment = x[t == 1]
        x_control = x[t == 0]
        self.regressor_one.fit(x_treatment, y[t == 1])
        self.regressor_two.fit(x_control, y[t == 0])

    def predict_ite(self, x):
        return self.regressor_one.predict(x) - self.regressor_two.predict(x)

