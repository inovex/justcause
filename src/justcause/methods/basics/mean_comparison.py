import numpy as np

from ..causal_method import CausalMethod


class SimpleMeanComparison(CausalMethod):
    """
    Implements a generic T-learner :py:meth:`.fit()`

    :references:
    [T-Learner](﻿https://arxiv.org/pdf/1706.03461.pdf)
    """

    def __init__(self, seed=0):
        """

        :param seed: Random seed
        :param regressor: a sklearn regressor with methods `fit` and `predict`
        :param regressor_two: a sklearn regressor with methods `fit` and `predict`
        """
        super().__init__(seed)
        self.is_trained = False

    def __str__(self):
        return 'SimpleMeanComparison'

    def predict_ate(self, x, t=None, y=None):
        return self.ate

    def fit(self, x, t, y, refit=False) -> None:
        self.ate = np.mean(y[t==1]) - np.mean(y[t==0])

    def predict_ite(self, x, t=None, y=None):
        return np.full(x.shape[0], self.predict_ate(x, t, y))

