import copy
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from ..causal_method import CausalMethod
from ...utils import get_regressor_name


class DoubleRobust(CausalMethod):

    def __init__(self, propensity_regressor, outcome_regressor):
        super().__init__()
        self.given_regressor = propensity_regressor
        self.propensity_regressor = propensity_regressor
        self.outcome_regressor = outcome_regressor
        self.outcome_regressor_ctrl = copy.deepcopy(outcome_regressor)
        self.delta = 0.0001

    def requires_provider(self):
        return False

    def predict_ate(self, x, t=None, y=None):
        # Predict ATE always for training set, thus test set evaluation is pretty bad
        if t is not None and y is not None:
            # Fit for requested set
            self.fit(x, t, y)
            self.x = x
            self.t = t
            self.y = y

        prop = self.propensity_regressor.predict_proba(x)[:, 1]
        dr1 = np.sum(((self.t*self.y)/ (prop + self.delta)) - ((self.t - prop + self.delta)/(prop +self.delta))*self.outcome_regressor.predict(x)) / x.shape[0]
        dr0 = np.sum(((1 - self.t)*self.y/(1- prop + self.delta)) - ((self.t - prop + self.delta)/(1-prop + self.delta))*self.outcome_regressor_ctrl.predict(x))/ x.shape[0]
        return dr1 - dr0

    def fit(self, x, t, y, refit=False) -> None:
        # Fit propensity score model
        self.t = t
        self.y = y
        self.propensity_regressor = CalibratedClassifierCV(self.given_regressor)
        self.propensity_regressor.fit(x, t)
        # Fit the two outcome models
        self.outcome_regressor.fit(x[t==1], y[t==1])
        self.outcome_regressor_ctrl.fit(x[t==0], y[t==0])


    def __str__(self):
        return "DoubleRobustEstimator - P: " +  get_regressor_name(self.given_regressor) + " O: " + get_regressor_name(self.outcome_regressor)

    def predict_ite(self, x, t=None, y=None):
        # Broadcast ATE to all instances
        return np.full(x.shape[0], self.predict_ate(x, t, y))



