import copy
import numpy as np

from sklearn.calibration import CalibratedClassifierCV

from ..causal_method import CausalMethod
from ...utils import get_regressor_name


class PropensityScoreWeighting(CausalMethod):

    def __init__(self, propensity_regressor):
        super().__init__()
        self.given_regressor = propensity_regressor
        self.propensity_regressor = propensity_regressor
        self.delta = 0.0001

    def requires_provider(self):
        return False

    def predict_ate(self, x, t=None, y=None):
        # Predict ATE always for training set, thus test set evaluation is pretty bad
        if t is not None and y is not None:
            # Fit for requested set
            # self.fit(x, t, y)
            self.x = x
            self.t = t
            self.y = y

        prop = self.propensity_regressor.predict_proba(self.x)[:, 1]
        m1 = np.sum(((self.t*self.y + self.delta)/(prop + self.delta)) / self.x.shape[0])
        m0 = np.sum((((1-self.t)*self.y + self.delta)/(1-prop + self.delta)) / self.x.shape[0])
        return m1 - m0

    def fit(self, x, t, y, refit=False) -> None:
        # Fit propensity score model
        self.x = x
        self.t = t
        self.y = y
        self.propensity_regressor = CalibratedClassifierCV(self.given_regressor)
        self.propensity_regressor.fit(x, t)


    def __str__(self):
        return "PropensityScoreWeighting - " + get_regressor_name(self.given_regressor)

    def predict_ite(self, x, t=None, y=None):
        # Broadcast ATE to all instances
        return np.full(x.shape[0], self.predict_ate(x, t, y))


