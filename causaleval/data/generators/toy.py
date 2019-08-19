
import numpy as np

import config
from causaleval.data.data_provider import DataProvider

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SWagerDataProvider(DataProvider):
    """
    Implements the toy examples proposed by S. Wager in a personal communication
    to show the efficacy of Causal Forests
    """

    def __init__(self,seed=0, setting="simple"):
        self.setting = setting
        super().__init__(seed)

    def __str__(self):
        return "S.Wager-Toy-"+self.setting

    def load_training_data(self):

        n = 1000
        p = 10

        rand = np.random.normal(0, 1, size=n*p)
        X = rand.reshape((n, p))
        X_test = rand.reshape((n, p))

        if self.setting == 'simple':
            ite = X[:, 2] + X[:, 3]
            ite_test = X_test[:, 2] + X_test[:, 3]
            T = np.random.binomial(1, 0.5, size=n) # random assignment
            Y = sigmoid(X[:, 1]) + ite * T + np.random.normal(size=n) # add some noise
        else:
            ite = sigmoid(X[:, 2] + X[:, 3])/2
            ite_test = sigmoid(X_test[:, 2] + X_test[:, 3]) / 2
            T = np.random.binomial(1, sigmoid(X[:, 1]), size=n) # random assignment
            Y = sigmoid(X[:, 1]) + ite * T + np.random.normal(size=n) # add some noise

        self.x = X
        self.y = Y
        self.t = T
        self.x_test = X_test
        self.true_train_ite = ite
        self.true_test_ite = ite_test

    def get_test_data(self):
        return self.x_test

    def get_test_ite(self):
        return self.true_test_ite

    def get_training_data(self, size=None):
        """Always return full dataset"""
        return self.x, self.t, self.y

    def get_train_ite(self, subset=False):
        """Always return full dataset"""
        return self.true_train_ite




import os
os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'


if __name__ == "__main__":

    from methods.causal_forest import CausalForest
    from methods.basics.outcome_regression import DoubleOutcomeRegression
    from sklearn.ensemble import RandomForestRegressor
    from metrics import StandardEvaluation

    rf = DoubleOutcomeRegression(RandomForestRegressor())
    cf = CausalForest()

    gen = SWagerDataProvider(setting='hard')
    gen.load_training_data()

    cf.fit(gen.x, gen.t, gen.y)
    rf.fit(gen.x, gen.t, gen.y)

    cf_ite = cf.predict_ite(gen.x_test)
    rf_ite = rf.predict_ite(gen.x_test)

    print('cf: ', StandardEvaluation.pehe_score(gen.true_test_ite, cf_ite))
    print('rf: ', StandardEvaluation.pehe_score(gen.true_test_ite, rf_ite))


