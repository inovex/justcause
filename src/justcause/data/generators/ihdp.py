import os

import numpy as np
import pandas as pd

from ..data_provider import DataProvider
from ... import utils

import config


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def multi_modal_effect(X):
    prob = (sigmoid(X[:, 7]) > 0.5)
    return np.random.normal((3*prob)+1*(1-prob), 0.1, size=len(X)) # Explicitly multimodal


def exponential_effect(X):
    return np.exp(1 + sigmoid(X[:, 7])) # use birth weight


def small_exponential_effect(X):
    return np.exp(sigmoid(X[:, 7]) - 1) # use birth weight


class IHDPGenerator(DataProvider):

    def __init__(self,seed=0, setting='multi-modal', base='gamma'):
        self.setting = setting
        self.base = base
        super().__init__(seed)

    def __str__(self):
        return "IHDP-"+self.setting+"-"+self.base

    def get_training_data(self, size=None):
        self.load_training_data()
        self.counter += 1
        return super(IHDPGenerator, self).get_training_data()


    def load_training_data(self):

        path = config.IHDP_REPLICA_PATH
        dirname = os.path.dirname(__file__)
        filedir = os.path.join(dirname, path)
        all_files = os.listdir(filedir)

        if self.counter > 110: # IHDP has 1000 replications at max
            self.counter = 0 # reset counter


        fname = os.path.join(filedir, all_files[self.counter])
        data = pd.read_csv(fname)
        covariates_df = data.drop(columns=['mu.0', 'mu.1', 'y', 'z.r'])
        X = covariates_df.values


        if self.base == 'gamma':
            Y_0 = np.random.gamma(0.2, 1, size=len(X))
        else:
            Y_0 = np.random.normal(X[:, 0]+1, 0.2, size=len(X))

        if self.setting == 'multi-modal':
            Y_1 = Y_0 + multi_modal_effect(X)
            val = X[:, 7]
        elif self.setting == 'small-exponential':
            Y_1 = Y_0 + small_exponential_effect(X)
            val = X[:, 7]
        else:
            Y_1 = Y_0 + exponential_effect(X)
            val = X[:, 7]

        self.x = X[:, 0:8]
        self.y_1 = Y_1
        self.y_0 = Y_0

        self.t = np.random.binomial(1, p=sigmoid(val), size=len(X))
        union = np.c_[self.y_0, self.y_1]
        self.y_cf = np.array([row[int(1 - ix)] for row, ix in zip(union, self.t)])
        self.y = np.array([row[int(ix)] for row, ix in zip(union, self.t)])


if __name__ == '__main__':
    ihdp = IHDPGenerator(setting='multi-modal', base='non-gamma')
    ihdp.load_training_data()
    utils.confounder_outcome_plot(ihdp.x[:, 7], ihdp.y_1 - ihdp.y_0, dataset='IDHP-Multi-Modal')
    utils.dist_plot(ihdp.x[:, 7], method_name='ihdp-confounder')
