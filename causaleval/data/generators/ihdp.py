import os

import numpy as np
import pandas as pd

from data.data_provider import DataProvider
from utils import ite_plot, surface_plot, plot_y_dist, simple_comparison_mean
import config

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def multi_modal_effect(X):
    prob = np.random.binomial(1, p=sigmoid(X[:, 1]), size=len(X))
    return np.random.normal((3*prob)+1*(1-prob), 0.1, size=len(X)) # Explicitly multimodal

def exponential_effect(X):
    return np.exp(1 + sigmoid(X[:, 1]))

class IHDPGenerator(DataProvider):

    def __init__(self,seed=0):
        super().__init__(seed)

    def __str__(self):
        return "IHDP-multi-modal"

    def load_training_data(self):

        path = config.IHDP_REPLICA_PATH
        dirname = os.path.dirname(__file__)
        filedir = os.path.join(dirname, path)
        all_files = os.listdir(filedir)

        fname = os.path.join(filedir, all_files[0])
        data = pd.read_csv(fname)
        covariates_df = data.drop(columns=['mu.0', 'mu.1', 'y', 'z.r'])
        X = covariates_df.values

        Y_0 = np.random.gamma(0.2, 1, size=len(X))
        Y_1 = Y_0 + multi_modal_effect(X)
        self.x = X
        self.y_1 = Y_1
        self.y_0 = Y_0

        # val = (1/3)*X[:, 5]*(2/3)*X[:, 0] + 0.5*(X[:, 16]) - 0.1

        val = X[:, 1]
        self.t = np.random.binomial(1, p=sigmoid(val), size=len(X))
        union = np.c_[self.y_0, self.y_1]
        self.y_cf = np.array([row[int(1 - ix)] for row, ix in zip(union, self.t)])
        self.y = np.array([row[int(ix)] for row, ix in zip(union, self.t)])


if __name__ == '__main__':
    ihdp = IHDPGenerator()
    ihdp.load_training_data()
    surface_plot(ihdp.y_1, ihdp.y_0, ihdp.y, ihdp.y_cf, ihdp.x)
    ite_plot(ihdp.y_1, ihdp.y_0)
    plot_y_dist(ihdp.y, ihdp.y_cf)
    simple_comparison_mean(ihdp.y, ihdp.t)
    print('true: ', np.mean(ihdp.get_true_ite()))
