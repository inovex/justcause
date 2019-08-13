import os

import numpy as np

from causaleval.data.data_provider import DataProvider
from causaleval import config

from itertools import cycle

class IHDPDataProvider(DataProvider):

    def __init__(self,seed=0, train_size=0.8):
        super().__init__(seed, train_size)

    def __str__(self):
        return "IHDP"

    def load_training_data(self):
        path = config.IHDP_PATH
        dirname = os.path.dirname(__file__)
        filedir = os.path.join(dirname, path)
        all_files = os.listdir(filedir)

        T, Y, Y_cf, X = np.array([]), np.array([]), np.array([]), np.empty((1,25))

        for file in all_files:
            filepath = os.path.join(filedir, file)
            data = np.loadtxt(filepath, delimiter=',')
            T, Y, Y_cf = np.append(T, data[:, 0]), np.append(Y,data[:, 1][:, np.newaxis]), np.append(Y_cf, data[:, 2][:, np.newaxis])
            X = np.append(X, data[:, 5:], axis=0)
            break # only use one of the different data set versions


        X = X[1:]
        self.x = np.array(X)
        self.t = np.array(T)
        self.y = np.array(Y)
        self.y_cf = np.array(Y_cf)
        union = np.c_[self.y, self.y_cf]
        self.y_1 = np.array([row[int(1 - ix)] for row, ix in zip(union, self.t)])
        self.y_0 = np.array([row[int(ix)] for row, ix in zip(union, self.t)])

    def get_num_covariates(self):
        return 25





