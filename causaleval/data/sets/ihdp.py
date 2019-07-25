import os

import numpy as np

from causaleval.data.data_provider import DataProvider
from causaleval import config

class IHDPDataProvider(DataProvider):

    def __init__(self):
        super().__init__()
        self.x = None
        self.t = None
        self.y = None
        self.y_cf = None

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

        X = X[1:]
        self.x = X
        self.t = T
        self.y = Y
        self.y_cf = Y_cf

    def get_training_data(self):
        if self.x is None:
            self.load_training_data()
        return self.x, self.t, self.y

    def get_true_ite(self, data=None):
        pass

    def get_true_ate(self, subset=None):
        pass




