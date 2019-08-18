import os

import numpy as np
import pandas as pd

from causaleval.data.data_provider import DataProvider
from causaleval import config
from utils import surface_plot, ite_plot, plot_y_dist, simple_comparison_mean


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


class IHDPReplicaProvider(DataProvider):

    def __init__(self,seed=0, train_size=0.8, setting="A"):
        self.setting = setting
        self.counter = 0
        super().__init__(seed, train_size)

    def __str__(self):
        return "IHDP-Replica"+self.setting

    def get_training_data(self, size=None):
        self.load_training_data()
        self.counter += 1
        return super(IHDPReplicaProvider, self).get_training_data()

    def load_training_data(self):
        if self.setting == "A":
            path = config.IHDP_REPLICA_PATH
        else:
            path = config.IHDP_REPLICA_PATH_SETTING_B

        dirname = os.path.dirname(__file__)
        filedir = os.path.join(dirname, path)
        all_files = os.listdir(filedir)

        if self.counter > 999:
            self.counter = 0 # reset counter

        fname = os.path.join(filedir, all_files[self.counter])
        data = pd.read_csv(fname)
        Y_0 = data['y.0'].values
        Y_1 = data['y.1'].values
        Y = data['y'].values
        T = data['z.r'].values
        X = data.drop(columns=['y.0', 'y.1', 'y', 'z.r']).values

        self.x = np.array(X)
        self.t = np.array(T)
        self.y = np.array(Y)
        self.y_0 = np.array(Y_0)
        self.y_1 = np.array(Y_1)
        union = np.c_[self.y_0, self.y_1]
        self.y_cf = np.array([row[int(1 - ix)] for row, ix in zip(union, self.t)])


if __name__ == '__main__':

    ihdp = IHDPDataProvider()
    surface_plot(ihdp.y_1, ihdp.y_0, ihdp.y, ihdp.y_cf, ihdp.x)
    ite_plot(ihdp.y_1, ihdp.y_0)
    plot_y_dist(ihdp.y, ihdp.y_cf)
    simple_comparison_mean(ihdp.y, ihdp.t)
    print('true: ', ihdp.get_true_ate())


