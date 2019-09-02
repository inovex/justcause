import os

import numpy as np
import pandas as pd

from causaleval.data.data_provider import DataProvider
from causaleval import config
from utils import surface_plot, ite_plot, plot_y_dist, simple_comparison_mean, true_ate_plot, true_ate_dist_plot


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
    """Using script-generated files"""

    def __init__(self,seed=0, train_size=0.8, setting="A"):
        self.setting = setting
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

        if self.counter > 110: # IHDP has 1000 replications at max
            self.counter = 0 # reset counter

        fname = os.path.join(filedir, all_files[self.counter])
        print(fname)
        data = pd.read_csv(fname)
        Y_0 = data["mu.0"].values
        Y_1 = data["mu.1"].values
        Y = data["y"].values
        T = data["z.r"].values
        X = data.drop(columns=['mu.0', 'mu.1', 'y', 'z.r']).values

        self.x = np.array(X)
        self.t = np.array(T)
        self.y = np.array(Y)
        self.y_0 = np.array(Y_0)
        self.y_1 = np.array(Y_1)
        union = np.c_[self.y_0, self.y_1]
        self.y_cf = np.array([row[int(1 - ix)] for row, ix in zip(union, self.t)])

class IHDPCfrProvider(DataProvider):
    """Using .npz file provided in [cfr-general] implementation here:

    https://github.com/clinicalml/cfrnet
    """
    def __init__(self,seed=0, train_size=1.0):
        """

        :param seed: random seed
        :param train_size: equal to 1, as test-data is provided manually
        """
        self.load_all() # Load set once
        super().__init__(seed, train_size)

    def __str__(self):
        return "IHDP-CFR"

    def get_training_data(self, size=None):
        self.counter += 1
        self.load_training_data()
        return super(IHDPCfrProvider, self).get_training_data()

    def get_test_data(self):
        """
        Manually provide test data from separate file"""
        return self.test['x'][:,:, self.counter], \
               self.test['t'][:, self.counter], \
               self.test['yf'][:, self.counter]

    def get_test_ite(self):
        """Manually provide true test results"""
        return self.test['mu1'][:, self.counter] - self.test['mu0'][:, self.counter]

    def load_all(self):
        fname = os.path.join(config.ROOT_DIR, "datasets/ihdp-cfr/train.npz")
        train = np.load(fname)
        fname = os.path.join(config.ROOT_DIR, "datasets/ihdp-cfr/test.npz")
        self.test = np.load(fname)
        self.train = train

        self.y1_all = train['mu1']
        self.y0_all = train['mu0']
        self.y_all = train['yf']
        self.ycf_all = train['ycf']
        self.t_all = train['t']
        self.x_all = train['x'] # has shape (672, 25, 100)



    def load_training_data(self):
        """Just updates the specific replications used for the next run"""

        if self.counter > 999: # This NPZ provides 1000 replications
            self.counter = 0 # reset counter

        self.x = self.x_all[:,:,self.counter]
        self.t = self.t_all[:, self.counter]
        self.y = self.y_all[:, self.counter]
        self.y_0 = self.y0_all[:, self.counter]
        self.y_1 = self.y1_all[:, self.counter]
        self.y_cf = self.ycf_all[:, self.counter]


if __name__ == '__main__':

    ihdp = IHDPReplicaProvider(setting='A')
    true_ates = []

    for i in range(110):
        ihdp.get_training_data() # to up the counter
        true_ates.append(ihdp.get_true_ate())


    true_ate_plot(true_ates, dataset='IHDP-Replica')
    true_ate_dist_plot(true_ates, dataset='IHDP-Replica')

