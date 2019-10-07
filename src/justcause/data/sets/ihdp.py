import os

import numpy as np

from ..data_provider import DataProvider


class IHDPCfrProvider(DataProvider):
    """Using .npz file provided in [cfr-general] implementation here:

    https://github.com/clinicalml/cfrnet
    """

    def __init__(self, seed=0, train_size=1.0):
        """

        :param seed: random seed
        :param train_size: equal to 1, as test-data is provided manually
        """
        self.load_all()  # Load set once
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
        return (
            self.test["x"][:, :, self.counter],
            self.test["t"][:, self.counter],
            self.test["yf"][:, self.counter],
        )

    def get_test_ite(self):
        """Manually provide true test results"""
        return self.test["mu1"][:, self.counter] - self.test["mu0"][:, self.counter]

    def load_all(self):
        path = "/Users/MaximilianFranz/Documents/ba/eval/justcause/datasets/ihdp-cfr"
        fname = os.path.join(path, "train.npz")
        train = np.load(fname)
        fname = os.path.join(path, "test.npz")
        self.test = np.load(fname)
        self.train = train

        self.y1_all = train["mu1"]
        self.y0_all = train["mu0"]
        self.y_all = train["yf"]
        self.ycf_all = train["ycf"]
        self.t_all = train["t"]
        self.x_all = train["x"]  # has shape (672, 25, 100)

    def load_training_data(self):
        """Just updates the specific replications used for the next run"""

        if self.counter > 999:  # This NPZ provides 1000 replications
            self.counter = 0  # reset counter

        self.x = self.x_all[:, :, self.counter]
        self.t = self.t_all[:, self.counter]
        self.y = self.y_all[:, self.counter]
        self.y_0 = self.y0_all[:, self.counter]
        self.y_1 = self.y1_all[:, self.counter]
        self.y_cf = self.ycf_all[:, self.counter]
