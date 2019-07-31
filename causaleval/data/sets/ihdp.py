import os

import numpy as np

from causaleval.data.data_provider import DataProvider
from causaleval import config

from itertools import cycle

class IHDPDataProvider(DataProvider):

    def __init__(self):
        super().__init__()
        self.x = None
        self.t = None
        self.y = None
        self.y_cf = None

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

        X = X[1:]
        self.x = np.array(X)
        self.t = np.array(T)
        self.y = np.array(Y)
        self.y_cf = np.array(Y_cf)
        union = np.c_[self.y, self.y_cf]
        self.treated_outcome = np.array([row[int(ix)] for row, ix in zip(union, self.t)])
        self.control_outcome = np.array([row[int(ix)] for row, ix in zip(union, (1 - self.t))])

    def get_training_data(self, size=None):
        if self.x is None:
            self.load_training_data()
        if size is None:
            return self.x, self.t, self.y
        else:
            sample = np.random.choice(self.x.shape[0], size)
            return self.x[sample], self.t[sample], self.y[sample]

    def get_true_ite(self, data=None):
        return self.treated_outcome - self.control_outcome

    def get_true_ate(self, subset=None):
        return np.mean(self.get_true_ite())


    def get_train_generator_single(self, random=False, replacement=False):
        """
        Return a cycle generator of single objects
        :param random:
        :param replacement:
        :return:
        """
        id_generator = cycle(range(self.x.shape[0]))
        while True:
            ID = next(id_generator)
            yield self.x[ID], self.t[ID], self.y[ID]


    def get_train_generator_batch(self, batch_size):
        num_batches = self.x.shape[0] / batch_size
        batch_id_generator = cycle(range(num_batches))

        while True:
            start = next(batch_id_generator) * batch_size
            end = next(batch_id_generator) * (batch_size + 1)
            yield self.x[start:end], self.t[start:end], self.y[start:end]




